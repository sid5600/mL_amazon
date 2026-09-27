"""
features.py — Pairwise Feature Engineering
============================================
Author : Member 4 (Feature Engineering + Experiments)
Challenge: Amazon ML Challenge — Business Entity Resolution

PURPOSE
-------
Converts (S1_id, candidate_id) pairs into a rich numerical feature vector
that a downstream ML model can use for binary classification (match / no-match).

FEATURE GROUPS
--------------
1. Name similarity      — Jaccard, Jaro-Winkler, token overlap, TF-IDF cosine,
                          char n-gram Jaccard
2. Address similarity   — same metrics on address fields
3. Name x Address cross — token overlap between name tokens and address tokens
4. Structural           — country match (bool), name/address length ratios,
                          token count ratios
5. Embeddings (optional)— multilingual sentence-transformers cosine similarity
                          (set USE_EMBEDDINGS=True; requires ~2 GB RAM per batch)

INPUTS
------
- output/candidate_pairs.tsv        (from Member 3 / blocking)
- <PROCESSED_DIR>/train_source1_processed.tsv
- <PROCESSED_DIR>/train_source2_processed.tsv
- <PROCESSED_DIR>/train_source3_processed.tsv
- <PROCESSED_DIR>/test_source1_processed.tsv
- <PROCESSED_DIR>/test_source2_processed.tsv
- <PROCESSED_DIR>/test_source3_processed.tsv
- dataset/train/train_ground_truth.tsv   (for label assignment during training)

OUTPUTS
-------
- feature DataFrame (in-memory) with columns:
    s1_id | cand_id | label (train only) | f_* (feature columns)

HOW TO USE
----------
>>> from src.features import FeatureBuilder
>>> fb = FeatureBuilder()
>>> fb.load_entity_data()
>>> train_df = fb.build_train_features()          # labelled pairs
>>> test_df  = fb.build_test_features()           # unlabelled pairs
"""

from __future__ import annotations

import gc
import math
import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  -- adjust paths to match where your processed files live
# =============================================================================

# Root of the repo (one level above src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Where Member 2 saved the processed TSVs
PROCESSED_DIR = Path(os.environ.get(
    "AMAZON_ML_PROCESSED_DIR",
    str(_REPO_ROOT / "data" / "processed")   # default: data/processed/
))

# Raw dataset dir (only needed for ground-truth label loading)
DATASET_DIR = Path(os.environ.get(
    "AMAZON_ML_DATASET_DIR",
    str(_REPO_ROOT / "data" / "raw")          # default: data/raw/
))

# Candidate pairs produced by Member 3 (blocking)
CANDIDATE_PAIRS_PATH = _REPO_ROOT / "output" / "candidate_pairs.tsv"

# Where to write intermediate features (optional caching)
FEATURES_CACHE_DIR = _REPO_ROOT / "output" / "features_cache"
FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Toggle sentence-transformer embeddings (requires: pip install sentence-transformers)
USE_EMBEDDINGS = False
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # ~470 MB; fast + multilingual
EMBEDDING_BATCH_SIZE = 512

# Processed file names (output from preprocessing.py)
PROCESSED_FILES = {
    "train_s1": "train_source1_processed.tsv",
    "train_s2": "train_source2_processed.tsv",
    "train_s3": "train_source3_processed.tsv",
    "test_s1":  "test_source1_processed.tsv",
    "test_s2":  "test_source2_processed.tsv",
    "test_s3":  "test_source3_processed.tsv",
}

GROUND_TRUTH_PATH = DATASET_DIR / "train" / "train_ground_truth.tsv"

# =============================================================================
# HELPER -- Text Normalization
# =============================================================================

_WHITESPACE_RE = re.compile(r"\s+")
_LEGAL_SUFFIXES = re.compile(
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|llc|inc\.?|corp\.?|"
    r"ltd\.?|limited|s\.?a\.?s\.?|sarl|gmbh|co\.?)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """NFKC normalize -> lowercase -> collapse whitespace -> strip legal suffixes."""
    if not text or not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _LEGAL_SUFFIXES.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> set:
    """Split normalized text into a set of word tokens."""
    return set(text.split()) if text else set()


# =============================================================================
# HELPER -- Similarity Metrics
# =============================================================================

def jaccard(set_a: set, set_b: set) -> float:
    """Token-level Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def token_overlap_ratio(set_a: set, set_b: set) -> float:
    """Overlap coefficient: |A intersect B| / min(|A|,|B|)."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def ngrams(s: str) -> set:
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    na, nb = ngrams(a), ngrams(b)
    return len(na & nb) / len(na | nb)


def jaro_winkler(s1: str, s2: str) -> float:
    """
    Jaro-Winkler similarity (pure Python, no external deps).
    Returns value in [0, 1].
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    match_dist = max(0, match_dist)

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3

    # Winkler prefix boost (up to 4 chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def safe_len_ratio(a: str, b: str) -> float:
    """len(shorter) / len(longer) -- 1.0 means same length."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def safe_token_ratio(a: str, b: str) -> float:
    """token_count(shorter) / token_count(longer)."""
    ta = len(a.split()) if a else 0
    tb = len(b.split()) if b else 0
    if ta == 0 and tb == 0:
        return 1.0
    if ta == 0 or tb == 0:
        return 0.0
    return min(ta, tb) / max(ta, tb)


# =============================================================================
# TF-IDF COSINE SIMILARITY (batch)
# =============================================================================

def build_tfidf_cosine_features(
    texts_a: List[str],
    texts_b: List[str],
    analyzer: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50_000,
) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity for a list of paired texts.
    Fits the vectorizer on both corpora together.

    Returns
    -------
    np.ndarray of shape (N,) with cosine similarities.
    """
    corpus = texts_a + texts_b
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    n = len(texts_a)
    mat_a = tfidf_matrix[:n]
    mat_b = tfidf_matrix[n:]

    # Row-wise cosine similarity
    sims = np.array(
        [cosine_similarity(mat_a[i], mat_b[i])[0, 0] for i in range(n)]
    )
    return sims


# =============================================================================
# EMBEDDING SIMILARITY (optional)
# =============================================================================

def embed_texts(texts: List[str], model) -> np.ndarray:
    """Encode a list of texts using sentence-transformers."""
    return model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def build_embedding_cosine_features(
    texts_a: List[str],
    texts_b: List[str],
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Returns per-pair cosine similarity using multilingual sentence embeddings.
    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers\n"
            "Or set USE_EMBEDDINGS=False in features.py"
        )
    model = SentenceTransformer(model_name)
    emb_a = embed_texts(texts_a, model)
    emb_b = embed_texts(texts_b, model)
    # Dot product (vectors are already L2-normalized)
    sims = np.einsum("ij,ij->i", emb_a, emb_b)
    return sims


# =============================================================================
# ENTITY LOOKUP TABLE
# =============================================================================

class EntityLookup:
    """
    Loads processed TSV files and provides O(1) field access by entity_id.

    Fields returned per entity:
        name_norm     -- business_name_normalized
        addr_norm     -- business_address_normalized
        country       -- country_clean
        name_raw      -- business_name (original)
        addr_raw      -- business_address (original)
    """

    REQUIRED_COLS = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "business_name_clean",
        "business_name_normalized",
        "business_address_clean",
        "business_address_normalized",
        "country_clean",
    ]

    def __init__(self):
        self._store: Dict[str, Dict[str, str]] = {}

    def _load_file(self, path: Path, label: str, valid_ids: Optional[set] = None) -> None:
        if not path.exists():
            print(f"  [WARN] File not found -- skipping: {path}")
            return
        print(f"  Loading {label}: {path}")
        df = pd.read_csv(
            str(path), sep="\t", dtype=str,
            keep_default_na=False, encoding="utf-8",
            usecols=lambda c: c in self.REQUIRED_COLS,
        )
        if valid_ids is not None:
            df = df[df["entity_id"].isin(valid_ids)]
            
        for row in df.itertuples(index=False):
            eid = row.entity_id
            self._store[eid] = {
                "name_norm": getattr(row, "business_name_normalized", ""),
                "addr_norm": getattr(row, "business_address_normalized", ""),
                "country":   getattr(row, "country_clean", ""),
                "name_raw":  getattr(row, "business_name", ""),
                "addr_raw":  getattr(row, "business_address", ""),
            }
        print(f"    -> {len(df)} entities loaded")
        del df
        gc.collect()

    def load(self, split: str = "train", valid_ids: Optional[set] = None) -> "EntityLookup":
        """
        split: 'train', 'test', or 'both'
        """
        files_to_load = {
            "train": ["train_s1", "train_s2", "train_s3"],
            "test":  ["test_s1",  "test_s2",  "test_s3"],
            "both":  list(PROCESSED_FILES.keys()),
        }[split]

        print(f"\n[EntityLookup] Loading {split} processed files from: {PROCESSED_DIR}")
        for key in files_to_load:
            fname = PROCESSED_FILES[key]
            self._load_file(PROCESSED_DIR / fname, key, valid_ids)
        print(f"  Total entities in lookup: {len(self._store)}\n")
        return self

    def get(self, entity_id: str) -> Dict[str, str]:
        """Return entity fields. Returns empty dict if not found."""
        return self._store.get(entity_id, {
            "name_norm": "", "addr_norm": "",
            "country": "", "name_raw": "", "addr_raw": ""
        })

    def __len__(self):
        return len(self._store)


# =============================================================================
# GROUND TRUTH LABEL BUILDER
# =============================================================================

def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Dict[str, set]:
    """
    Loads train_ground_truth.tsv.
    Returns dict: { s1_id -> set of matching S2/S3 ids }
    Singletons (empty match_entity_ids) map to empty set.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            f"Set AMAZON_ML_DATASET_DIR env var or adjust GROUND_TRUTH_PATH in features.py"
        )
    df = pd.read_csv(str(path), sep="\t", dtype=str,
                     keep_default_na=False, encoding="utf-8")
    gt: Dict[str, set] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = row.get("matched_entity_ids", "").strip()
        gt[s1_id] = set(m.strip() for m in matches_str.split(",") if m.strip())
    return gt


# =============================================================================
# CANDIDATE PAIR LOADER
# =============================================================================

def load_candidate_pairs(path: Path = CANDIDATE_PAIRS_PATH) -> pd.DataFrame:
    """
    Loads candidate_pairs.tsv and explodes into one (s1_id, cand_id) row per pair.

    Input format:
        source1_entity_id  | candidate_entity_ids
        S1-xxx             | S2-aaa,S2-bbb,S3-ccc,...

    Returns DataFrame with columns: [s1_id, cand_id]
    """
    if not path.exists():
        raise FileNotFoundError(f"Candidate pairs not found: {path}")

    print(f"[load_candidate_pairs] Reading: {path}")
    df = pd.read_csv(str(path), sep="\t", dtype=str,
                     keep_default_na=False, encoding="utf-8")

    print(f"  Rows before explode: {len(df)}")

    # Explode comma-separated candidate IDs into individual rows
    df["candidate_entity_ids"] = df["candidate_entity_ids"].fillna("")
    df["cand_id"] = df["candidate_entity_ids"].str.split(",")
    df = df.explode("cand_id")
    df["cand_id"] = df["cand_id"].str.strip()
    df = df[df["cand_id"] != ""]   # drop empty strings from trailing commas
    df = df.rename(columns={"source1_entity_id": "s1_id"})
    df = df[["s1_id", "cand_id"]].reset_index(drop=True)

    print(f"  Total pairs after explode: {len(df)}")
    return df


# =============================================================================
# CORE: FEATURE COMPUTATION (per batch of pairs)
# =============================================================================

def compute_pairwise_features(
    pairs_df: pd.DataFrame,
    lookup: EntityLookup,
    tfidf_name_sims: Optional[np.ndarray] = None,
    tfidf_addr_sims: Optional[np.ndarray] = None,
    emb_name_sims: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute feature vector for each row in pairs_df.

    Parameters
    ----------
    pairs_df : DataFrame with columns [s1_id, cand_id] (+ optionally [label])
    lookup   : EntityLookup with entity data
    tfidf_name_sims : pre-computed TF-IDF cosine for names (len == len(pairs_df))
    tfidf_addr_sims : pre-computed TF-IDF cosine for addresses
    emb_name_sims   : pre-computed embedding cosine for names

    Returns
    -------
    DataFrame with original columns + feature columns (prefix 'f_')
    """
    records = []

    for idx, row in enumerate(pairs_df.itertuples(index=False)):
        s1 = lookup.get(row.s1_id)
        ca = lookup.get(row.cand_id)

        # -- normalized text fields --
        n1  = s1["name_norm"]
        n2  = ca["name_norm"]
        a1  = s1["addr_norm"]
        a2  = ca["addr_norm"]
        c1  = s1["country"]
        c2  = ca["country"]
        nr1 = s1["name_raw"]
        nr2 = ca["name_raw"]
        ar1 = s1["addr_raw"]
        ar2 = ca["addr_raw"]

        # -- token sets --
        nt1 = tokenize(n1)
        nt2 = tokenize(n2)
        at1 = tokenize(a1)
        at2 = tokenize(a2)

        feat: Dict[str, float] = {}

        # -- Name similarities --
        feat["f_name_jaccard"]          = jaccard(nt1, nt2)
        feat["f_name_overlap"]          = token_overlap_ratio(nt1, nt2)
        feat["f_name_jaro_winkler"]     = jaro_winkler(n1[:128], n2[:128])
        feat["f_name_jaro_winkler_raw"] = jaro_winkler(nr1[:128], nr2[:128])
        feat["f_name_char3gram"]        = char_ngram_jaccard(n1, n2, n=3)
        feat["f_name_char4gram"]        = char_ngram_jaccard(n1, n2, n=4)
        feat["f_name_len_ratio"]        = safe_len_ratio(n1, n2)
        feat["f_name_token_ratio"]      = safe_token_ratio(n1, n2)
        feat["f_name_exact_match"]      = float(n1 == n2 and n1 != "")

        # -- Address similarities --
        feat["f_addr_jaccard"]          = jaccard(at1, at2)
        feat["f_addr_overlap"]          = token_overlap_ratio(at1, at2)
        feat["f_addr_jaro_winkler"]     = jaro_winkler(a1[:128], a2[:128])
        feat["f_addr_jaro_winkler_raw"] = jaro_winkler(ar1[:128], ar2[:128])
        feat["f_addr_char3gram"]        = char_ngram_jaccard(a1, a2, n=3)
        feat["f_addr_len_ratio"]        = safe_len_ratio(a1, a2)
        feat["f_addr_exact_match"]      = float(a1 == a2 and a1 != "")
        feat["f_both_addr_empty"]       = float(a1 == "" and a2 == "")
        feat["f_addr_missing_one"]      = float((a1 == "") != (a2 == ""))

        # -- Country --
        feat["f_country_match"]         = float(c1 == c2 and c1 != "")
        feat["f_both_country_missing"]  = float(c1 == "" and c2 == "")

        # -- Cross-field: name tokens in address --
        feat["f_name_in_addr_overlap"]  = (
            token_overlap_ratio(nt1, at2) if nt1 and at2 else 0.0
        )

        # -- TF-IDF cosine (pre-computed externally) --
        feat["f_tfidf_name_cosine"]     = (
            float(tfidf_name_sims[idx]) if tfidf_name_sims is not None else -1.0
        )
        feat["f_tfidf_addr_cosine"]     = (
            float(tfidf_addr_sims[idx]) if tfidf_addr_sims is not None else -1.0
        )

        # -- Embedding cosine (pre-computed externally) --
        feat["f_emb_name_cosine"]       = (
            float(emb_name_sims[idx]) if emb_name_sims is not None else -1.0
        )

        records.append(feat)

    feat_df = pd.DataFrame(records, index=pairs_df.index)
    result = pd.concat(
        [pairs_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1
    )
    return result


# =============================================================================
# FEATURE BUILDER  -- high-level API
# =============================================================================

class FeatureBuilder:
    """
    High-level API for building features for the Amazon ML Challenge.

    Usage
    -----
    >>> fb = FeatureBuilder()
    >>> fb.load_entity_data(split="both")
    >>> train_df = fb.build_train_features(sample_frac=0.3)
    >>> test_df  = fb.build_test_features()
    """

    def __init__(
        self,
        candidate_pairs_path: Path = CANDIDATE_PAIRS_PATH,
        processed_dir: Path = PROCESSED_DIR,
        ground_truth_path: Path = GROUND_TRUTH_PATH,
        use_embeddings: bool = USE_EMBEDDINGS,
        chunk_size: int = 100_000,
    ):
        self.candidate_pairs_path = candidate_pairs_path
        self.processed_dir = processed_dir
        self.ground_truth_path = ground_truth_path
        self.use_embeddings = use_embeddings
        self.chunk_size = chunk_size

        self.lookup: Optional[EntityLookup] = None
        self.gt: Optional[Dict[str, set]] = None
        self._all_pairs: Optional[pd.DataFrame] = None

    # -- Data loading ----------------------------------------------------------

    def load_entity_data(self, split: str = "both", valid_ids: Optional[set] = None) -> "FeatureBuilder":
        """Load processed entity files into lookup. split: 'train'|'test'|'both'"""
        self.lookup = EntityLookup().load(split=split, valid_ids=valid_ids)
        return self

    def load_ground_truth(self) -> "FeatureBuilder":
        """Load train_ground_truth.tsv for labelling training pairs."""
        print(f"[FeatureBuilder] Loading ground truth: {self.ground_truth_path}")
        self.gt = load_ground_truth(self.ground_truth_path)
        print(f"  S1 entities with GT: {len(self.gt)}")
        return self

    def load_candidate_pairs(self) -> "FeatureBuilder":
        """Load and explode candidate_pairs.tsv."""
        self._all_pairs = load_candidate_pairs(self.candidate_pairs_path)
        return self

    # -- Internal helpers ------------------------------------------------------

    def _assign_labels(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign binary labels using ground truth.
        label=1 if cand_id is in gt[s1_id], else label=0.
        Only call after load_ground_truth().
        """
        if self.gt is None:
            raise RuntimeError("Call load_ground_truth() first.")

        def _label(row):
            matches = self.gt.get(row["s1_id"], set())
            return 1 if row["cand_id"] in matches else 0

        pairs_df = pairs_df.copy()
        pairs_df["label"] = pairs_df.apply(_label, axis=1)
        pos = pairs_df["label"].sum()
        neg = (pairs_df["label"] == 0).sum()
        print(f"  Positives: {pos:,}  Negatives: {neg:,}  Ratio: 1:{neg/max(pos,1):.0f}")
        return pairs_df

    def _compute_tfidf_and_embed(
        self, pairs_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Batch TF-IDF + (optional) embedding similarity computation."""
        if self.lookup is None:
            raise RuntimeError("Call load_entity_data() first.")

        names_a = [self.lookup.get(r.s1_id)["name_norm"]  for r in pairs_df.itertuples()]
        names_b = [self.lookup.get(r.cand_id)["name_norm"] for r in pairs_df.itertuples()]
        addrs_a = [self.lookup.get(r.s1_id)["addr_norm"]  for r in pairs_df.itertuples()]
        addrs_b = [self.lookup.get(r.cand_id)["addr_norm"] for r in pairs_df.itertuples()]

        print("  Computing TF-IDF cosine (names)...")
        tfidf_name = build_tfidf_cosine_features(names_a, names_b, analyzer="word")

        print("  Computing TF-IDF cosine (addresses)...")
        tfidf_addr = build_tfidf_cosine_features(addrs_a, addrs_b, analyzer="word")

        emb_name = None
        if self.use_embeddings:
            print("  Computing multilingual embedding cosine (names)...")
            emb_name = build_embedding_cosine_features(names_a, names_b)

        return tfidf_name, tfidf_addr, emb_name

    def _build_features_chunked(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process pairs in chunks to keep RAM usage bounded.
        Returns full feature DataFrame.
        """
        if self.lookup is None:
            raise RuntimeError("Call load_entity_data() first.")

        n = len(pairs_df)
        chunks = []
        total_chunks = math.ceil(n / self.chunk_size)

        print(f"\n[FeatureBuilder] Building features for {n:,} pairs "
              f"in {total_chunks} chunks of {self.chunk_size:,} ...")

        for i, start in enumerate(range(0, n, self.chunk_size)):
            chunk = pairs_df.iloc[start:start + self.chunk_size].copy()
            print(f"  Chunk {i+1}/{total_chunks}  rows {start}-{start+len(chunk)-1}")

            tfidf_name, tfidf_addr, emb_name = self._compute_tfidf_and_embed(chunk)

            feat_chunk = compute_pairwise_features(
                chunk, self.lookup,
                tfidf_name_sims=tfidf_name,
                tfidf_addr_sims=tfidf_addr,
                emb_name_sims=emb_name,
            )
            chunks.append(feat_chunk)
            del tfidf_name, tfidf_addr, emb_name
            gc.collect()

        return pd.concat(chunks, ignore_index=True)

    # -- Public API ------------------------------------------------------------

    def build_train_features(
        self,
        sample_frac: Optional[float] = None,
        neg_pos_ratio: int = 10,
        random_state: int = 42,
        cache: bool = True,
    ) -> pd.DataFrame:
        """
        Build labelled feature DataFrame for model training.

        Parameters
        ----------
        sample_frac    : if set, sample this fraction of all pairs (memory saving)
        neg_pos_ratio  : keep at most this many negatives per positive (class balancing)
        random_state   : random seed for reproducibility
        cache          : if True, save/load to FEATURES_CACHE_DIR

        Returns
        -------
        DataFrame with columns: [s1_id, cand_id, label, f_*...]
        """
        cache_path = FEATURES_CACHE_DIR / "train_features.parquet"
        if cache and cache_path.exists():
            print(f"[FeatureBuilder] Loading cached train features: {cache_path}")
            return pd.read_parquet(cache_path)

        # Ensure dependencies are loaded
        if self.gt is None:
            self.load_ground_truth()

        train_cand_path = _REPO_ROOT / "output" / "train_candidate_pairs.tsv"
        cand_path = train_cand_path if train_cand_path.exists() else self.candidate_pairs_path
        all_train_pairs = load_candidate_pairs(cand_path)

        # Filter to train S1 IDs (those covered by GT)
        gt_s1_ids = set(self.gt.keys())
        train_pairs = all_train_pairs[
            all_train_pairs["s1_id"].isin(gt_s1_ids)
        ].copy()

        print(f"\n[FeatureBuilder] Train pairs (GT-covered S1): {len(train_pairs):,}")

        # Label assignment
        train_pairs = self._assign_labels(train_pairs)

        # Optional sub-sampling for large datasets
        if sample_frac is not None:
            print(f"  Sampling {sample_frac*100:.0f}% of pairs ...")
            pos = train_pairs[train_pairs["label"] == 1]
            neg = train_pairs[train_pairs["label"] == 0]
            pos_sampled = pos.sample(frac=sample_frac, random_state=random_state)
            max_neg = len(pos_sampled) * neg_pos_ratio
            neg_sampled = neg.sample(
                n=min(len(neg), max_neg), random_state=random_state
            )
            train_pairs = pd.concat([pos_sampled, neg_sampled]).sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
            print(f"  Sampled: {len(pos_sampled):,} pos, {len(neg_sampled):,} neg")
        else:
            # Balance negatives even without sampling
            pos = train_pairs[train_pairs["label"] == 1]
            neg = train_pairs[train_pairs["label"] == 0]
            if len(neg) > len(pos) * neg_pos_ratio:
                neg = neg.sample(n=len(pos) * neg_pos_ratio, random_state=random_state)
                train_pairs = pd.concat([pos, neg]).sample(
                    frac=1, random_state=random_state
                ).reset_index(drop=True)
                print(f"  After balance: {len(pos):,} pos, {len(neg):,} neg")

        if self.lookup is None:
            valid_ids = set(train_pairs["s1_id"]).union(set(train_pairs["cand_id"]))
            self.load_entity_data(split="train", valid_ids=valid_ids)

        feature_df = self._build_features_chunked(train_pairs)

        if cache:
            print(f"  Caching train features -> {cache_path}")
            feature_df.to_parquet(cache_path, index=False)

        return feature_df

    def build_test_features(self, cache: bool = True) -> pd.DataFrame:
        """
        Build feature DataFrame for all candidate pairs (test inference).
        No labels assigned.

        Returns
        -------
        DataFrame with columns: [s1_id, cand_id, f_*...]
        """
        cache_path = FEATURES_CACHE_DIR / "test_features.parquet"
        if cache and cache_path.exists():
            print(f"[FeatureBuilder] Loading cached test features: {cache_path}")
            return pd.read_parquet(cache_path)

        if self._all_pairs is None:
            self.load_candidate_pairs()
            
        if self.lookup is None:
            valid_ids = set(self._all_pairs["s1_id"]).union(set(self._all_pairs["cand_id"]))
            self.load_entity_data(split="test", valid_ids=valid_ids)

        feature_df = self._build_features_chunked(self._all_pairs)

        if cache:
            print(f"  Caching test features -> {cache_path}")
            feature_df.to_parquet(cache_path, index=False)

        return feature_df


# =============================================================================
# CLI -- quick sanity check
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("features.py -- Quick Sanity Check")
    print("=" * 60)

    # Smoke-test with dummy data (no real files needed)
    dummy_pairs = pd.DataFrame({
        "s1_id":   ["S1-001", "S1-002", "S1-003"],
        "cand_id": ["S2-001", "S2-002", "S3-999"],
    })

    class _DummyLookup:
        _data = {
            "S1-001": {"name_norm": "amazon india pvt", "addr_norm": "12 main street",       "country": "India", "name_raw": "Amazon India Pvt. Ltd.", "addr_raw": "12 Main Street"},
            "S2-001": {"name_norm": "amazon india",     "addr_norm": "12 main street",       "country": "India", "name_raw": "Amazon India",           "addr_raw": "12 Main Street"},
            "S1-002": {"name_norm": "reliance retail",  "addr_norm": "andheri west mumbai",  "country": "India", "name_raw": "Reliance Retail Ltd",    "addr_raw": "Andheri West, Mumbai"},
            "S2-002": {"name_norm": "reliance retail",  "addr_norm": "andheri west",         "country": "India", "name_raw": "Reliance Retail",        "addr_raw": "Andheri West"},
            "S1-003": {"name_norm": "walmart",          "addr_norm": "bentonville ar",        "country": "US",    "name_raw": "Walmart Inc.",           "addr_raw": "Bentonville, AR"},
            "S3-999": {"name_norm": "target",           "addr_norm": "minneapolis mn",        "country": "US",    "name_raw": "Target Corp",            "addr_raw": "Minneapolis, MN"},
        }

        def get(self, eid):
            return self._data.get(
                eid, {"name_norm": "", "addr_norm": "", "country": "", "name_raw": "", "addr_raw": ""}
            )

    dummy_lookup = _DummyLookup()

    names_a = [dummy_lookup.get(r.s1_id)["name_norm"]  for r in dummy_pairs.itertuples()]
    names_b = [dummy_lookup.get(r.cand_id)["name_norm"] for r in dummy_pairs.itertuples()]
    addrs_a = [dummy_lookup.get(r.s1_id)["addr_norm"]  for r in dummy_pairs.itertuples()]
    addrs_b = [dummy_lookup.get(r.cand_id)["addr_norm"] for r in dummy_pairs.itertuples()]

    tfidf_n = build_tfidf_cosine_features(names_a, names_b)
    tfidf_a = build_tfidf_cosine_features(addrs_a, addrs_b)

    result = compute_pairwise_features(
        dummy_pairs, dummy_lookup,
        tfidf_name_sims=tfidf_n,
        tfidf_addr_sims=tfidf_a,
    )

    print("\nFeature matrix shape:", result.shape)
    print("\nSample features:")
    print(result[[
        "s1_id", "cand_id",
        "f_name_jaccard", "f_name_jaro_winkler",
        "f_tfidf_name_cosine", "f_country_match"
    ]].to_string(index=False))
    print("\nAll feature columns:")
    feat_cols = [c for c in result.columns if c.startswith("f_")]
    print(feat_cols)
    print(f"\nTotal features: {len(feat_cols)}")
    print("\n[OK] features.py smoke-test passed.")
