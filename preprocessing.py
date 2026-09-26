"""
Business Entity Resolution - EDA + Data Preprocessing Pipeline
==============================================================
Author: EDA/Preprocessing stage
Challenge: Amazon ML Challenge - Business Entity Resolution

ROLES:
  - Performs thorough EDA on all source files
  - Builds a reproducible, information-preserving preprocessing pipeline
  - Outputs processed TSV files ready for candidate generation & matching

SCHEMA (all sources):
  entity_id | business_name | business_address | country

OBSERVED DATA CHARACTERISTICS:
  - Countries: US, India in train; US, India, France in test (open-set)
  - Business names contain: Devanagari (Hindi), Kannada, French, English
  - Addresses contain: Devanagari, Kannada scripts; PIN codes, ZIP codes
  - Ground truth: comma-separated match IDs (S2-xxx,S3-xxx format)
  - Missing addresses are present (empty string, not NaN)
  - Legal suffix variations: Pvt. Ltd., Private Limited, LLC, SARL, S.A.S
  - Noise: leading dashes (-- Holloway Peak Inc), extra spaces, casing variation

CONSTRAINTS:
  - All files are TSV; read with sep=tab
  - Never overwrite original columns
  - Never drop rows
  - Never modify entity IDs
  - Country is open-set; never hard-code known countries
  - No external APIs, geocoding, or external data
  - Fit preprocessing only on train data; apply consistently to test
  - Do not build the matching model
  - Do not generate final submission
"""

import gc
import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Force UTF-8 stdout so Devanagari/Kannada/French chars don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION - Adjust BASE_DATA_DIR if needed
# =============================================================================

BASE_DATA_DIR = Path("D:/amazon_ml_extracted/student_resource/dataset")
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR  = BASE_DATA_DIR / "test"

OUTPUT_DIR = Path("D:/amazon_ml_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"
TEST_S1  = TEST_DIR  / "test_source1.tsv"
TEST_S2  = TEST_DIR  / "test_source2.tsv"
TEST_S3  = TEST_DIR  / "test_source3.tsv"

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_tsv(path: Path, desc: str = "") -> pd.DataFrame:
    """Load a TSV file with UTF-8 encoding. Never assume CSV."""
    print(f"  Loading {desc}: {path}")
    df = pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"    -> Shape: {df.shape}, Columns: {list(df.columns)}")
    return df

def load_all_files():
    """Load all train and test files. Return dict."""
    print("\n" + "="*70)
    print("LOADING ALL DATA FILES")
    print("="*70)
    data = {
        "train_s1": load_tsv(TRAIN_S1, "Train Source1"),
        "train_s2": load_tsv(TRAIN_S2, "Train Source2"),
        "train_s3": load_tsv(TRAIN_S3, "Train Source3"),
        "train_gt": load_tsv(TRAIN_GT, "Train Ground Truth"),
        "test_s1":  load_tsv(TEST_S1,  "Test Source1"),
        "test_s2":  load_tsv(TEST_S2,  "Test Source2"),
        "test_s3":  load_tsv(TEST_S3,  "Test Source3"),
    }
    return data


# =============================================================================
# SECTION 2: EDA UTILITIES
# =============================================================================

def describe_dataset(df: pd.DataFrame, name: str) -> dict:
    """Compute dataset-level EDA statistics."""
    print(f"\n{'='*60}")
    print(f"EDA: {name}")
    print(f"{'='*60}")
    sys.stdout.flush()
    stats = {}
    stats["name"]    = name
    stats["rows"]    = len(df)
    stats["cols"]    = len(df.columns)
    stats["columns"] = list(df.columns)
    print(f"  Rows: {stats['rows']}, Cols: {stats['cols']}")
    print(f"  Columns: {stats['columns']}")

    # Missing values (NaN and empty string both count as missing)
    missing = {}
    for col in df.columns:
        n_missing = (df[col].isna() | (df[col].str.strip() == "")).sum()
        pct = 100.0 * n_missing / len(df) if len(df) > 0 else 0
        missing[col] = {"count": int(n_missing), "pct": round(pct, 2)}
    stats["missing"] = missing
    print(f"\n  Missing values (NaN + empty string):")
    for col, m in missing.items():
        if m["count"] > 0:
            print(f"    {col}: {m['count']} ({m['pct']}%)")
        else:
            print(f"    {col}: 0 (0.00%) - complete")

    # Duplicate rows
    n_dup_rows = int(df.duplicated().sum())
    stats["duplicate_rows"] = n_dup_rows
    print(f"\n  Duplicate rows: {n_dup_rows}")

    # Unique values per column
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}
    stats["unique_values"] = unique_counts
    print(f"\n  Unique values per column:")
    for col, u in unique_counts.items():
        print(f"    {col}: {u}")

    # Entity ID analysis
    if "entity_id" in df.columns:
        n_dup_ids = int(df["entity_id"].duplicated().sum())
        stats["duplicate_entity_ids"] = n_dup_ids
        print(f"\n  Duplicate entity_ids: {n_dup_ids}")
        # Sample IDs
        print(f"  Sample entity_ids: {df['entity_id'].head(3).tolist()}")

    return stats


def analyze_string_lengths(df: pd.DataFrame, col: str, name: str) -> dict:
    """Analyze string length distribution for a column."""
    if col not in df.columns:
        return {}
    lengths = df[col].fillna("").str.len()
    token_counts = df[col].fillna("").str.split().str.len().fillna(0)
    stats = {
        "col": col,
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "mean_len": round(float(lengths.mean()), 2),
        "median_len": round(float(lengths.median()), 2),
        "min_tokens": int(token_counts.min()),
        "max_tokens": int(token_counts.max()),
        "mean_tokens": round(float(token_counts.mean()), 2),
        "median_tokens": round(float(token_counts.median()), 2),
    }
    print(f"\n  String lengths for [{col}] in {name}:")
    print(f"    Length - min:{stats['min_len']} max:{stats['max_len']} "
          f"mean:{stats['mean_len']} median:{stats['median_len']}")
    print(f"    Tokens - min:{stats['min_tokens']} max:{stats['max_tokens']} "
          f"mean:{stats['mean_tokens']} median:{stats['median_tokens']}")
    return stats


def analyze_country(df: pd.DataFrame, name: str) -> dict:
    """Analyze country distribution. Country is OPEN-SET - never hard-code."""
    if "country" not in df.columns:
        return {}
    vc = df["country"].value_counts(dropna=False)
    print(f"\n  Country distribution in {name}:")
    for c, cnt in vc.items():
        pct = 100.0 * cnt / len(df)
        print(f"    '{c}': {cnt} ({pct:.1f}%)")
    return dict(vc)


def analyze_business_names(df: pd.DataFrame, name: str):
    """Detailed business_name EDA."""
    if "business_name" not in df.columns:
        print(f"  No business_name column in {name}")
        return

    col = df["business_name"].fillna("").astype(str)
    print(f"\n  ---- Business Name Analysis [{name}] ----")

    # Casing analysis
    n_all_upper = int((col.str.isupper() & (col != "")).sum())
    n_all_lower = int((col.str.islower() & (col != "")).sum())
    n_title     = int((col.str.istitle() & (col != "")).sum())
    n_mixed     = int(len(col[col != ""]) - n_all_upper - n_all_lower - n_title)
    print(f"    Casing: ALL-UPPER={n_all_upper} all-lower={n_all_lower} Title={n_title} mixed={n_mixed}")

    # Special characters
    has_ampersand = int(col.str.contains(r"&", regex=False).sum())
    has_slash     = int(col.str.contains(r"/", regex=False).sum())
    has_dot       = int(col.str.contains(r"\.", regex=True).sum())
    has_comma     = int(col.str.contains(r",", regex=False).sum())
    has_paren     = int(col.str.contains(r"[()]", regex=True).sum())
    has_unicode   = int(col.apply(lambda x: any(ord(c) > 127 for c in x)).sum())
    has_multi_ws  = int(col.str.contains(r"\s{2,}", regex=True).sum())
    has_lead_dash = int(col.str.startswith("-").sum())
    has_lead_lt   = int(col.str.startswith("<").sum())
    print(f"    Has '&':         {has_ampersand}")
    print(f"    Has '/':         {has_slash}")
    print(f"    Has '.':         {has_dot}")
    print(f"    Has ',':         {has_comma}")
    print(f"    Has '()':        {has_paren}")
    print(f"    Has unicode>127: {has_unicode}")
    print(f"    Multi-space:     {has_multi_ws}")
    print(f"    Starts with '-': {has_lead_dash}")
    print(f"    Starts with '<': {has_lead_lt}")

    # Legal suffixes
    suffix_patterns = {
        "Ltd/Limited":        r"\b(ltd\.?|limited)\b",
        "Pvt/Private":        r"\b(pvt\.?|private)\b",
        "Corp/Corporation":   r"\b(corp\.?|corporation)\b",
        "Inc/Incorporated":   r"\b(inc\.?|incorporated)\b",
        "LLC":                r"\bllc\.?\b",
        "LLP":                r"\bllp\.?\b",
        "Co/Company":         r"\b(co\.?|company)\b",
        "Pvt.Ltd. (combined)":r"pvt\.?\s*ltd\.?",
        "SARL (French)":      r"\bsarl\b",
        "SAS/S.A.S (French)": r"\bs\.?a\.?s\.?\b",
        "SCI (French)":       r"\bsci\b",
    }
    print(f"\n    Legal suffix occurrences (case-insensitive):")
    found_any = False
    for label, pattern in suffix_patterns.items():
        cnt = int(col.str.contains(pattern, case=False, regex=True).sum())
        if cnt > 0:
            print(f"      '{label}': {cnt}")
            found_any = True
    if not found_any:
        print(f"      (none found)")

    # '& / and' variations
    n_ampersand = int(col.str.contains(r"&", regex=False).sum())
    n_and_word  = int(col.str.contains(r"\band\b", case=False, regex=True).sum())
    print(f"\n    '&' occurrences: {n_ampersand}, ' and ' word occurrences: {n_and_word}")

    # Duplicate names within source
    n_dup_names = int(col[col != ""].duplicated().sum())
    print(f"\n    Duplicate business_name within {name}: {n_dup_names}")
    if n_dup_names > 0:
        dup_names = col[col != ""][col[col != ""].duplicated(keep=False)]
        vc_dup = dup_names.value_counts()
        print(f"    Top 10 most frequent duplicate names:")
        for n_val, cnt in vc_dup.head(10).items():
            print(f"      '{n_val}': {cnt} times")

    # Sample names including unicode
    print(f"\n    Sample names (first 10 incl. any unicode):")
    shown = 0
    for v in col:
        if shown >= 10:
            break
        safe = repr(v).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(f"      {safe}")
        shown += 1


def analyze_addresses(df: pd.DataFrame, name: str):
    """Detailed business_address EDA."""
    if "business_address" not in df.columns:
        print(f"  No business_address column in {name}")
        return

    col = df["business_address"].fillna("").astype(str)
    print(f"\n  ---- Business Address Analysis [{name}] ----")

    n_empty = int((col.str.strip() == "").sum())
    print(f"    Empty/missing addresses: {n_empty} ({100.0*n_empty/len(col):.1f}%)")

    has_comma   = int(col.str.contains(",", regex=False).sum())
    has_dash    = int(col.str.contains("-", regex=False).sum())
    has_slash   = int(col.str.contains("/", regex=False).sum())
    has_hash    = int(col.str.contains("#", regex=False).sum())
    has_unicode = int(col.apply(lambda x: any(ord(c) > 127 for c in x)).sum())
    has_mws     = int(col.str.contains(r"\s{2,}", regex=True).sum())
    has_null_str= int(col.str.contains(r"\bnull\b", case=False, regex=True).sum())
    print(f"    Has ',':       {has_comma}")
    print(f"    Has '-':       {has_dash}")
    print(f"    Has '/':       {has_slash}")
    print(f"    Has '#':       {has_hash}")
    print(f"    Has unicode:   {has_unicode}")
    print(f"    Multi-space:   {has_mws}")
    print(f"    Contains 'null' literal: {has_null_str}")

    # Postal/ZIP code patterns
    has_indian_pin = int(col.str.contains(r"\b\d{6}\b", regex=True).sum())
    has_us_zip5    = int(col.str.contains(r"\b\d{5}\b", regex=True).sum())
    has_us_zip54   = int(col.str.contains(r"\b\d{5}-\d{4}\b", regex=True).sum())
    print(f"    Indian PIN (6-digit):  {has_indian_pin}")
    print(f"    US ZIP 5-digit:        {has_us_zip5}")
    print(f"    US ZIP+4 hyphen:       {has_us_zip54}")

    # Component pattern checks
    road_patterns = {
        "St/Street":   r"\b(st|street)\b",
        "Rd/Road":     r"\b(rd|road)\b",
        "Ave/Avenue":  r"\b(ave|avenue)\b",
        "Blvd":        r"\bblvd\b",
        "Dr/Drive":    r"\b(dr|drive)\b",
        "Nagar":       r"\bnagar\b",
        "Marg":        r"\bmarg\b",
        "Colony":      r"\bcolony\b",
        "Ganj":        r"\bganj\b",
        "Rue (French)":r"\brue\b",
        "Boulevard (French)": r"\bboulevard\b",
        "Vihar":       r"\bvihar\b",
        "PO Box":      r"\bpo\s*box\b",
    }
    print(f"\n    Address component patterns (case-insensitive):")
    for label, pattern in road_patterns.items():
        cnt = int(col.str.contains(pattern, case=False, regex=True).sum())
        if cnt > 0:
            print(f"      '{label}': {cnt}")

    # State abbreviations / casing
    all_upper_words = col.apply(lambda x: sum(1 for w in x.split() if w.isupper() and len(w) >= 2))
    print(f"\n    Records with ALL-CAPS tokens (state abbrevs): {int((all_upper_words > 0).sum())}")

    # Sample addresses
    print(f"\n    Sample addresses (first 10):")
    for v in col[:10]:
        safe = repr(v).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(f"      {safe}")


# =============================================================================
# SECTION 3: GROUND TRUTH EDA
# =============================================================================

def analyze_ground_truth(gt: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """Comprehensive analysis of train_ground_truth.tsv."""
    print(f"\n{'='*60}")
    print("EDA: GROUND TRUTH ANALYSIS")
    print(f"{'='*60}")
    print(f"  GT shape: {gt.shape}")
    print(f"  Columns: {list(gt.columns)}")
    print(f"\n  Sample rows (first 5):")
    for _, row in gt.head(5).iterrows():
        print(f"    source1_entity_id={row.get('source1_entity_id','?')} "
              f"matched_entity_ids={row.get('matched_entity_ids','?')[:80]}")

    # Parse matched_entity_ids
    # Observed format: "S2-xxx,S3-xxx,S2-yyy" (comma-separated, no brackets)
    def parse_match_ids(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        val = str(val).strip()
        # Remove surrounding brackets if any
        val = val.strip("[]").strip()
        if val == "" or val.lower() in ("nan", "none", "null"):
            return []
        ids = [x.strip().strip("'\"") for x in val.split(",") if x.strip()]
        return ids

    gt_work = gt.copy()
    gt_work["_match_list"] = gt_work["matched_entity_ids"].apply(parse_match_ids)
    gt_work["_n_matches"]  = gt_work["_match_list"].apply(len)

    s2_ids = set(s2["entity_id"].tolist()) if "entity_id" in s2.columns else set()
    s3_ids = set(s3["entity_id"].tolist()) if "entity_id" in s3.columns else set()

    n_total = len(gt_work)
    n_zero  = int((gt_work["_n_matches"] == 0).sum())
    n_one   = int((gt_work["_n_matches"] == 1).sum())
    n_two   = int((gt_work["_n_matches"] == 2).sum())
    n_three = int((gt_work["_n_matches"] == 3).sum())
    n_multi = int((gt_work["_n_matches"] >= 2).sum())
    n_max   = int(gt_work["_n_matches"].max())
    n_mean  = round(float(gt_work["_n_matches"].mean()), 4)

    print(f"\n  Total S1 entities in GT: {n_total:,}")
    print(f"  S1 entities with 0 matches (singletons): {n_zero:,} ({100.0*n_zero/n_total:.2f}%)")
    print(f"  S1 entities with 1 match:                {n_one:,} ({100.0*n_one/n_total:.2f}%)")
    print(f"  S1 entities with 2 matches:              {n_two:,} ({100.0*n_two/n_total:.2f}%)")
    print(f"  S1 entities with 3 matches:              {n_three:,} ({100.0*n_three/n_total:.2f}%)")
    print(f"  S1 entities with 2+ matches:             {n_multi:,} ({100.0*n_multi/n_total:.2f}%)")
    print(f"  Max matches for a single S1 entity:      {n_max}")
    print(f"  Mean matches per S1 entity:              {n_mean}")

    # Match count distribution
    match_dist = gt_work["_n_matches"].value_counts().sort_index()
    print(f"\n  Match count distribution:")
    for k, v in match_dist.head(15).items():
        print(f"    {k} matches: {v:,} S1 entities")

    # Breakdown by source
    all_match_ids = [mid for lst in gt_work["_match_list"] for mid in lst]
    s2_matches   = [m for m in all_match_ids if str(m).startswith("S2-")]
    s3_matches   = [m for m in all_match_ids if str(m).startswith("S3-")]
    s1_matches   = [m for m in all_match_ids if str(m).startswith("S1-")]
    other_matches= [m for m in all_match_ids if not str(m).startswith(("S1-","S2-","S3-"))]
    print(f"\n  Total matched IDs across all GT rows: {len(all_match_ids):,}")
    print(f"  S2 matches: {len(s2_matches):,}")
    print(f"  S3 matches: {len(s3_matches):,}")
    if s1_matches:
        print(f"  WARNING: S1 self-matches found: {len(s1_matches):,}")
    if other_matches:
        print(f"  WARNING: Unrecognized match IDs: {len(other_matches):,}")
        print(f"    Examples: {other_matches[:5]}")

    # Duplicate match IDs in same row
    has_dup_in_row = int(gt_work["_match_list"].apply(
        lambda lst: len(lst) != len(set(lst))).sum())
    print(f"\n  GT rows with duplicate match IDs in same row: {has_dup_in_row}")

    # Referential integrity - sample check (full check is expensive for 5M rows)
    s2_match_set = set(s2_matches[:100000])
    s3_match_set = set(s3_matches[:100000])
    invalid_s2 = len(s2_match_set - s2_ids)
    invalid_s3 = len(s3_match_set - s3_ids)
    print(f"\n  Referential integrity check (sample):")
    print(f"    S2 match IDs not in train_source2: {invalid_s2}")
    print(f"    S3 match IDs not in train_source3: {invalid_s3}")

    singleton_rate = 100.0 * n_zero / n_total
    print(f"\n  SINGLETON RATE: {n_zero:,}/{n_total:,} = {singleton_rate:.2f}%")
    print(f"  IMPORTANT for F0.5: correctly predicting singletons is rewarded")

    return {
        "total_s1":       n_total,
        "n_singletons":   n_zero,
        "singleton_rate": round(singleton_rate, 4),
        "n_one_match":    n_one,
        "n_multi_match":  n_multi,
        "max_matches":    n_max,
        "mean_matches":   n_mean,
    }


# =============================================================================
# SECTION 4: DATA INTEGRITY CHECKS
# =============================================================================

def integrity_checks(data: dict) -> list:
    """Cross-source integrity checks."""
    print(f"\n{'='*60}")
    print("DATA INTEGRITY CHECKS")
    print(f"{'='*60}")
    issues = []

    for key, df in data.items():
        if "gt" in key:
            continue
        if "entity_id" not in df.columns:
            issues.append(f"CRITICAL: {key} has no entity_id column")
            continue

        # Check ID prefix correctness
        expected = {"s1": "S1-", "s2": "S2-", "s3": "S3-"}
        for src, prefix in expected.items():
            if src in key:
                n_bad = int((~df["entity_id"].str.startswith(prefix)).sum())
                if n_bad > 0:
                    bad_examples = df[~df["entity_id"].str.startswith(prefix)]["entity_id"].head(5).tolist()
                    issues.append(f"WARNING: {key} has {n_bad} IDs not starting with '{prefix}': {bad_examples}")
                else:
                    print(f"  OK: {key} all entity_ids start with '{prefix}'")

        # Check uniqueness
        n_dup = int(df["entity_id"].duplicated().sum())
        if n_dup > 0:
            issues.append(f"WARNING: {key} has {n_dup} duplicate entity_ids")
        else:
            print(f"  OK: {key} entity_ids are unique")

        # Check S1 IDs not used as match IDs in GT (if GT present)
        if key in ("train_s1", "test_s1") and "train_gt" in data:
            gt = data["train_gt"]
            if "matched_entity_ids" in gt.columns:
                s1_ids_in_gt_matches = gt["matched_entity_ids"].fillna("").str.contains("S1-")
                n_s1_as_match = int(s1_ids_in_gt_matches.sum())
                if n_s1_as_match > 0:
                    issues.append(f"WARNING: {n_s1_as_match} GT rows may have S1 IDs as match targets")
                else:
                    print(f"  OK: No S1 IDs used as match targets in GT")

    # Train/test schema consistency
    for split in ["s1", "s2", "s3"]:
        train_key = f"train_{split}"
        test_key  = f"test_{split}"
        if train_key in data and test_key in data:
            train_cols = set(data[train_key].columns)
            test_cols  = set(data[test_key].columns)
            if train_cols != test_cols:
                only_train = train_cols - test_cols
                only_test  = test_cols - train_cols
                issues.append(
                    f"Schema mismatch for {split}: "
                    f"only_in_train={only_train}, only_in_test={only_test}"
                )
            else:
                print(f"  OK: train_{split} and test_{split} have same schema: {sorted(train_cols)}")

    # GT source1_entity_id check
    if "train_gt" in data and "train_s1" in data:
        gt_ids = set(data["train_gt"]["source1_entity_id"].tolist())
        s1_ids = set(data["train_s1"]["entity_id"].tolist())
        gt_not_s1 = gt_ids - s1_ids
        s1_not_gt = s1_ids - gt_ids
        if gt_not_s1:
            issues.append(f"GT has {len(gt_not_s1)} source1_entity_ids not in train_source1")
        else:
            print(f"  OK: All GT source1_entity_ids exist in train_source1")
        if s1_not_gt:
            issues.append(f"train_source1 has {len(s1_not_gt)} entity_ids not in GT")
        else:
            print(f"  OK: All train_source1 entity_ids have a GT row")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  All integrity checks passed.")

    return issues


# =============================================================================
# SECTION 5: CROSS-SOURCE OBSERVATIONS
# =============================================================================

def cross_source_observations(data: dict):
    """Name overlap and ambiguity observations across sources."""
    print(f"\n{'='*60}")
    print("CROSS-SOURCE OBSERVATIONS (sample-based)")
    print(f"{'='*60}")

    sources = {}
    for key in ["train_s1", "train_s2", "train_s3"]:
        if key in data and "business_name" in data[key].columns:
            # Sample 200K for speed — sufficient to detect overlap patterns
            sample_size = min(200000, len(data[key]))
            sources[key] = (
                data[key]["business_name"]
                .sample(n=sample_size, random_state=42)
                .fillna("").str.lower().str.strip()
            )

    if len(sources) >= 2:
        s1_names = set(sources.get("train_s1", pd.Series([]))) - {""}
        s2_names = set(sources.get("train_s2", pd.Series([]))) - {""}
        s3_names = set(sources.get("train_s3", pd.Series([]))) - {""}

        s1_s2_overlap   = s1_names & s2_names
        s1_s3_overlap   = s1_names & s3_names
        s2_s3_overlap   = s2_names & s3_names
        s1_s2_s3_overlap= s1_names & s2_names & s3_names

        print(f"\n  Exact-lowercase name overlaps (from samples):")
        print(f"    S1 n S2: {len(s1_s2_overlap)} shared names")
        print(f"    S1 n S3: {len(s1_s3_overlap)} shared names")
        print(f"    S2 n S3: {len(s2_s3_overlap)} shared names")
        print(f"    S1 n S2 n S3: {len(s1_s2_s3_overlap)} shared names")

        if s1_s2_s3_overlap:
            print(f"\n    Sample names appearing in all 3 sources (first 10):")
            for n in list(s1_s2_s3_overlap)[:10]:
                print(f"      '{n}'")

    # Within-source duplicates (ambiguous names)
    for key in ["train_s1", "train_s2", "train_s3"]:
        if key not in data or "business_name" not in data[key].columns:
            continue
        col = data[key]["business_name"].fillna("").str.lower().str.strip()
        vc  = col[col != ""].value_counts()
        n_ambiguous = int((vc > 1).sum())
        print(f"\n  {key} — ambiguous names (same normalized name, >1 record): {n_ambiguous:,}")
        if n_ambiguous > 0:
            print(f"    Top ambiguous names (first 10):")
            for n_val, cnt in vc[vc > 1].head(10).items():
                print(f"      '{n_val}': {cnt} times")

    # Country-specific patterns
    for key in ["train_s1", "train_s2", "train_s3"]:
        if key not in data:
            continue
        df = data[key]
        if "country" in df.columns and "business_name" in df.columns:
            countries = df["country"].value_counts()
            for country in countries.head(5).index:
                subset = df[df["country"] == country]["business_name"].fillna("")
                has_unicode = subset.apply(lambda x: any(ord(c) > 127 for c in x)).sum()
                print(f"\n  {key} [{country}]: {len(subset):,} records, "
                      f"{has_unicode:,} ({100.0*has_unicode/max(len(subset),1):.1f}%) "
                      f"with non-ASCII business names")


# =============================================================================
# SECTION 6: PREPROCESSING PIPELINE FUNCTIONS
# =============================================================================
# Design principles:
#   1. Unicode normalization (NFKC) - canonical form for all scripts
#   2. Whitespace normalization - collapse multi-spaces, strip leading/trailing
#   3. Lowercase (for _normalized columns only)
#   4. Punctuation normalization (conservative)
#   5. Legal suffix normalization (for _normalized columns only)
#   6. Address abbreviation normalization (for _normalized columns only)
#
# Two output columns per field:
#   *_clean      = minimal cleaning only (unicode + whitespace)
#   *_normalized = full normalization for matching (lowercase + punctuation + suffixes)
#
# Original columns are NEVER modified.
# All transformations are applied identically to train and test.
# Country uses no vocabulary mapping (open-set safe).

def normalize_unicode(text: str) -> str:
    """
    Apply NFKC unicode normalization.
    WHY: Converts visually-similar characters to canonical form.
         e.g., full-width letters -> ASCII, ligatures -> components.
         Devanagari and Kannada scripts are preserved (already canonical).
    RISK: Low. NFKC is the standard for text comparison.
    OBSERVED: Source 2 has Devanagari names (Hindi), Source 3 has Kannada in addresses.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace sequences to single space, strip edges.
    WHY: EDA shows multi-space occurrences (e.g., 'Shri Supreme  (Limited)').
         Whitespace is not semantically meaningful in business names/addresses.
    RISK: Very low.
    OBSERVED: 'Shri Supreme Consulting Private  (Limited)' has double space.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_casing(text: str) -> str:
    """
    Convert to lowercase.
    WHY: EDA shows significant casing variation: 'PRIME MONEY' vs 'Prime Money' vs 'prime money'.
         Source 2 addresses often use ALL CAPS (e.g., 'KH NO. -570/13, NEW DELHI').
    RISK: Low. Original is preserved in raw and _clean columns.
    """
    return text.lower()


def normalize_punctuation_name(text: str) -> str:
    """
    Normalize punctuation in business names conservatively.
    WHY (justified by observed data):
      - '&' and 'and' are equivalent: "B & B" vs "B and B"
      - Dots in abbreviations: "P.V.T." -> "PVT", "Ltd." -> "Ltd"
      - Leading noise tokens: "--" prefix observed in S2 ("-- Holloway Peak Inc Seafood")
      - "<<" prefix observed in test_s1 ("<< Team Ecole")
      - Commas within names (less common, act as separators)
    RISK: Medium. Applied only to _normalized column.
           '&'->'and' is safe for business names.
    BEFORE: '-- Holloway Peak Inc Seafood'
    AFTER:  'holloway peak inc seafood'
    BEFORE: 'B+ Retail Inc'
    AFTER:  'b+ retail inc'  ('+' kept - meaningful in business context)
    """
    # Remove leading noise punctuation: --, <<, >>, ==, **
    text = re.sub(r"^[-<>=*#!@~`|]+\s*", "", text)
    # Normalize & -> and
    text = re.sub(r"\s*&\s*", " and ", text)
    # Remove dots from abbreviations: P.V.T. -> PVT, Ltd. -> Ltd
    # Pattern: letter followed by dot (not followed by digit)
    text = re.sub(r"([A-Za-z])\.(?=[A-Za-z])", r"\1", text)  # P.V.T. -> PVT
    text = re.sub(r"([A-Za-z])\.$", r"\1", text)             # trailing dot
    text = re.sub(r"([A-Za-z])\.\s", r"\1 ", text)           # mid-word dot+space
    # Replace comma with space (commas are separators in names)
    text = re.sub(r",", " ", text)
    # Normalize multiple hyphens
    text = re.sub(r"-{2,}", " ", text)
    # Remove brackets content? NO - could be informative (e.g., "Pvt. (Limited)")
    return text


def normalize_legal_suffixes(text: str) -> str:
    """
    Normalize common legal suffixes to canonical forms.
    WHY (justified by observed data):
      - 'Pvt. Ltd.' observed in S3: 'Pvt. EFS Print Ventures Ltd.'
      - 'Private Limited' common in Indian businesses
      - 'LLC' in US ('LLC Moncada Learning Center')
      - 'SARL', 'S.A.S', 'SCI' in French businesses (test set)
      - 'Corp/Corporation', 'Inc/Incorporated' in US businesses
    MAPPING:
      pvt. ltd. / pvt ltd -> pvt ltd
      private limited -> pvt ltd
      limited -> ltd
      corporation -> corp
      incorporated -> inc
      (SARL, SAS kept as-is - French equivalents, already handled by lowercase)
    RISK: Medium. Could collapse distinct entities if a suffix is also a word.
          ONLY applied to _normalized column, not raw or _clean.
          'Co' not normalized to 'company' - 'Co' is part of many names.
    BEFORE: 'pvt. efs print ventures ltd.'
    AFTER:  'pvt efs print ventures ltd'
    BEFORE: 'international south consultants private ltd'
    AFTER:  'international south consultants pvt ltd'
    """
    # pvt. ltd. combined form first (before individual patterns)
    text = re.sub(r"pvt\.?\s*ltd\.?", "pvt ltd", text)
    text = re.sub(r"private\s+limited", "pvt ltd", text)
    text = re.sub(r"\blimited\b", "ltd", text)
    text = re.sub(r"\bcorporation\b", "corp", text)
    text = re.sub(r"\bincorporated\b", "inc", text)
    # French: keep sarl, sas, sci as-is (already lowercase)
    return text


def clean_business_name(raw: str) -> str:
    """
    business_name_clean: Unicode normalize + whitespace normalize only.
    Minimal transformation; all information preserved.
    COLUMNS AFFECTED: business_name_clean
    ROWS AFFECTED: Those with multi-whitespace or non-NFKC unicode.
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    text = normalize_unicode(raw)
    text = normalize_whitespace(text)
    return text


def normalize_business_name(clean: str) -> str:
    """
    business_name_normalized: Full normalization for entity matching.
    Pipeline: unicode (from _clean) -> lowercase -> punctuation -> suffixes -> whitespace.
    COLUMNS AFFECTED: business_name_normalized
    Original always preserved in 'business_name' column.
    """
    text = clean  # already unicode-normalized and whitespace-normalized
    text = normalize_casing(text)
    text = normalize_punctuation_name(text)
    text = normalize_legal_suffixes(text)
    text = normalize_whitespace(text)
    return text


def clean_business_address(raw: str) -> str:
    """
    business_address_clean: Unicode normalize + whitespace normalize only.
    Preserves: commas, hyphens, slashes, postal codes, all text.
    COLUMNS AFFECTED: business_address_clean
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    text = normalize_unicode(raw)
    text = normalize_whitespace(text)
    return text


def normalize_business_address(clean: str) -> str:
    """
    business_address_normalized: Full normalization for address matching.
    Pipeline: (from _clean) -> lowercase -> punctuation -> abbreviations -> whitespace.
    WHY (justified by observed data):
      - S2 addresses ALL-CAPS: 'KH NO. -570/13, NEW DELHI'
      - S3 addresses mixed case: 'Mack Rd, Haltom City, Texas'
      - Commas as separators: '1795 Westchester Drive, High Point, NC'
      - Slash notation: '3/115, East Delhi'
      - Observed French address components: 'Rue', 'Boulevard'
      - US: 'St', 'Rd', 'Ave', 'Blvd'
      - India: 'Nagar', 'Marg', 'Colony', 'Vihar'
    RISK: Medium for abbreviation expansion (could create false matches).
          'null' string observed in addresses -> preserved (informative missingness signal).
    BEFORE: 'KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi'
    AFTER:  'kh number -570 13 new delhi west delhi delhi'
    """
    text = clean
    text = normalize_casing(text)
    # Normalize separators: comma, semicolon -> space
    text = re.sub(r"[,;]", " ", text)
    # Normalize standalone slashes (not part of fractions like 1/2)
    # Keep slash in "19 1/2 Stardust" - only remove if surrounded by spaces
    text = re.sub(r"\s/\s", " ", text)
    # Remove standalone hyphens (surrounded by spaces)
    text = re.sub(r"\s+-\s+", " ", text)
    # Expand common US road abbreviations (confirmed in data)
    text = re.sub(r"\bst\b(?!ate)", "street", text)   # st but not state
    text = re.sub(r"\brd\b", "road", text)
    text = re.sub(r"\bave\b", "avenue", text)
    text = re.sub(r"\bblvd\b", "boulevard", text)
    text = re.sub(r"\bdr\b(?!ive)", "drive", text)     # dr but not drive
    text = re.sub(r"\bapt\b", "apartment", text)
    # Common address tokens
    text = re.sub(r"\bno\b\.?", "number", text)
    text = re.sub(r"\bfl\b(?!\w)", "floor", text)
    text = normalize_whitespace(text)
    return text


def clean_country(raw: str) -> str:
    """
    country_clean: Unicode normalize + whitespace normalize only.
    CRITICAL: NO vocabulary mapping. Works for ANY unseen country value.
    WHY: Problem statement requires open-set country treatment.
         France appears in test but not training.
    RISK: Zero - purely structural normalization.
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    text = normalize_unicode(raw)
    text = normalize_whitespace(text)
    return text


# =============================================================================
# SECTION 7: VECTORIZED PREPROCESSING (replaces row-by-row apply for speed)
# =============================================================================
# All transformations use pandas str methods (C-speed) instead of Python apply().
# Produces identical results to the scalar functions above but 10-50x faster.

def _vec_unicode_nfkc(s: pd.Series) -> pd.Series:
    """Vectorized NFKC unicode normalization."""
    return s.apply(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)


def preprocess_source_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline using VECTORIZED pandas str operations.
    Returns NEW dataframe with original columns preserved + new derived columns.
    NEVER modifies original columns.
    """
    print(f"\n  Preprocessing {source_name} ({len(df):,} rows)...", flush=True)
    df_out = df.copy()

    # --- Business Name ---
    if "business_name" in df_out.columns:
        raw = df_out["business_name"].fillna("").astype(str)

        # --- _clean: NFKC + whitespace collapse ---
        print(f"    [name] unicode + whitespace clean...", flush=True)
        s = _vec_unicode_nfkc(raw)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        df_out["business_name_clean"] = s

        # --- _normalized: lowercase + punctuation + legal suffixes ---
        print(f"    [name] normalizing (lower+punct+suffixes)...", flush=True)
        n = s.str.lower()
        # Strip leading noise: --, <<, >>, ==, ** etc.
        n = n.str.replace(r"^[-<>=*#!@~`|]+\s*", "", regex=True)
        # & -> and
        n = n.str.replace(r"\s*&\s*", " and ", regex=True)
        # Remove dots from abbreviations: P.V.T. -> PVT
        n = n.str.replace(r"([a-z])\.(?=[a-z])", r"\1", regex=True)
        n = n.str.replace(r"([a-z])\.$", r"\1", regex=True)
        n = n.str.replace(r"([a-z])\.\s", r"\1 ", regex=True)
        # Comma -> space
        n = n.str.replace(r",", " ", regex=False)
        # Multiple hyphens -> space
        n = n.str.replace(r"-{2,}", " ", regex=True)
        # Legal suffixes
        n = n.str.replace(r"pvt\.?\s*ltd\.?", "pvt ltd", regex=True)
        n = n.str.replace(r"private\s+limited", "pvt ltd", regex=True)
        n = n.str.replace(r"\blimited\b", "ltd", regex=True)
        n = n.str.replace(r"\bcorporation\b", "corp", regex=True)
        n = n.str.replace(r"\bincorporated\b", "inc", regex=True)
        # Final whitespace collapse
        n = n.str.replace(r"\s+", " ", regex=True).str.strip()
        df_out["business_name_normalized"] = n

        df_out["business_name_length"]      = raw.str.len().astype(int)
        df_out["business_name_token_count"] = raw.str.count(r"\S+").astype(int)

        n_changed_clean = int((df_out["business_name_clean"] != raw).sum())
        n_changed_norm  = int((df_out["business_name_normalized"] != raw.str.lower()).sum())
        print(f"    business_name_clean:      {n_changed_clean:,} records changed vs raw", flush=True)
        print(f"    business_name_normalized: {n_changed_norm:,} records changed vs raw.lower()", flush=True)

    # --- Business Address ---
    if "business_address" in df_out.columns:
        raw_addr = df_out["business_address"].fillna("").astype(str)

        # --- _clean: NFKC + whitespace ---
        print(f"    [addr] unicode + whitespace clean...", flush=True)
        a = _vec_unicode_nfkc(raw_addr)
        a = a.str.replace(r"\s+", " ", regex=True).str.strip()
        df_out["business_address_clean"] = a

        # --- _normalized: lowercase + separators + abbrev expansion ---
        print(f"    [addr] normalizing (lower+abbrevs)...", flush=True)
        an = a.str.lower()
        # comma/semicolon -> space
        an = an.str.replace(r"[,;]", " ", regex=True)
        # standalone slash surrounded by spaces
        an = an.str.replace(r"\s/\s", " ", regex=True)
        # standalone hyphen surrounded by spaces
        an = an.str.replace(r"\s+-\s+", " ", regex=True)
        # Road abbreviations (confirmed in data)
        an = an.str.replace(r"\bst\b(?!ate)", "street", regex=True)
        an = an.str.replace(r"\brd\b", "road", regex=True)
        an = an.str.replace(r"\bave\b", "avenue", regex=True)
        an = an.str.replace(r"\bblvd\b", "boulevard", regex=True)
        an = an.str.replace(r"\bdr\b(?!ive)", "drive", regex=True)
        an = an.str.replace(r"\bapt\b", "apartment", regex=True)
        an = an.str.replace(r"\bno\b\.?", "number", regex=True)
        an = an.str.replace(r"\bfl\b(?!\w)", "floor", regex=True)
        # Final whitespace collapse
        an = an.str.replace(r"\s+", " ", regex=True).str.strip()
        df_out["business_address_normalized"] = an

        df_out["business_address_length"]      = raw_addr.str.len().astype(int)
        df_out["business_address_token_count"] = raw_addr.str.count(r"\S+").astype(int)

        n_changed_addr = int((df_out["business_address_clean"] != raw_addr).sum())
        print(f"    business_address_clean:   {n_changed_addr:,} records changed vs raw", flush=True)

    # --- Country (open-set safe: NFKC + whitespace only) ---
    if "country" in df_out.columns:
        raw_country = df_out["country"].fillna("").astype(str)
        c = _vec_unicode_nfkc(raw_country)
        c = c.str.replace(r"\s+", " ", regex=True).str.strip()
        df_out["country_clean"] = c
        print(f"    country_clean: applied (open-set safe)", flush=True)

    print(f"    Output shape: {df_out.shape}", flush=True)
    print(f"    Output columns: {list(df_out.columns)}", flush=True)
    return df_out


# =============================================================================
# SECTION 8: VALIDATION
# =============================================================================

def validate_output(original: pd.DataFrame, processed: pd.DataFrame, name: str) -> list:
    """Post-processing validation to ensure no data was destroyed."""
    print(f"\n  Validating {name}...")
    issues = []

    # Row count must match exactly
    if len(original) != len(processed):
        issues.append(f"ROW COUNT MISMATCH: original={len(original):,} processed={len(processed):,}")
    else:
        print(f"    OK: Row count preserved ({len(processed):,})")

    # entity_id must be bit-for-bit identical
    if "entity_id" in original.columns:
        if not original["entity_id"].equals(processed["entity_id"]):
            issues.append("entity_id column has been modified!")
        else:
            print(f"    OK: entity_ids unchanged")

    # Every original column must exist unchanged
    for col in original.columns:
        if col not in processed.columns:
            issues.append(f"Original column '{col}' is MISSING in output!")
        elif not original[col].equals(processed[col]):
            issues.append(f"Original column '{col}' has been MODIFIED!")
        else:
            print(f"    OK: Original column '{col}' unchanged")

    # Derived columns must not contain NaN
    derived_cols = [c for c in processed.columns if c.endswith(("_clean", "_normalized"))]
    for col in derived_cols:
        n_nan = int(processed[col].isna().sum())
        if n_nan > 0:
            issues.append(f"Derived column '{col}' has {n_nan} NaN values!")
        else:
            print(f"    OK: {col} has no NaN")

    # No duplicate rows introduced beyond what was already there
    n_dup_orig = int(original.duplicated().sum())
    n_dup_proc = int(processed.duplicated().sum())
    if n_dup_proc > n_dup_orig:
        issues.append(f"Duplicate rows increased: {n_dup_orig} -> {n_dup_proc}")
    else:
        print(f"    OK: Duplicate rows not increased (was {n_dup_orig})")

    if issues:
        print(f"    VALIDATION ISSUES:")
        for issue in issues:
            print(f"      [FAIL] {issue}")
    else:
        print(f"    ALL CHECKS PASSED for {name}")

    return issues


# =============================================================================
# SECTION 9: MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "="*70)
    print("BUSINESS ENTITY RESOLUTION - EDA + PREPROCESSING PIPELINE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # -------------------------------------------------------------------------
    # LOAD ALL FILES
    # -------------------------------------------------------------------------
    data = load_all_files()

    # -------------------------------------------------------------------------
    # INTEGRITY CHECKS
    # -------------------------------------------------------------------------
    issues = integrity_checks(data)

    # -------------------------------------------------------------------------
    # EDA: Per-source analysis
    # -------------------------------------------------------------------------
    eda_stats = {}
    gt_stats = {}
    skip_eda = os.environ.get("SKIP_EDA", "0") == "1"
    
    if not skip_eda:
        for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
            if key not in data:
                print(f"\nSKIP EDA: {key} not available", flush=True)
                continue
            df = data[key]
            stats = describe_dataset(df, key)
            eda_stats[key] = stats
            analyze_string_lengths(df, "business_name", key)
            analyze_string_lengths(df, "business_address", key)
            analyze_country(df, key)
            analyze_business_names(df, key)
            analyze_addresses(df, key)

        # -------------------------------------------------------------------------
        # EDA: Ground Truth
        # -------------------------------------------------------------------------
        gt_stats = analyze_ground_truth(
            data["train_gt"],
            data["train_s2"],
            data["train_s3"]
        )

        # -------------------------------------------------------------------------
        # CROSS-SOURCE OBSERVATIONS
        # -------------------------------------------------------------------------
        cross_source_observations(data)
    else:
        print("\n[INFO] Skipping EDA (SKIP_EDA=1) -> Proceeding directly to Preprocessing.", flush=True)

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("APPLYING PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    print("""
Preprocessing steps per record:
  1. business_name_clean      = NFKC unicode normalize + whitespace normalize
  2. business_name_normalized = _clean + lowercase + punctuation norm + legal suffix norm
  3. business_name_length     = character count of raw name
  4. business_name_token_count = word count of raw name
  5. business_address_clean      = NFKC unicode normalize + whitespace normalize
  6. business_address_normalized = _clean + lowercase + punctuation norm + abbrev expand
  7. business_address_length     = character count of raw address
  8. business_address_token_count = word count of raw address
  9. country_clean = NFKC unicode normalize + whitespace normalize (open-set safe)
""")

    output_map = {
        "train_s1": OUTPUT_DIR / "train_source1_processed.tsv",
        "train_s2": OUTPUT_DIR / "train_source2_processed.tsv",
        "train_s3": OUTPUT_DIR / "train_source3_processed.tsv",
        "test_s1":  OUTPUT_DIR / "test_source1_processed.tsv",
        "test_s2":  OUTPUT_DIR / "test_source2_processed.tsv",
        "test_s3":  OUTPUT_DIR / "test_source3_processed.tsv",
    }

    all_issues = []
    processed_shapes = {}

    for key in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
        if key not in data:
            print(f"  SKIP: {key} not available", flush=True)
            continue
        
        # Preprocess
        df_proc = preprocess_source_df(data[key], key)
        processed_shapes[key] = df_proc.shape
        
        # Validate against original in-memory data
        print(f"\n  Validating {key}...", flush=True)
        v_issues = validate_output(data[key], df_proc, key)
        all_issues.extend(v_issues)
        
        # Save to disk immediately
        out_path = output_map[key]
        print(f"  Saving {key} -> {out_path.name}  shape={df_proc.shape} ...", flush=True)
        df_proc.to_csv(str(out_path), sep="\t", index=False, encoding="utf-8")
        print(f"    Saved.", flush=True)
        
        # Free memory immediately
        del df_proc
        del data[key]
        gc.collect()

    # Ground truth (copy as-is - no modification)
    gt_out = OUTPUT_DIR / "train_ground_truth.tsv"
    if "train_gt" in data:
        data["train_gt"].to_csv(str(gt_out), sep="\t", index=False, encoding="utf-8")
        print(f"  Saved: train_ground_truth.tsv  shape={data['train_gt'].shape}", flush=True)
        del data["train_gt"]
        gc.collect()

    # -------------------------------------------------------------------------
    # READ-BACK VERIFICATION
    # -------------------------------------------------------------------------
    print(f"\n  Verifying saved files can be read back...", flush=True)
    for key, out_path in output_map.items():
        if not out_path.exists():
            continue
        df_check = pd.read_csv(str(out_path), sep="\t", dtype=str,
                               keep_default_na=False, encoding="utf-8", nrows=10)
        expected_cols = [
            "entity_id", "business_name", "business_address", "country",
            "business_name_clean", "business_name_normalized", "business_name_length", "business_name_token_count",
            "business_address_clean", "business_address_normalized", "business_address_length", "business_address_token_count",
            "country_clean"
        ]
        check_cols = list(df_check.columns)
        if check_cols != expected_cols:
            print(f"  ERROR: {out_path.name} column mismatch! Got {check_cols}", flush=True)
        else:
            print(f"  OK: {out_path.name} readable, all 13 columns match", flush=True)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY REPORT
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")

    for label, shp in processed_shapes.items():
        print(f"  {label}: {shp[0]:,} rows x {shp[1]} cols", flush=True)

    print(f"\nPREPROCESSING PERFORMED", flush=True)
    print(f"  - Unicode normalization (NFKC) applied to name, address, country", flush=True)
    print(f"  - Whitespace normalization (collapse multi-spaces, strip edges)", flush=True)
    print(f"  - Lowercase conversion (in _normalized columns)", flush=True)
    print(f"  - Punctuation normalization: & -> and, leading --, <<, dots in abbrevs", flush=True)
    print(f"  - Legal suffix normalization: pvt.ltd./private limited -> pvt ltd, etc.", flush=True)
    print(f"  - Address abbreviation expansion: st->street, rd->road, ave->avenue, etc.", flush=True)
    print(f"  - Country handled as open-set (France supported without re-fitting)", flush=True)
    print(f"  - Original columns NEVER modified", flush=True)

    print(f"\nFILES CREATED", flush=True)
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = round(f.stat().st_size / 1e6, 2)
        print(f"  {f.name} ({size_mb} MB)", flush=True)

    print(f"\nVALIDATION CHECKS", flush=True)
    if all_issues:
        print(f"  FAILED - Issues:", flush=True)
        for issue in all_issues:
            print(f"    {issue}", flush=True)
    else:
        print(f"  ALL PASSED:", flush=True)
        print(f"    - Row counts preserved", flush=True)
        print(f"    - entity_ids unchanged", flush=True)
        print(f"    - Original columns unchanged", flush=True)
        print(f"    - No NaN in derived columns", flush=True)
        print(f"    - No extra duplicate rows introduced", flush=True)
        print(f"    - Files read back correctly", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("HANDOFF TO NEXT TEAMMATE", flush=True)
    print(f"{'='*70}", flush=True)
    print("""
COMPLETED BY THIS STAGE:
  [x] EDA (dataset-level, name, address, country, ground truth)
  [x] Data quality and integrity checks
  [x] Conservative normalization pipeline
  [x] Derived single-record features (length, token count, clean, normalized)
  [x] Processed train/test TSV files
  [x] Ground truth copy
  [x] Preprocessing documentation

NOT DONE - NEXT TEAMMATE RESPONSIBILITIES:
  [ ] Candidate generation / blocking
  [ ] Pair construction (S1 x S2, S1 x S3)
  [ ] Pairwise similarity features (Jaccard, Levenshtein, TF-IDF cosine, etc.)
  [ ] Matching model training and validation
  [ ] Threshold selection (optimize for F0.5)
  [ ] Final inference on test set
  [ ] matching_results.tsv generation

RECOMMENDED COLUMNS FOR BLOCKING:
  - business_name_normalized (token-based blocking)
  - country_clean (must-match for country-aware blocking)
  - business_address_normalized (address-based blocking)

SINGLETON HANDLING NOTE:
  Ground truth has singletons (S1 entities with no matches).
  Singletons must be predicted as empty match list.
  F0.5 rewards precision - be conservative on false merges.
""", flush=True)

    return processed_shapes, gt_stats, eda_stats


if __name__ == "__main__":
    processed_shapes, gt_stats, eda_stats = main()
