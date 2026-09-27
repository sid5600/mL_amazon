"""
model.py -- ML Model Training, Threshold Tuning & Inference
=============================================================
Author : Member 4 (Feature Engineering + Experiments)
Challenge: Amazon ML Challenge -- Business Entity Resolution

PURPOSE
-------
1. Train a binary classifier on labelled pairwise features
2. Tune the decision threshold to maximize F0.5 score
   (F0.5 weights precision 4x more than recall -- penalizes false positives)
3. Run inference on all test candidate pairs
4. Output matching_results.tsv in the required submission format

METRIC
------
F0.5 = (1 + 0.5^2) * precision * recall / (0.5^2 * precision + recall)
     = 1.25 * precision * recall / (0.25 * precision + recall)
Higher precision matters more than higher recall.

MODEL CHOICE
------------
Primary  : scikit-learn RandomForest (always available)
Secondary: LightGBM if installed (pip install lightgbm)
           Automatically selected if available -- much faster on large data.

OUTPUTS
-------
output/matching_results.tsv  -- submission file
    columns: source1_entity_id | match_entity_ids (comma-separated)

HOW TO USE (command line)
-------------------------
  # Train + evaluate on train data:
  python src/model.py --mode train

  # Train + run full inference on test candidates:
  python src/model.py --mode train_and_predict

  # Load saved model + run inference only:
  python src/model.py --mode predict

HOW TO USE (Python API)
-----------------------
>>> from src.model import Matcher
>>> m = Matcher()
>>> m.load_features()            # loads train_features.parquet
>>> m.train()                    # trains model + tunes threshold
>>> m.predict_and_save()         # generates matching_results.tsv
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Feature cache from features.py
FEATURES_CACHE_DIR  = _REPO_ROOT / "output" / "features_cache"
TRAIN_FEATURES_PATH = FEATURES_CACHE_DIR / "train_features.parquet"
TEST_FEATURES_PATH  = FEATURES_CACHE_DIR / "test_features.parquet"

# Output paths
OUTPUT_DIR             = _REPO_ROOT / "output"
MATCHING_RESULTS_PATH  = OUTPUT_DIR / "matching_results.tsv"
MODEL_SAVE_PATH        = OUTPUT_DIR / "model.pkl"
THRESHOLD_SAVE_PATH    = OUTPUT_DIR / "best_threshold.json"

# Feature columns to use (all columns prefixed with 'f_')
# Set to None to auto-detect from the feature DataFrame
FEATURE_COLS: Optional[List[str]] = None

# Random Forest hyperparameters (tuned for F0.5 / precision-heavy task)
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",   # handles class imbalance
    "n_jobs": -1,
    "random_state": 42,
}

# LightGBM hyperparameters (used if lgbm is installed)
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "min_child_samples": 20,
    "scale_pos_weight": 10,       # approx neg/pos ratio
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# Threshold search range
THRESHOLD_LOW  = 0.05
THRESHOLD_HIGH = 0.95
THRESHOLD_STEP = 0.01

# =============================================================================
# METRIC
# =============================================================================

def f05_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)."""
    return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    low: float = THRESHOLD_LOW,
    high: float = THRESHOLD_HIGH,
    step: float = THRESHOLD_STEP,
) -> Tuple[float, float]:
    """
    Grid search over threshold values to maximize F0.5.

    Returns
    -------
    (best_threshold, best_f05)
    """
    best_t, best_f = 0.5, 0.0
    thresholds = np.arange(low, high + step, step)
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f05_score(y_true, preds)
        if score > best_f:
            best_f = score
            best_t = t
    return float(best_t), float(best_f)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def _try_lgbm():
    """Returns LightGBM classifier if installed, else None."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**LGBM_PARAMS)
    except ImportError:
        return None


def get_model(force_rf: bool = False):
    """
    Returns the best available classifier.
    LightGBM if installed, else scikit-learn RandomForest.
    """
    if not force_rf:
        lgbm = _try_lgbm()
        if lgbm is not None:
            print("[Model] Using LightGBM classifier.")
            return lgbm, "lightgbm"
    print("[Model] Using scikit-learn RandomForest classifier.")
    return RandomForestClassifier(**RF_PARAMS), "random_forest"


# =============================================================================
# MATCHER -- main API
# =============================================================================

class Matcher:
    """
    End-to-end training and inference API.

    Usage
    -----
    >>> m = Matcher()
    >>> m.load_features()          # loads pre-built feature parquet files
    >>> m.train()                  # train + cross-validate + tune threshold
    >>> m.predict_and_save()       # output matching_results.tsv
    """

    def __init__(
        self,
        train_features_path: Path = TRAIN_FEATURES_PATH,
        test_features_path: Path  = TEST_FEATURES_PATH,
        matching_output_path: Path = MATCHING_RESULTS_PATH,
        force_rf: bool = False,
        n_cv_folds: int = 5,
    ):
        self.train_features_path  = train_features_path
        self.test_features_path   = test_features_path
        self.matching_output_path = matching_output_path
        self.force_rf = force_rf
        self.n_cv_folds = n_cv_folds

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df:  Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []

        self.model = None
        self.model_name: str = ""
        self.best_threshold: float = 0.5
        self.cv_results: List[Dict] = []

    # -- Loading ---------------------------------------------------------------

    def load_features(self) -> "Matcher":
        """Load pre-built feature DataFrames from parquet, building them if missing."""
        try:
            from src.features import FeatureBuilder
        except ImportError:
            try:
                from features import FeatureBuilder
            except ImportError:
                FeatureBuilder = None

        if not self.train_features_path.exists():
            print(f"[Matcher] Train features not found at: {self.train_features_path}")
            if FeatureBuilder is not None:
                print("  [Matcher] Generating train features via FeatureBuilder...")
                fb = FeatureBuilder()
                self.train_df = fb.build_train_features(cache=True)
            else:
                print("  [Error] FeatureBuilder could not be imported.")
        else:
            print(f"[Matcher] Loading train features: {self.train_features_path}")
            self.train_df = pd.read_parquet(self.train_features_path)
            print(f"  Shape: {self.train_df.shape}")

        if not self.test_features_path.exists():
            print(f"[Matcher] Test features not found at: {self.test_features_path}")
            if FeatureBuilder is not None:
                print("  [Matcher] Generating test features via FeatureBuilder...")
                fb = FeatureBuilder()
                self.test_df = fb.build_test_features(cache=True)
            else:
                print("  [Error] FeatureBuilder could not be imported.")
        else:
            print(f"[Matcher] Loading test features:  {self.test_features_path}")
            self.test_df = pd.read_parquet(self.test_features_path)
            print(f"  Shape: {self.test_df.shape}")

        # Auto-detect feature columns
        if self.train_df is not None:
            self.feature_cols = [c for c in self.train_df.columns if c.startswith("f_")]
            # Drop embedding feature if it was never computed (all -1.0)
            if "f_emb_name_cosine" in self.feature_cols:
                if self.train_df["f_emb_name_cosine"].eq(-1.0).all():
                    self.feature_cols.remove("f_emb_name_cosine")
                    print("  [Info] Embedding feature dropped (not computed).")
            print(f"  Feature columns: {len(self.feature_cols)}")

        return self

    def load_model(self) -> "Matcher":
        """Load a previously saved model from disk."""
        with open(MODEL_SAVE_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(THRESHOLD_SAVE_PATH) as f:
            d = json.load(f)
            self.best_threshold = d["threshold"]
            self.model_name     = d.get("model_name", "unknown")
        print(f"[Matcher] Loaded model ({self.model_name}) from {MODEL_SAVE_PATH}")
        print(f"  Threshold: {self.best_threshold:.4f}")
        return self

    # -- Training + Evaluation -------------------------------------------------

    def _get_X_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols].fillna(0).values
        y = df["label"].values if "label" in df.columns else None
        return X, y

    def cross_validate(self) -> List[Dict]:
        """
        Stratified K-Fold CV to estimate model quality.
        Returns list of per-fold metrics.
        """
        if self.train_df is None:
            raise RuntimeError("Call load_features() first.")

        X, y = self._get_X_y(self.train_df)
        model, name = get_model(self.force_rf)
        self.model_name = name

        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        results = []

        print(f"\n[Matcher] {self.n_cv_folds}-fold cross-validation ({name})...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]

            best_t, best_f = tune_threshold(y_val, y_proba)
            preds = (y_proba >= best_t).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec  = recall_score(y_val, preds, zero_division=0)

            fold_result = {
                "fold": fold + 1,
                "threshold": round(best_t, 4),
                "f0.5": round(best_f, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
            results.append(fold_result)
            print(f"  Fold {fold+1}: threshold={best_t:.2f}  "
                  f"F0.5={best_f:.4f}  P={prec:.4f}  R={rec:.4f}")

        self.cv_results = results

        mean_f05  = np.mean([r["f0.5"]      for r in results])
        mean_prec = np.mean([r["precision"] for r in results])
        mean_rec  = np.mean([r["recall"]    for r in results])
        print(f"\n  CV Mean -> F0.5={mean_f05:.4f}  P={mean_prec:.4f}  R={mean_rec:.4f}")

        return results

    def train(self, run_cv: bool = True) -> "Matcher":
        """
        Full training pipeline:
        1. (Optional) Cross-validate to estimate performance
        2. Train on ALL training data
        3. Tune threshold on training data itself (final threshold)
        4. Save model + threshold to disk

        Parameters
        ----------
        run_cv : whether to run cross-validation before final training
        """
        if self.train_df is None:
            raise RuntimeError("Call load_features() first.")

        X, y = self._get_X_y(self.train_df)

        if run_cv and self.n_cv_folds > 1:
            self.cross_validate()

        # Final model: retrain on all data
        print(f"\n[Matcher] Training final model on all {len(X):,} training pairs...")
        self.model, self.model_name = get_model(self.force_rf)
        self.model.fit(X, y)
        print("  Training done.")

        # Tune threshold on training probabilities
        y_proba = self.model.predict_proba(X)[:, 1]
        best_t, best_f = tune_threshold(y, y_proba)
        self.best_threshold = best_t
        print(f"  Threshold (train): {best_t:.4f}  Train F0.5: {best_f:.4f}")
        print("  Note: final threshold may slightly overfit; CV threshold is more reliable.")

        # Feature importance
        self._print_feature_importance()

        # Save
        self._save()
        return self

    def _print_feature_importance(self, top_n: int = 15) -> None:
        """Print top feature importances."""
        if not hasattr(self.model, "feature_importances_"):
            return
        importances = self.model.feature_importances_
        paired = sorted(
            zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True
        )
        print(f"\n  Top {top_n} feature importances:")
        for name, imp in paired[:top_n]:
            bar = "#" * int(imp * 200)
            print(f"    {name:<35s} {imp:.4f}  {bar}")

    def _save(self) -> None:
        """Persist model + threshold to disk."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(THRESHOLD_SAVE_PATH, "w") as f:
            json.dump({
                "threshold":  self.best_threshold,
                "model_name": self.model_name,
                "cv_results": self.cv_results,
                "feature_cols": self.feature_cols,
            }, f, indent=2)
        print(f"\n  Model saved  -> {MODEL_SAVE_PATH}")
        print(f"  Threshold saved -> {THRESHOLD_SAVE_PATH}")

    # -- Inference -------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted match probability for each row in df."""
        if self.model is None:
            raise RuntimeError("Call train() or load_model() first.")
        X = df[self.feature_cols].fillna(0).values
        return self.model.predict_proba(X)[:, 1]

    def predict_and_save(
        self,
        threshold: Optional[float] = None,
        chunk_size: int = 500_000,
    ) -> pd.DataFrame:
        """
        Run inference on test features and write matching_results.tsv.

        Format of matching_results.tsv (tab-separated, no index):
            source1_entity_id   match_entity_ids
            S1-xxx              S2-aaa,S3-bbb
            S1-yyy              (empty -- singleton, no matches)

        Parameters
        ----------
        threshold  : decision threshold (uses self.best_threshold if None)
        chunk_size : process inference in chunks to control RAM

        Returns
        -------
        DataFrame of predictions (source1_entity_id, match_entity_ids)
        """
        if self.test_df is None:
            raise RuntimeError("Test features not loaded. Call load_features() first.")
        if self.model is None:
            raise RuntimeError("Call train() or load_model() first.")

        t = threshold if threshold is not None else self.best_threshold
        print(f"\n[Matcher] Running inference on {len(self.test_df):,} candidate pairs "
              f"(threshold={t:.4f})...")

        # Predict in chunks
        n = len(self.test_df)
        proba_chunks = []
        for start in range(0, n, chunk_size):
            chunk = self.test_df.iloc[start:start + chunk_size]
            proba_chunks.append(self.predict_proba(chunk))
            if (start // chunk_size) % 5 == 0:
                print(f"  Processed {min(start + chunk_size, n):,} / {n:,}")

        all_proba = np.concatenate(proba_chunks)
        self.test_df = self.test_df.copy()
        self.test_df["match_prob"]  = all_proba
        self.test_df["is_match"]    = (all_proba >= t).astype(int)

        total_matches = self.test_df["is_match"].sum()
        print(f"  Predicted matches: {total_matches:,} out of {n:,} pairs")
        print(f"  Match rate: {total_matches/n:.4f}")

        # Aggregate: for each S1, collect matched candidate IDs
        matched = self.test_df[self.test_df["is_match"] == 1].copy()
        agg = (
            matched.groupby("s1_id")["cand_id"]
            .apply(lambda ids: ",".join(sorted(ids)))
            .reset_index()
            .rename(columns={"s1_id": "source1_entity_id", "cand_id": "match_entity_ids"})
        )

        # All S1 IDs that appeared in candidate pairs (including singletons)
        all_s1_ids = pd.DataFrame(
            {"source1_entity_id": self.test_df["s1_id"].unique()}
        )

        result = all_s1_ids.merge(agg, on="source1_entity_id", how="left")
        result["match_entity_ids"] = result["match_entity_ids"].fillna("")

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(str(self.matching_output_path), sep="\t", index=False)
        print(f"\n  Saved: {self.matching_output_path}")
        print(f"  Total S1 entities: {len(result):,}")
        print(f"  S1 with >=1 match: {(result['match_entity_ids'] != '').sum():,}")
        print(f"  Singletons (0 matches): {(result['match_entity_ids'] == '').sum():,}")

        return result

    # -- Evaluation on labelled data -------------------------------------------

    def evaluate_train(self, threshold: Optional[float] = None) -> Dict:
        """
        Evaluate model performance on the training set (in-sample, optimistic).
        For true OOV estimate, use cross_validate() instead.
        """
        if self.train_df is None or self.model is None:
            raise RuntimeError("Call load_features() and train() first.")

        t = threshold if threshold is not None else self.best_threshold
        X, y = self._get_X_y(self.train_df)
        y_proba = self.model.predict_proba(X)[:, 1]
        preds = (y_proba >= t).astype(int)

        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f05  = f05_score(y, preds)

        print(f"\n[Matcher] Train-set evaluation (threshold={t:.4f}):")
        print(f"  F0.5      = {f05:.4f}")
        print(f"  Precision = {prec:.4f}")
        print(f"  Recall    = {rec:.4f}")
        print(f"  Predicted positives: {preds.sum():,} / {len(preds):,}")

        return {"f0.5": f05, "precision": prec, "recall": rec, "threshold": t}

    # -- Threshold Experiments -------------------------------------------------

    def threshold_sweep(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Show F0.5, precision, recall for a range of thresholds on training data.
        Useful for picking the right operating point.

        Returns a DataFrame sorted by F0.5 (descending).
        """
        if self.train_df is None or self.model is None:
            raise RuntimeError("Call load_features() and train() first.")

        X, y = self._get_X_y(self.train_df)
        y_proba = self.model.predict_proba(X)[:, 1]

        if thresholds is None:
            thresholds = list(np.arange(0.1, 0.95, 0.05))

        rows = []
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            prec = precision_score(y, preds, zero_division=0)
            rec  = recall_score(y, preds, zero_division=0)
            f05  = f05_score(y, preds)
            rows.append({
                "threshold": round(t, 3),
                "f0.5": round(f05, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "predicted_matches": int(preds.sum()),
            })

        sweep_df = pd.DataFrame(rows).sort_values("f0.5", ascending=False)
        print("\n[Matcher] Threshold sweep (sorted by F0.5):")
        print(sweep_df.to_string(index=False))
        return sweep_df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge -- Model Training & Inference"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "train_and_predict", "cv_only"],
        default="train_and_predict",
        help=(
            "train            = train model + evaluate (no test inference)\n"
            "predict          = load saved model + run inference\n"
            "train_and_predict= full pipeline\n"
            "cv_only          = cross-validate without saving model"
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override threshold for inference (default: auto-tuned)"
    )
    parser.add_argument(
        "--force-rf", action="store_true",
        help="Force scikit-learn RandomForest even if LightGBM is installed"
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip cross-validation during training"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Print threshold sweep table after training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    m = Matcher(force_rf=args.force_rf)
    m.load_features()

    if args.mode == "cv_only":
        m.cross_validate()
        return

    if args.mode in ("train", "train_and_predict"):
        m.train(run_cv=not args.no_cv)
        m.evaluate_train()
        if args.sweep:
            m.threshold_sweep()

    if args.mode in ("predict", "train_and_predict"):
        if args.mode == "predict":
            m.load_model()
        m.predict_and_save(threshold=args.threshold)

    print("\n[Done]")


if __name__ == "__main__":
    main()
