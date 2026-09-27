"""
run_experiments.py
===================
Executes the Member 4 Feature Engineering & Model Experimentation Pipeline.
Generates cached feature datasets, runs 5-Fold CV, tunes threshold for F0.5,
saves feature importances & model artifacts.
"""

import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features import FeatureBuilder
from src.model import Matcher, tune_threshold, f05_score

def main():
    start_time = time.time()
    print("=" * 70)
    print("MEMBER 4: FEATURE ENGINEERING & ML EXPERIMENTS PIPELINE")
    print("=" * 70)

    # 1. Feature Building
    print("\n[Step 1/4] Building / Loading Pairwise Train Features...")
    fb = FeatureBuilder()
    train_df = fb.build_train_features(cache=True, neg_pos_ratio=10)
    print(f"Train Feature Dataset Shape: {train_df.shape}")
    print(f"Total Positive Matches: {train_df['label'].sum():,}")
    print(f"Total Negative Pairs: {(train_df['label'] == 0).sum():,}")

    # 2. Model Training & 5-Fold CV
    print("\n[Step 2/4] Initializing Matcher & Running 5-Fold Stratified CV...")
    matcher = Matcher()
    matcher.train_df = train_df
    matcher.feature_cols = [c for c in train_df.columns if c.startswith("f_")]
    if "f_emb_name_cosine" in matcher.feature_cols and train_df["f_emb_name_cosine"].eq(-1.0).all():
        matcher.feature_cols.remove("f_emb_name_cosine")

    print(f"Using {len(matcher.feature_cols)} pairwise features:")
    for f in matcher.feature_cols:
        print(f" - {f}")

    cv_results = matcher.cross_validate()

    # 3. Final Model Training & Threshold Tuning
    print("\n[Step 3/4] Training Final Model & Performing Threshold Sweep...")
    matcher.train(run_cv=False)

    print("\nExecuting Precision-Focused Threshold Sweep for F0.5:")
    sweep_df = matcher.threshold_sweep()

    # Best threshold results
    best_row = sweep_df.iloc[0]
    print("\n" + "=" * 50)
    print("OPTIMAL OPERATING POINT FOR F0.5:")
    print(f"  Best Threshold : {best_row['threshold']:.3f}")
    print(f"  Validation F0.5: {best_row['f0.5']:.4f}")
    print(f"  Precision      : {best_row['precision']:.4f}")
    print(f"  Recall         : {best_row['recall']:.4f}")
    print("=" * 50)

    # 4. Summary & Execution Time
    elapsed = time.time() - start_time
    print(f"\nPipeline Execution Complete in {elapsed:.2f}s!")

if __name__ == "__main__":
    main()
