"""
Script to generate the 03_features_model.ipynb notebook.
Run: python3 create_notebook.py
"""

import json

cells = []

def code(src, tag=""):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag] if tag else []},
        "outputs": [],
        "source": src if isinstance(src, list) else [src]
    }

def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else [src]
    }

# ─────────────────────────────────────────
cells.append(md([
    "# 03 — Feature Engineering + Model Training\n",
    "**Amazon ML Challenge — Business Entity Resolution**  \n",
    "**Author: Member 4 (Feature Engineering + Experiments)**\n\n",
    "## Notebook Structure\n",
    "1. Setup & configuration\n",
    "2. Load entity lookup data\n",
    "3. Build pairwise features (train + test)\n",
    "4. Exploratory analysis of features\n",
    "5. Model training (scikit-learn RandomForest / LightGBM)\n",
    "6. Cross-validation + threshold tuning for F₀.₅\n",
    "7. Feature importance analysis\n",
    "8. Inference + generate `matching_results.tsv`\n",
    "9. Submission validation\n\n",
    "> **Metric**: F₀.₅ (precision-weighted) — precision counts 4× more than recall"
]))

# ─────────────────────────────────────────
cells.append(md("## 1. Setup & Configuration"))

cells.append(code([
    "import sys, os\n",
    "sys.path.insert(0, '..')  # add repo root to path\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "# ── Path overrides (edit if your data is elsewhere) ──────────────────────────\n",
    "# Where Member 2's processed TSVs are stored:\n",
    "os.environ['AMAZON_ML_PROCESSED_DIR'] = str(Path('../data/processed'))\n",
    "\n",
    "# Where raw dataset files (incl. train_ground_truth.tsv) are stored:\n",
    "os.environ['AMAZON_ML_DATASET_DIR'] = str(Path('../data/raw'))\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "from src.features import FeatureBuilder, CANDIDATE_PAIRS_PATH, FEATURES_CACHE_DIR\n",
    "from src.model import Matcher, f05_score, tune_threshold\n",
    "\n",
    "print('Setup complete.')\n",
    "print(f'candidate_pairs.tsv: {CANDIDATE_PAIRS_PATH}')\n",
    "print(f'Features cache dir : {FEATURES_CACHE_DIR}')\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 2. Load Entity Lookup Data\n\n> **Note**: This requires the processed TSVs from Member 2. If they aren't ready yet, skip to Section 3 which can load from cache."))

cells.append(code([
    "# Load processed entity records into an O(1) lookup\n",
    "fb = FeatureBuilder()\n",
    "fb.load_entity_data(split='both')   # loads train + test processed TSVs\n",
    "fb.load_candidate_pairs()            # loads + explodes candidate_pairs.tsv\n",
    "\n",
    "print(f'Entities in lookup : {len(fb.lookup):,}')\n",
    "print(f'Total candidate pairs: {len(fb._all_pairs):,}')\n",
    "fb._all_pairs.head()\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 3. Build Pairwise Features\n\n> First run takes **~20-60 min** depending on data size. Cached to Parquet after first run."))

cells.append(code([
    "# ── TRAIN FEATURES ──────────────────────────────────────────────────────────\n",
    "# sample_frac=0.3 means use 30% of candidate pairs for training (faster iteration)\n",
    "# Set sample_frac=None to use all pairs (more accurate but slower)\n",
    "fb.load_ground_truth()\n",
    "\n",
    "train_features = fb.build_train_features(\n",
    "    sample_frac=None,    # None = use all, 0.3 = use 30%\n",
    "    neg_pos_ratio=10,    # cap negatives at 10x positives\n",
    "    cache=True,          # saves to output/features_cache/train_features.parquet\n",
    ")\n",
    "\n",
    "print(f'\\nTrain feature shape: {train_features.shape}')\n",
    "print(f'Label distribution:\\n{train_features[\"label\"].value_counts()}')\n",
    "train_features.head(3)\n"
]))

cells.append(code([
    "# ── TEST FEATURES ───────────────────────────────────────────────────────────\n",
    "test_features = fb.build_test_features(cache=True)\n",
    "print(f'Test feature shape: {test_features.shape}')\n",
    "test_features.head(3)\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 4. Exploratory Analysis of Features"))

cells.append(code([
    "# Feature column names\n",
    "feat_cols = [c for c in train_features.columns if c.startswith('f_')]\n",
    "print(f'Number of features: {len(feat_cols)}')\n",
    "print(feat_cols)\n"
]))

cells.append(code([
    "# Distribution of each feature — split by label\n",
    "fig, axes = plt.subplots(4, 6, figsize=(22, 14))\n",
    "axes = axes.flatten()\n",
    "\n",
    "plot_cols = [c for c in feat_cols if 'emb' not in c and 'tfidf' not in c][:20]\n",
    "\n",
    "for i, col in enumerate(plot_cols):\n",
    "    ax = axes[i]\n",
    "    for lbl, color in [(0, 'steelblue'), (1, 'tomato')]:\n",
    "        subset = train_features[train_features['label'] == lbl][col].dropna()\n",
    "        ax.hist(subset, bins=30, alpha=0.5, color=color,\n",
    "                label=f'label={lbl}', density=True)\n",
    "    ax.set_title(col.replace('f_', ''), fontsize=8)\n",
    "    ax.set_xlabel('')\n",
    "    ax.legend(fontsize=6)\n",
    "\n",
    "for j in range(len(plot_cols), len(axes)):\n",
    "    axes[j].set_visible(False)\n",
    "\n",
    "plt.suptitle('Feature Distributions: Match (red) vs No-Match (blue)', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../output/feature_distributions.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved: output/feature_distributions.png')\n"
]))

cells.append(code([
    "# Correlation heatmap of features\n",
    "numeric_df = train_features[feat_cols].fillna(0)\n",
    "corr = numeric_df.corr()\n",
    "\n",
    "plt.figure(figsize=(14, 12))\n",
    "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
    "sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,\n",
    "            annot=False, fmt='.2f', linewidths=0.3,\n",
    "            xticklabels=True, yticklabels=True)\n",
    "plt.title('Feature Correlation Matrix', fontsize=13)\n",
    "plt.xticks(fontsize=7, rotation=45, ha='right')\n",
    "plt.yticks(fontsize=7)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../output/feature_correlation.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n"
]))

cells.append(code([
    "# Mean feature values per class — quick discriminative power check\n",
    "mean_by_label = train_features.groupby('label')[feat_cols].mean().T\n",
    "mean_by_label.columns = ['no_match (0)', 'match (1)']\n",
    "mean_by_label['delta'] = mean_by_label['match (1)'] - mean_by_label['no_match (0)']\n",
    "mean_by_label = mean_by_label.sort_values('delta', ascending=False)\n",
    "print('Feature means per class (sorted by match-vs-no-match delta):')\n",
    "mean_by_label.round(4)\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 5. Model Training\n\n> Uses LightGBM if installed (faster), otherwise scikit-learn RandomForest."))

cells.append(code([
    "matcher = Matcher()\n",
    "matcher.load_features()\n",
    "print(f'Features loaded: {len(matcher.feature_cols)} columns')\n"
]))

cells.append(code([
    "# Cross-validation to estimate real performance\n",
    "cv_results = matcher.cross_validate()\n",
    "\n",
    "cv_df = pd.DataFrame(cv_results)\n",
    "print('\\nCV Results:')\n",
    "print(cv_df.to_string(index=False))\n",
    "print(f\"\\nMean F0.5: {cv_df['f0.5'].mean():.4f} +/- {cv_df['f0.5'].std():.4f}\")\n",
    "print(f\"Mean Precision: {cv_df['precision'].mean():.4f}\")\n",
    "print(f\"Mean Recall: {cv_df['recall'].mean():.4f}\")\n"
]))

cells.append(code([
    "# Train final model on all training data\n",
    "matcher.train(run_cv=False)   # run_cv=False since we already ran it above\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 6. Threshold Tuning for F₀.₅\n\n> The threshold is critical! Lower threshold = more matches = higher recall but lower precision. \n> F₀.₅ penalizes false positives more, so we want a slightly conservative threshold."))

cells.append(code([
    "sweep_df = matcher.threshold_sweep()\n",
    "\n",
    "# Plot\n",
    "fig, ax1 = plt.subplots(figsize=(12, 5))\n",
    "ax2 = ax1.twinx()\n",
    "\n",
    "ax1.plot(sweep_df['threshold'], sweep_df['f0.5'],      color='purple', lw=2, label='F0.5')\n",
    "ax1.plot(sweep_df['threshold'], sweep_df['precision'],  color='steelblue', lw=1.5, ls='--', label='Precision')\n",
    "ax1.plot(sweep_df['threshold'], sweep_df['recall'],     color='tomato', lw=1.5, ls='--', label='Recall')\n",
    "ax2.bar(sweep_df['threshold'], sweep_df['predicted_matches'],\n",
    "        width=0.03, alpha=0.2, color='gray', label='# Predicted Matches')\n",
    "\n",
    "best_row = sweep_df.loc[sweep_df['f0.5'].idxmax()]\n",
    "ax1.axvline(best_row['threshold'], color='purple', ls=':', alpha=0.7)\n",
    "ax1.text(best_row['threshold'] + 0.01, best_row['f0.5'],\n",
    "         f\"Best t={best_row['threshold']}\\nF0.5={best_row['f0.5']:.4f}\",\n",
    "         fontsize=9, color='purple')\n",
    "\n",
    "ax1.set_xlabel('Threshold')\n",
    "ax1.set_ylabel('Score')\n",
    "ax2.set_ylabel('# Predicted Matches', color='gray')\n",
    "ax1.legend(loc='upper left')\n",
    "ax2.legend(loc='upper right')\n",
    "plt.title('Threshold Sweep — F0.5, Precision, Recall')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../output/threshold_sweep.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Best threshold: {best_row[\"threshold\"]}  F0.5: {best_row[\"f0.5\"]:.4f}')\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 7. Feature Importance Analysis"))

cells.append(code([
    "if hasattr(matcher.model, 'feature_importances_'):\n",
    "    importances = matcher.model.feature_importances_\n",
    "    imp_df = pd.DataFrame({'feature': matcher.feature_cols, 'importance': importances})\n",
    "    imp_df = imp_df.sort_values('importance', ascending=False)\n",
    "\n",
    "    plt.figure(figsize=(10, 7))\n",
    "    sns.barplot(data=imp_df.head(20), x='importance', y='feature',\n",
    "                palette='viridis_r')\n",
    "    plt.title('Top 20 Feature Importances')\n",
    "    plt.xlabel('Importance')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('../output/feature_importance.png', dpi=120, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "    print(imp_df.to_string(index=False))\n",
    "else:\n",
    "    print('Model does not expose feature_importances_ (e.g., LogisticRegression).')\n",
    "    print('Use SHAP or permutation importance instead.')\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 8. Inference — Generate matching_results.tsv"))

cells.append(code([
    "# Run inference on ALL test candidate pairs\n",
    "results_df = matcher.predict_and_save(\n",
    "    threshold=None,   # uses best_threshold from training\n",
    ")\n",
    "\n",
    "print('\\nSample output:')\n",
    "results_df.head(10)\n"
]))

cells.append(code([
    "# Sanity checks on output\n",
    "print('Output shape:', results_df.shape)\n",
    "print('Columns:', list(results_df.columns))\n",
    "print('\\nMatch distribution:')\n",
    "match_counts = results_df['match_entity_ids'].apply(\n",
    "    lambda x: len(x.split(',')) if x.strip() else 0\n",
    ")\n",
    "print(match_counts.describe())\n",
    "print(f'\\nS1 with 0 matches (singletons): {(match_counts == 0).sum():,}')\n",
    "print(f'S1 with 1+ matches: {(match_counts > 0).sum():,}')\n",
    "print(f'Max matches per S1: {match_counts.max()}')\n",
    "\n",
    "# Check format is correct\n",
    "assert list(results_df.columns) == ['source1_entity_id', 'match_entity_ids'], \\\n",
    "    f'Wrong columns: {list(results_df.columns)}'\n",
    "print('\\n[OK] Output format is correct.')\n"
]))

# ─────────────────────────────────────────
cells.append(md("## 9. Submission Validation\n\n> Run the official validator before submitting."))

cells.append(code([
    "import subprocess\n",
    "\n",
    "# Run the official validator\n",
    "# Adjust paths if your dataset is in a different location\n",
    "validator_cmd = [\n",
    "    'python3', '../utils/validate_submission.py',\n",
    "    '--matching', '../output/matching_results.tsv',\n",
    "    '--candidate', '../output/candidate_pairs.tsv',\n",
    "    '--test-dir', '../data/raw/test',\n",
    "]\n",
    "\n",
    "print('Running validator...')\n",
    "print(' '.join(validator_cmd))\n",
    "\n",
    "result = subprocess.run(validator_cmd, capture_output=True, text=True)\n",
    "print(result.stdout)\n",
    "if result.returncode == 0:\n",
    "    print('[PASS] Validation passed!')\n",
    "else:\n",
    "    print('[FAIL] Validation failed:')\n",
    "    print(result.stderr)\n"
]))

cells.append(code([
    "# Quick manual format check (no validator needed)\n",
    "results_df = pd.read_csv('../output/matching_results.tsv', sep='\\t', dtype=str,\n",
    "                         keep_default_na=False)\n",
    "\n",
    "print('=== matching_results.tsv Format Check ===')\n",
    "print(f'Rows: {len(results_df):,}')\n",
    "print(f'Columns: {list(results_df.columns)}')\n",
    "print(f'All source1_entity_id start with S1-: '\n",
    "      f\"{results_df['source1_entity_id'].str.startswith('S1-').all()}\")\n",
    "\n",
    "# Check match IDs look right\n",
    "non_empty = results_df[results_df['match_entity_ids'] != '']\n",
    "sample_matches = non_empty['match_entity_ids'].head(3)\n",
    "print('\\nSample match IDs:')\n",
    "for m in sample_matches:\n",
    "    print(' ', m[:80])\n",
    "\n",
    "print('\\n=== Ready for submission ===')\n"
]))

# ─────────────────────────────────────────
cells.append(md([
    "## Summary\n\n",
    "| Step | Status |\n",
    "|------|--------|\n",
    "| Feature engineering | ✅ |\n",
    "| Model training | ✅ |\n",
    "| Cross-validation | ✅ |\n",
    "| Threshold tuning (F₀.₅) | ✅ |\n",
    "| Inference | ✅ |\n",
    "| matching_results.tsv | ✅ |\n\n",
    "**Hand-off to Member 1 (submission):**\n",
    "- `output/matching_results.tsv` — upload to leaderboard portal\n",
    "- `output/candidate_pairs.tsv` — include in final ZIP\n",
    "- `src/features.py` + `src/model.py` — include in `code/` in final ZIP\n"
]))

# ─────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = "notebooks/03_features_model.ipynb"
import json
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
