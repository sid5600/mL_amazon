# Business Entity Resolution — EDA & Preprocessed Data Handoff

> **Challenge:** Amazon ML Challenge — Business Entity Resolution  
> **Stage:** EDA + Data Preprocessing (Completed & Validated)  
> **Output Location:** `D:\amazon_ml_processed\`

---

## 1. Processed Datasets Summary

All 6 source datasets and the ground truth file have been preprocessed, validated against originals, and saved as TSVs:

| File Name | Split | Rows | Columns | File Size | Description |
|---|---|---|---|---|---|
| `train_source1_processed.tsv` | Train | 2,206,821 | 13 | 552.8 MB | Cleaned & normalized Source 1 train data |
| `train_source2_processed.tsv` | Train | 5,034,616 | 13 | 1,298.0 MB | Cleaned & normalized Source 2 train data |
| `train_source3_processed.tsv` | Train | 5,285,603 | 13 | 1,333.2 MB | Cleaned & normalized Source 3 train data |
| `test_source1_processed.tsv` | Test | 1,732,544 | 13 | 461.2 MB | Cleaned & normalized Source 1 test data |
| `test_source2_processed.tsv` | Test | 4,887,273 | 13 | 1,354.6 MB | Cleaned & normalized Source 2 test data |
| `test_source3_processed.tsv` | Test | 5,082,316 | 13 | 1,339.5 MB | Cleaned & normalized Source 3 test data |
| `train_ground_truth.tsv` | Train | 2,206,821 | 2 | 123.2 MB | Untouched ground truth entity mappings |

---

## 2. Output Schema (13 Columns per Record)

Each processed file contains **4 original columns (strictly unchanged)** + **9 engineered single-record features**:

| Column Name | Type | Description / Transformation |
|---|---|---|
| `entity_id` | `string` | **Original & Untouched** unique ID (`S1-xxx`, `S2-xxx`, `S3-xxx`) |
| `business_name` | `string` | **Original & Untouched** raw business name |
| `business_address` | `string` | **Original & Untouched** raw business address |
| `country` | `string` | **Original & Untouched** country code (US, India, France) |
| `business_name_clean` | `string` | Unicode NFKC normalized + collapsed whitespace |
| `business_name_normalized` | `string` | `_clean` + lowercased + legal suffixes normalized (`pvt ltd`, `inc`, `corp`) + punctuation cleaned |
| `business_name_length` | `int` | Character length of the raw business name |
| `business_name_token_count` | `int` | Word/token count of the raw business name |
| `business_address_clean` | `string` | Unicode NFKC normalized + collapsed whitespace |
| `business_address_normalized` | `string` | `_clean` + lowercased + road abbreviations expanded (`st`→`street`, `rd`→`road`, `ave`→`avenue`, `blvd`→`boulevard`, `dr`→`drive`) |
| `business_address_length` | `int` | Character length of the raw address |
| `business_address_token_count` | `int` | Word/token count of the raw address |
| `country_clean` | `string` | Unicode NFKC normalized + whitespace normalized (Open-set safe) |

---

## 3. Key Findings from EDA

1. **Open-Set Country Distribution:**
   - Train data contains: `US` and `India`.
   - Test data contains: `US`, `India`, and **`France`** (e.g. 170k+ French `Rue` addresses in test set).
   - Preprocessing does **not** assume fixed countries or hardcoded vocabularies.

2. **Multilingual Text & Scripts:**
   - Names and addresses contain Devanagari (Hindi), Kannada, French, and English characters.
   - All string normalization uses Unicode **NFKC** to preserve semantic equivalences across scripts.

3. **Ground Truth Structure & Singletons:**
   - Ground truth contains `source1_entity_id` and comma-separated `match_entity_ids` (e.g. `S2-1002,S3-509`).
   - **Singletons:** S1 entities with no matches in S2 or S3 have empty match strings. Singletons should be predicted as empty list.
   - The competition metric is **$F_{0.5}$ score**, which places higher weight on Precision than Recall (penalizes false matches).

---

## 4. Next Steps for Modeling & Matching Teammates

- [ ] **Candidate Generation / Blocking:**
  - Block on `country_clean` + first 3 letters / word tokens of `business_name_normalized`.
  - Inverted index / TF-IDF blocking on `business_name_normalized` tokens.
- [ ] **Pairwise Feature Engineering:**
  - Jaccard similarity, Levenshtein distance, token overlap ratio on normalized names and addresses.
  - TF-IDF cosine similarity.
  - Cross-lingual embedding cosine similarity (e.g. `multilingual-e5` or `sentence-transformers`).
- [ ] **Model Training & Evaluation:**
  - LightGBM / XGBoost / CatBoost pairwise binary classifier.
  - Tune classification threshold for maximum $F_{0.5}$ score.
- [ ] **Inference & Submission:**
  - Generate matching predictions for `test_source1.tsv` against `test_source2.tsv` and `test_source3.tsv`.
  - Format output as `matching_results.tsv`.
