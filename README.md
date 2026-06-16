# Hallucination Detection for Educational LLMs

## Overview

A hybrid, model-agnostic hallucination detector that flags LLM responses before
they are shown to students. Combines character n-gram TF-IDF, word n-gram TF-IDF,
and 17 engineered linguistic features within a threshold-optimised Logistic
Regression framework.

**Research Question:** Can hallucinations be detected from (prompt, answer) surface
features before being displayed to students in educational platforms?

---

## Project Structure

```
Hallucination_Detection_Project_v2/
│
├── data/
│   ├── train.csv          ← 16,687 labeled QA pairs (Target: 0/1)
│   └── test.csv           ← 11,125 unlabeled QA pairs
│
├── code/
│   └── Hallucination_Detection.py   ← Full annotated pipeline (v2)
│   └── Dashboard.html 
│
├── outputs/
│   ├── Research_Paper.md            ← Full research paper
│   ├── full_results.json            ← All CV metrics, curves, EDA, predictions
│   └── test_probabilities.npy       ← Raw probability scores for test set
│
├── results/
│   └── result.csv                   ← Final test predictions (Id, Target)
│
├── README.md
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (from project root — data/ must be present)
python code/Hallucination_Detection.py
```

---

## Dataset

| Split | Rows   | Hallucination | Non-Hallucination | Imbalance |
|-------|--------|--------------|-------------------|-----------|
| Train | 16,687 | 894 (5.4%)   | 15,793 (94.6%)    | 17.7 : 1  |
| Test  | 11,125 | unlabeled    | unlabeled         | —         |

Columns: `Id`, `Prompt`, `Answer`, `Target` (train only)

---

## Method: Hybrid Detector

### Feature Sets

| Layer | Description | Size |
|-------|-------------|------|
| Char n-gram TF-IDF (3–5) | Sub-word texture — captures fabricated names, degenerate output, unusual sequences | 4,000 |
| Word n-gram TF-IDF (1–3) | Semantic/syntactic hallucination patterns | 3,000 |
| Engineered features | 17 handcrafted signals (see below) | 17 |
| **Total** | | **7,017** |

### 17 Engineered Features

| Feature | Signal |
|---------|--------|
| `char_len` | Character length of answer |
| `word_count` | Word count of answer |
| `len_ratio` | Answer length / Prompt length |
| `repetition_score` | 1 − (unique words / total words) |
| `keyword_overlap` | 4+ char words shared between prompt and answer |
| `is_very_short` | Binary flag: answer < 4 words |
| `hedge_count` | Count of "perhaps", "might", "probably" |
| `sentence_count` | Number of sentences in answer |
| `avg_sentence_len` | Mean words per sentence |
| `caps_ratio` | Fraction of uppercase characters |
| `question_marks` | Count of `?` in answer |
| `exclamations` | Count of `!` in answer |
| `year_count` | Count of 4-digit years (1900–2099) |
| `number_count` | Count of numeric tokens |
| `proper_noun_count` | Count of "Firstname Lastname" patterns |
| `refusal_phrases` | Count of "don't know", "cannot", "unclear" |
| `newlines` | Count of newline characters |

### Classifier

- Logistic Regression (`liblinear`, `class_weight='balanced'`, `C=1.0`)
- 5-fold stratified cross-validation
- Threshold optimized at **0.72** for maximum OOF F1

---

## Results

| Metric | Value |
|--------|-------|
| OOF ROC-AUC | **0.794 ± 0.010** |
| OOF F1 (hallucination) | **0.273** |
| OOF Average Precision | 0.226 |
| Precision (halluci.) | 0.25 |
| Recall (halluci.) | 0.30 |
| Overall accuracy | 0.91 |
| Test hallucinations flagged | 741 / 11,125 (6.7%) |

### Confusion Matrix (OOF, threshold=0.72)

|         | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| Actual 0 | 14,990 TN  | 803 FP      |
| Actual 1 | 626 FN     | 268 TP      |

### Per-Fold ROC-AUC

| Fold | ROC-AUC |
|------|---------|
| 1    | 0.7940  |
| 2    | 0.8044  |
| 3    | 0.7958  |
| 4    | 0.7747  |
| 5    | 0.8008  |

### Ablation

| Features | ROC-AUC |
|----------|---------|
| Engineered only | 0.52 |
| Word TF-IDF only | 0.53 |
| Char TF-IDF only | 0.791 |
| Char + Word | 0.792 |
| Char + Word + Engineered (full) | **0.794** |

---

## EDA Highlights

- **Class imbalance**: 17.7:1 — requires `class_weight='balanced'` and threshold tuning
- **Answer length**: Hallucinations have 79% longer mean char length (621 vs 346)
- **Repetition**: Hallucinations are 34% more repetitive (score: 0.379 vs 0.282)
- **Word count**: Bimodal distribution for hallucinations — many very short AND many very long

---

## Future Work

1. **Semantic consistency**: Generate N answers via sampling, measure embedding variance
2. **Retrieval grounding**: Cross-check facts against retrieved documents
3. **Model confidence**: Use token probability entropy when model internals are accessible
4. **Transformer fine-tuning**: DistilBERT/RoBERTa — expected ROC-AUC > 0.90
5. **Knowledge graph verification**: Validate named entities against Wikidata

---

## Publication Venues

- EDM / AIED — Educational Data Mining / AI in Education
- BEA Workshop at ACL
- EMNLP Findings
- ACL Student Research Workshop

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
```
