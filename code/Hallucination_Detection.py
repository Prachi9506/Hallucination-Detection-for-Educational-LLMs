import pandas as pd
import numpy as np
import re
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    average_precision_score, confusion_matrix
)
from scipy.sparse import hstack, csr_matrix

print("=" * 60)
print("STEP 1: DATA LOADING & EXPLORATION")
print("=" * 60)

train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')

for df in [train, test]:
    df['Answer'] = df['Answer'].fillna('').astype(str)
    df['Prompt'] = df['Prompt'].fillna('').astype(str)

y = train['Target'].values

print(f"Train samples  : {len(train):,}")
print(f"Test samples   : {len(test):,}")
print(f"Positive (hall): {y.sum():,}  ({100*y.mean():.1f}%)")
print(f"Negative       : {(y==0).sum():,}  ({100*(1-y.mean()):.1f}%)")
print(f"Imbalance ratio: {(y==0).sum()/y.sum():.1f}:1")
print()

print("─── Sample Hallucinations ───")
hall_df = train[train['Target'] == 1].sample(3, random_state=1)
for _, row in hall_df.iterrows():
    print(f"  Prompt : {row['Prompt'][:100].strip()}...")
    print(f"  Answer : {row['Answer'][:100].strip()}...")
    print()

print("=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

def extract_engineered_features(df):
    """
    Hallucination signal features:
    - Lexical: length, word count, length ratio to prompt
    - Semantic: keyword overlap between prompt and answer
    - Stylistic: repetition score, hedging language, negations
    - Structural: sentence count, answer brevity
    """
    ans = df['Answer'].astype(str)
    prm = df['Prompt'].astype(str)
    out = []
    for a, p in zip(ans, prm):
        wa = str(a).lower().split()
        wp = str(p).lower().split()
        pw = set(w for w in wp if len(w) >= 4)
        aw = set(w for w in wa if len(w) >= 4)
        row = [
            len(a),                                                  
            len(wa),                                                 
            len(a) / (len(p) + 1),                                  
            1 - len(set(wa)) / (len(wa) + 1),                       
            len(pw & aw) / (len(pw) + 1),                           
            int(len(wa) < 4),                                       
            str(a).lower().count('perhaps') +
            str(a).lower().count('might') +
            str(a).lower().count('probably'),                       
        ]
        out.append(row)
    return csr_matrix(np.array(out, dtype=np.float32))

feature_names = [
    'ans_char_len', 'ans_word_count', 'len_ratio',
    'repetition_score', 'keyword_overlap', 'is_very_short', 'hedge_count'
]

print("Extracting engineered features...")
X_eng  = extract_engineered_features(train)
Xt_eng = extract_engineered_features(test)
print(f"  Engineered features: {len(feature_names)}")

print()
print("Fitting TF-IDF vectorizers...")

tv_char = TfidfVectorizer(
    max_features=4000, sublinear_tf=True,
    ngram_range=(3, 5), analyzer='char_wb', min_df=5
)

tv_word = TfidfVectorizer(
    max_features=3000, sublinear_tf=True,
    ngram_range=(1, 3), analyzer='word', min_df=3
)

X_char  = tv_char.fit_transform(train['Answer'])
Xt_char = tv_char.transform(test['Answer'])
X_word  = tv_word.fit_transform(train['Answer'])
Xt_word = tv_word.transform(test['Answer'])

print(f"  Char n-gram features : {X_char.shape[1]:,}")
print(f"  Word n-gram features : {X_word.shape[1]:,}")

X  = hstack([X_char, X_word, X_eng])
Xt = hstack([Xt_char, Xt_word, Xt_eng])
print(f"  Total features       : {X.shape[1]:,}")

print()
print("=" * 60)
print("STEP 3: 5-FOLD CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_probs, all_true = [], []
fold_rocs, fold_f1s = [], []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LogisticRegression(
        C=1.0, class_weight='balanced',
        max_iter=300, solver='liblinear'
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, probs)
    f1  = f1_score(y_val, (probs > 0.5).astype(int))
    fold_rocs.append(roc)
    fold_f1s.append(f1)
    all_probs.extend(probs)
    all_true.extend(y_val)
    print(f"  Fold {fold+1}: ROC-AUC={roc:.4f}  F1={f1:.4f}")

all_probs = np.array(all_probs)
all_true  = np.array(all_true)

oof_roc = roc_auc_score(all_true, all_probs)
oof_ap  = average_precision_score(all_true, all_probs)
print(f"\n  OOF ROC-AUC : {oof_roc:.4f} ± {np.std(fold_rocs):.4f}")
print(f"  OOF Avg Prec: {oof_ap:.4f}")

print()
print("=" * 60)
print("STEP 4: THRESHOLD OPTIMIZATION")
print("=" * 60)
print("(Optimizing F1 on OOF predictions)")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.2, 0.85, 0.05):
    f1 = f1_score(all_true, (all_probs > t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"  Optimal threshold  : {best_t:.2f}")
print(f"  Best OOF F1        : {best_f1:.4f}")
print()
print("─── Classification Report (OOF, optimal threshold) ───")
print(classification_report(
    all_true, (all_probs > best_t).astype(int),
    target_names=['Not Hallucination', 'Hallucination']
))

cm = confusion_matrix(all_true, (all_probs > best_t).astype(int))
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

print()
print("=" * 60)
print("STEP 5: FINAL MODEL & TEST PREDICTIONS")
print("=" * 60)

final_model = LogisticRegression(
    C=1.0, class_weight='balanced', max_iter=300, solver='liblinear'
)
final_model.fit(X, y)
test_probs = final_model.predict_proba(Xt)[:, 1]
test_preds = (test_probs > best_t).astype(int)

submission = pd.DataFrame({'Id': test['Id'], 'Target': test_preds})
submission.to_csv('../outputs/submission.csv', index=False)
np.save('../outputs/test_probs.npy', test_probs)

print(f"  Test samples         : {len(test_preds):,}")
print(f"  Predicted hallucinate: {test_preds.sum():,} ({100*test_preds.mean():.1f}%)")
print(f"  Submission saved     : submission.csv")

results = {
    "project": "Hallucination Detection for Educational LLMs",
    "model": "Logistic Regression (char+word TF-IDF + engineered features)",
    "features": {
        "char_ngrams_35": 4000,
        "word_ngrams_13": 3000,
        "engineered": len(feature_names),
        "total": int(X.shape[1])
    },
    "cv_results": {
        "oof_roc_auc": float(oof_roc),
        "oof_roc_std": float(np.std(fold_rocs)),
        "oof_avg_precision": float(oof_ap),
        "oof_f1_at_05": float(f1_score(all_true, (all_probs > 0.5).astype(int))),
        "oof_f1_optimal": float(best_f1),
        "optimal_threshold": float(best_t),
        "fold_roc_aucs": [float(x) for x in fold_rocs],
    },
    "test_predictions": {
        "total": int(len(test_preds)),
        "predicted_hallucinations": int(test_preds.sum()),
        "hallucination_rate": float(test_preds.mean())
    }
}

with open('../outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print("=" * 60)
print("COMPLETE. All artifacts saved.")
print("=" * 60)
