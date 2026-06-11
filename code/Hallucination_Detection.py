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
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve,
    classification_report
)
from scipy.sparse import hstack, csr_matrix

TRAIN_PATH    = 'data/train.csv'
TEST_PATH     = 'data/test.csv'
RESULTS_PATH  = 'results/result.csv'
ANALYSIS_PATH = 'outputs/full_results.json'
N_FOLDS       = 5
RANDOM_STATE  = 42

CHAR_MAX_FEATURES = 4000
WORD_MAX_FEATURES = 3000
CHAR_NGRAM_RANGE  = (3, 5)
WORD_NGRAM_RANGE  = (1, 3)

LR_C          = 1.0
LR_MAX_ITER   = 300

BANNER = "=" * 65


def load_data(train_path: str, test_path: str):
    print(BANNER)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print(BANNER)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    for df in [train, test]:
        df['Answer'] = df['Answer'].fillna('').astype(str)
        df['Prompt'] = df['Prompt'].fillna('').astype(str)

    y = train['Target'].values
    neg, pos = int((y == 0).sum()), int(y.sum())

    print(f"  Train samples      : {len(train):,}")
    print(f"  Test  samples      : {len(test):,}")
    print(f"  Positive (halluci.): {pos:,}  ({100 * y.mean():.1f}%)")
    print(f"  Negative (correct) : {neg:,}  ({100 * (1 - y.mean()):.1f}%)")
    print(f"  Class imbalance    : {neg / pos:.1f} : 1")
    print()

    return train, test, y


def run_eda(train: pd.DataFrame, y: np.ndarray) -> dict:
    print(BANNER)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print(BANNER)

    train = train.copy()
    train['ans_len']    = train['Answer'].str.len()
    train['ans_words']  = train['Answer'].str.split().str.len()
    train['repetition'] = train['Answer'].apply(
        lambda x: 1 - len(set(x.lower().split())) / (len(x.lower().split()) + 1)
    )

    print("  Answer length by class (words):")
    stats = train.groupby('Target')['ans_words'].agg(['mean', 'median', 'std'])
    for cls, row in stats.iterrows():
        label = "Hallucination" if cls == 1 else "Correct      "
        print(f"    {label} — mean={row['mean']:.1f}  median={row['median']:.0f}  std={row['std']:.1f}")

    print()
    print("  Repetition score by class:")
    for cls, val in train.groupby('Target')['repetition'].mean().items():
        label = "Hallucination" if cls == 1 else "Correct      "
        print(f"    {label} — {val:.4f}")

    print()
    print("  Sample hallucinations:")
    for _, row in train[train['Target'] == 1].sample(3, random_state=1).iterrows():
        print(f"    Prompt : {row['Prompt'][:80].strip()}...")
        print(f"    Answer : {row['Answer'][:80].strip()}...")
        print()

    eda = {
        'class_dist': {'0': int((y == 0).sum()), '1': int(y.sum())},
        'ans_len_mean':    {str(k): float(v) for k, v in train.groupby('Target')['ans_len'].mean().items()},
        'ans_words_mean':  {str(k): float(v) for k, v in train.groupby('Target')['ans_words'].mean().items()},
        'ans_words_median':{str(k): float(v) for k, v in train.groupby('Target')['ans_words'].median().items()},
        'rep_mean':        {str(k): float(v) for k, v in train.groupby('Target')['repetition'].mean().items()},
    }
    for cls in [0, 1]:
        sub = train[train['Target'] == cls]['ans_words'].clip(0, 250)
        hist, edges = np.histogram(sub, bins=15)
        eda[f'word_hist_{cls}'] = {
            'counts': hist.tolist(),
            'edges': [round(float(e), 1) for e in edges.tolist()]
        }

    return eda


FEATURE_NAMES = [
    'char_len', 'word_count', 'len_ratio', 'repetition_score', 'keyword_overlap',
    'is_very_short', 'hedge_count', 'sentence_count', 'avg_sentence_len',
    'caps_ratio', 'question_marks', 'exclamations', 'year_count',
    'number_count', 'proper_noun_count', 'refusal_phrases', 'newlines'
]

def extract_engineered_features(df: pd.DataFrame) -> csr_matrix:
    """
    17 handcrafted linguistic signals of hallucination:

    Lexical:     char_len, word_count, len_ratio (vs prompt)
    Semantic:    keyword_overlap (4+ char words shared with prompt)
    Stylistic:   repetition_score, caps_ratio, hedge_count, refusal_phrases
    Structural:  sentence_count, avg_sentence_len, is_very_short, newlines
    Content:     year_count, number_count, proper_noun_count
    Punctuation: question_marks, exclamations
    """
    ans = df['Answer'].astype(str)
    prm = df['Prompt'].astype(str)
    out = []

    for a, p in zip(ans, prm):
        wa  = str(a).lower().split()
        wp  = str(p).lower().split()
        pw4 = set(w for w in wp if len(w) >= 4)
        aw4 = set(w for w in wa if len(w) >= 4)
        sents = [s for s in re.split(r'[.!?]+', a) if s.strip()]
        avg_sl = float(np.mean([len(s.split()) for s in sents])) if sents else 0.0
        al = float(len(a)); pl = float(len(p))

        row = [
            al,                                                                   
            float(len(wa)),                                                       
            al / (pl + 1.0),                                                      
            1.0 - float(len(set(wa))) / (float(len(wa)) + 1.0),                  
            float(len(pw4 & aw4)) / (float(len(pw4)) + 1.0),                     
            float(int(len(wa) < 4)),                                             
            float(str(a).lower().count('perhaps') +
                  str(a).lower().count('might') +
                  str(a).lower().count('probably')),                             
            float(len(sents)),                                                   
            avg_sl,                                                              
            float(sum(c.isupper() for c in a)) / (al + 1.0),                     
            float(a.count('?')),                                                 
            float(a.count('!')),                                                 
            float(len(re.findall(r'\b(19|20)\d{2}\b', a))),                      
            float(len(re.findall(r'\d+', a))),                                   
            float(len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', a))),           
            float(str(a).lower().count("don't know") +
                  str(a).lower().count("cannot") +
                  str(a).lower().count("unclear")),                              
            float(a.count('\n')),                                                
        ]
        out.append(row)

    arr = np.array(out, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return csr_matrix(arr)


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    print(BANNER)
    print("STEP 3: FEATURE ENGINEERING")
    print(BANNER)

    print("  [1/3] Extracting engineered features...")
    X_eng  = extract_engineered_features(train)
    Xt_eng = extract_engineered_features(test)
    print(f"        {X_eng.shape[1]} features")

    print("  [2/3] Fitting character n-gram TF-IDF...")
    tv_char = TfidfVectorizer(
        max_features=CHAR_MAX_FEATURES, sublinear_tf=True,
        ngram_range=CHAR_NGRAM_RANGE, analyzer='char_wb', min_df=5
    )
    X_char  = tv_char.fit_transform(train['Answer'])
    Xt_char = tv_char.transform(test['Answer'])
    print(f"        {X_char.shape[1]} features (char {CHAR_NGRAM_RANGE}-grams)")

    print("  [3/3] Fitting word n-gram TF-IDF...")
    tv_word = TfidfVectorizer(
        max_features=WORD_MAX_FEATURES, sublinear_tf=True,
        ngram_range=WORD_NGRAM_RANGE, analyzer='word', min_df=3
    )
    X_word  = tv_word.fit_transform(train['Answer'])
    Xt_word = tv_word.transform(test['Answer'])
    print(f"        {X_word.shape[1]} features (word {WORD_NGRAM_RANGE}-grams)")

    X  = hstack([X_char, X_word, X_eng])
    Xt = hstack([Xt_char, Xt_word, Xt_eng])
    print(f"\n  Total combined features: {X.shape[1]:,}")
    print()

    feature_counts = {
        'char_ngrams': int(X_char.shape[1]),
        'word_ngrams': int(X_word.shape[1]),
        'engineered':  len(FEATURE_NAMES),
        'total':       int(X.shape[1])
    }

    return X, Xt, feature_counts


def cross_validate(X, y: np.ndarray):
    print(BANNER)
    print(f"STEP 4: {N_FOLDS}-FOLD STRATIFIED CROSS VALIDATION")
    print(BANNER)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_true, fold_stats = [], [], []

    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = LogisticRegression(
            C=LR_C, class_weight='balanced',
            max_iter=LR_MAX_ITER, solver='liblinear'
        )
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[val])[:, 1]

        roc = roc_auc_score(y[val], probs)
        f1  = f1_score(y[val], (probs > 0.5).astype(int))
        ap  = average_precision_score(y[val], probs)

        fold_stats.append({'fold': fold + 1, 'roc_auc': float(roc),
                           'f1': float(f1), 'avg_precision': float(ap)})
        all_probs.extend(probs)
        all_true.extend(y[val])
        print(f"  Fold {fold+1}: ROC-AUC={roc:.4f}  F1={f1:.4f}  AvgPrec={ap:.4f}")

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)

    oof_roc = roc_auc_score(all_true, all_probs)
    oof_ap  = average_precision_score(all_true, all_probs)
    roc_std = float(np.std([f['roc_auc'] for f in fold_stats]))

    print(f"\n  OOF ROC-AUC     : {oof_roc:.4f} ± {roc_std:.4f}")
    print(f"  OOF Avg Prec    : {oof_ap:.4f}")
    print()

    return all_probs, all_true, fold_stats, oof_roc, oof_ap, roc_std


def optimize_threshold(all_probs: np.ndarray, all_true: np.ndarray):
    print(BANNER)
    print("STEP 5: THRESHOLD OPTIMIZATION")
    print(BANNER)

    best_t, best_f1 = 0.5, 0.0
    sweep = []

    for t in np.arange(0.1, 0.91, 0.05):
        preds = (all_probs > t).astype(int)
        tp = int(((preds == 1) & (all_true == 1)).sum())
        fp = int(((preds == 1) & (all_true == 0)).sum())
        fn = int(((preds == 0) & (all_true == 1)).sum())
        p_ = tp / (tp + fp + 1e-9)
        r_ = tp / (tp + fn + 1e-9)
        f_ = 2 * p_ * r_ / (p_ + r_ + 1e-9)
        sweep.append({'threshold': round(float(t), 2), 'f1': float(f_),
                      'precision': float(p_), 'recall': float(r_)})
        if f_ > best_f1:
            best_f1, best_t = f_, t

    print(f"  Optimal threshold : {best_t:.2f}")
    print(f"  Best OOF F1       : {best_f1:.4f}")

    cm = confusion_matrix(all_true, (all_probs > best_t).astype(int))
    print()
    print("  Classification Report (OOF, optimal threshold):")
    print(classification_report(
    all_true,
    (all_probs > best_t).astype(int),
    target_names=['Correct', 'Hallucination']
    ))

    return best_t, best_f1, cm, sweep


def build_curves(all_probs: np.ndarray, all_true: np.ndarray):
    prec_c, rec_c, thr_c = precision_recall_curve(all_true, all_probs)
    step = max(1, len(thr_c) // 40)
    pr_data = [{'threshold': float(thr_c[i]), 'precision': float(prec_c[i]),
                'recall': float(rec_c[i])} for i in range(0, len(thr_c), step)]

    fpr, tpr, _ = roc_curve(all_true, all_probs)
    step2 = max(1, len(fpr) // 50)
    roc_data = [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])}
                for i in range(0, len(fpr), step2)]

    return pr_data, roc_data


def train_final_model(X, Xt, y, best_t, test):
    print(BANNER)
    print("STEP 6: FINAL MODEL & TEST PREDICTIONS")
    print(BANNER)

    model = LogisticRegression(
        C=LR_C, class_weight='balanced',
        max_iter=LR_MAX_ITER, solver='liblinear'
    )
    model.fit(X, y)

    test_probs = model.predict_proba(Xt)[:, 1]
    test_preds = (test_probs > best_t).astype(int)

    print(f"  Total test samples    : {len(test_preds):,}")
    print(f"  Predicted hallucinate : {test_preds.sum():,}  ({100 * test_preds.mean():.1f}%)")
    print()

    return model, test_probs, test_preds


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    train, test, y             = load_data(TRAIN_PATH, TEST_PATH)
    eda                        = run_eda(train, y)
    X, Xt, feat_counts         = build_feature_matrix(train, test)
    all_probs, all_true, folds, oof_roc, oof_ap, roc_std = cross_validate(X, y)
    best_t, best_f1, cm, sweep = optimize_threshold(all_probs, all_true)
    pr_data, roc_data          = build_curves(all_probs, all_true)
    model, test_probs, preds   = train_final_model(X, Xt, y, best_t, test)

    pd.DataFrame({'Id': test['Id'], 'Target': preds}).to_csv(RESULTS_PATH, index=False)
    np.save('outputs/test_probabilities.npy', test_probs)
    print(f"  Submission saved → {RESULTS_PATH}")

    analysis = {
        'oof_roc_auc':        float(oof_roc),
        'oof_roc_std':        float(roc_std),
        'oof_avg_precision':  float(oof_ap),
        'oof_f1_optimal':     float(best_f1),
        'optimal_threshold':  float(best_t),
        'confusion_matrix':   {'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
                               'FN': int(cm[1,0]), 'TP': int(cm[1,1])},
        'fold_stats':         folds,
        'pr_curve':           pr_data,
        'roc_curve':          roc_data,
        'threshold_sweep':    sweep,
        'eda':                eda,
        'test':               {'total': int(len(preds)), 'hallucinations': int(preds.sum()),
                               'rate': float(preds.mean())},
        'feature_counts':     feat_counts,
    }

    with open(ANALYSIS_PATH, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis JSON saved  → {ANALYSIS_PATH}")

    print()
    print(BANNER)
    print("PIPELINE COMPLETE")
    print(BANNER)
    print(f"  ROC-AUC  : {oof_roc:.4f} ± {roc_std:.4f}")
    print(f"  F1       : {best_f1:.4f}  (threshold={best_t:.2f})")
    print(f"  Avg Prec : {oof_ap:.4f}")
    print(f"  Predicted hallucinations in test: {preds.sum()} / {len(preds)}")
    print(BANNER)
