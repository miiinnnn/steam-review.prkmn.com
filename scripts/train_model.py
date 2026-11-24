"""
Step 2: Train a binary classifier (Bug & Optimization Issue vs Other)
on all labeled Steam reviews (105,000+) using clause-level TF-IDF +
Logistic Regression with review-level max-pooling, and export the portable
model to JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from label_reviews import CLAUSE_SPLIT_RE, is_bug_or_opt_clause  # noqa: E402

DATA_CSV = ROOT / "data" / "reviews_labeled.csv"
MODEL_DIR = ROOT / "model"
MODEL_JSON = MODEL_DIR / "bug_opt_classifier.json"
METRICS_JSON = MODEL_DIR / "metrics.json"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def extract_clauses(rows: list[dict[str, str]], indices: np.ndarray):
    clause_texts: list[str] = []
    clause_labels: list[int] = []
    review_map: list[int] = []
    for local_i, idx in enumerate(indices):
        raw = rows[int(idx)]["review_text"]
        parts = [c.strip() for c in CLAUSE_SPLIT_RE.split(raw) if c.strip()]
        if not parts:
            parts = [raw.strip()]
        for part in parts:
            clause_texts.append(normalize_text(part))
            clause_labels.append(1 if is_bug_or_opt_clause(part) else 0)
            review_map.append(local_i)
    return clause_texts, np.array(clause_labels, dtype=np.int32), np.array(review_map, dtype=np.int32)


def pool_review_probs(clause_probs: np.ndarray, review_map: np.ndarray, num_reviews: int) -> np.ndarray:
    out = np.zeros(num_reviews, dtype=np.float64)
    np.maximum.at(out, review_map, clause_probs)
    return out


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {DATA_CSV}")

    with DATA_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    y_rev = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    total = len(y_rev)
    pos_total = int(y_rev.sum())
    neg_total = int(total - pos_total)

    # Stratified review-level split: 64% train, 16% val, 20% test (no review leakage)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, test_idx = next(sss_test.split(rows, y_rev))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
    tr_rel, va_rel = next(sss_val.split(dev_idx, y_rev[dev_idx]))
    train_idx, val_idx = dev_idx[tr_rel], dev_idx[va_rel]

    tr_texts, tr_y, _ = extract_clauses(rows, train_idx)
    va_texts, _, va_map = extract_clauses(rows, val_idx)
    te_texts, _, te_map = extract_clauses(rows, test_idx)

    vec_eval = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=3,
        max_features=35000,
    )
    X_train = vec_eval.fit_transform(tr_texts)
    X_val = vec_eval.transform(va_texts)
    X_test = vec_eval.transform(te_texts)

    best_cfg = None
    best_val_f1 = -1.0
    best_model = None

    weight_candidates = [
        ("none", None),
        ("pos3", {0: 1, 1: 3}),
        ("pos5", {0: 1, 1: 5}),
        ("balanced", "balanced"),
    ]

    for w_name, w_val in weight_candidates:
        for c_val in [1.0, 3.0, 6.0, 12.0]:
            clf = LogisticRegression(
                solver="liblinear",
                class_weight=w_val,
                C=c_val,
                max_iter=500,
                random_state=42,
            )
            clf.fit(X_train, tr_y)
            val_probs = pool_review_probs(clf.predict_proba(X_val)[:, 1], va_map, len(val_idx))
            for th in np.arange(0.25, 0.90, 0.05):
                th_r = round(float(th), 2)
                preds = (val_probs >= th_r).astype(int)
                f1 = float(f1_score(y_rev[val_idx], preds, zero_division=0))
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_cfg = (w_name, w_val, c_val, th_r)
                    best_model = clf

    best_w_name, best_w_val, best_c, best_th = best_cfg
    val_probs = pool_review_probs(best_model.predict_proba(X_val)[:, 1], va_map, len(val_idx))
    val_preds = (val_probs >= best_th).astype(int)

    test_probs = pool_review_probs(best_model.predict_proba(X_test)[:, 1], te_map, len(test_idx))
    test_preds = (test_probs >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_rev[test_idx], test_preds).ravel()

    metrics = {
        "dataset": {
            "total_reviews": total,
            "bug_opt_count": pos_total,
            "other_count": neg_total,
            "bug_opt_ratio": round(pos_total / total, 4),
            "train_reviews": int(len(train_idx)),
            "val_reviews": int(len(val_idx)),
            "test_reviews": int(len(test_idx)),
        },
        "hyperparameters": {
            "method": "Clause-level TF-IDF + LogisticRegression with review-level max-pooling",
            "ngram_range": [2, 4],
            "class_weight": best_w_name,
            "C": best_c,
            "threshold": best_th,
        },
        "validation": {
            "f1": round(float(f1_score(y_rev[val_idx], val_preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_rev[val_idx], val_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_rev[val_idx], val_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_rev[val_idx], val_probs)), 4),
        },
        "test": {
            "accuracy": round(float(accuracy_score(y_rev[test_idx], test_preds)), 4),
            "precision": round(float(precision_score(y_rev[test_idx], test_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_rev[test_idx], test_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_rev[test_idx], test_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_rev[test_idx], test_probs)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "final_model_trained_on": total,
    }

    # Refit final production model on all labeled reviews
    all_idx = np.arange(total)
    all_texts, all_y, _ = extract_clauses(rows, all_idx)

    vec_full = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=3,
        max_features=35000,
    )
    X_full = vec_full.fit_transform(all_texts)
    clf_full = LogisticRegression(
        solver="liblinear",
        class_weight=best_w_val,
        C=best_c,
        max_iter=500,
        random_state=42,
    )
    clf_full.fit(X_full, all_y)

    portable = {
        "model_type": "binary_clause_tfidf_logreg",
        "trained_reviews": total,
        "trained_clauses": len(all_texts),
        "labels": {"0": "그 외 리뷰", "1": "버그 및 최적화 이슈"},
        "ngram_range": [2, 4],
        "threshold": best_th,
        "bias": round(float(clf_full.intercept_[0]), 6),
        "vocab": vec_full.get_feature_names_out().tolist(),
        "idf": [round(float(v), 6) for v in vec_full.idf_.tolist()],
        "weights": [round(float(w), 6) for w in clf_full.coef_[0].tolist()],
        "issue_tags": [
            {"tag": "버그·오류", "pattern": "버그|오류|에러|글리치|진행\\s*불가|퀘스트|맵\\s*뚫|끼임|bug|glitch"},
            {"tag": "튕김·강종", "pattern": "튕|팅|퉁겨|크래[시쉬]|강종|강제\\s*종료|충돌|꺼짐|꺼져|안\\s*켜|실행\\s*(?:불가|안)|crash"},
            {"tag": "프리징·멈춤", "pattern": "프리징|멈춤|멈춰|멈추|먹통|응답\\s*없음|블랙\\s*스크린|블루\\s*스크린|freeze"},
            {"tag": "프레임·최적화", "pattern": "최적화|발적화|프레임|fps|스터터링|드랍|드롭|사양|gpu|cpu|메모리|셰이더|쉐이더|고주사율|해상도|stutter|optimi"},
            {"tag": "렉·끊김", "pattern": "(?<![가-힣])[렉랙]|버벅|끊김|끊겨|끊기|지연|핑|패킷|lag"},
            {"tag": "로딩·저장·서버", "pattern": "로딩|세이브|저장|파일\\s*파손|초기화|서버|접속|연결"},
        ],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(
        json.dumps(portable, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    METRICS_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Trained on {total:,} reviews ({len(all_texts):,} clauses) | weight={best_w_name}, C={best_c}, threshold={best_th}",
        flush=True,
    )
    print(f"Test metrics: {json.dumps(metrics['test'], ensure_ascii=False)}", flush=True)
    print(f"Saved model ({MODEL_JSON.stat().st_size / 1024:.1f} KB) -> {MODEL_JSON}", flush=True)
    print(f"Saved metrics -> {METRICS_JSON}", flush=True)


if __name__ == "__main__":
    main()
