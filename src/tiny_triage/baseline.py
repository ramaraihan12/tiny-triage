"""TF-IDF (word + char n-grams) and class-weighted logistic regression."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .labels import CATEGORIES, NEEDS_INFO

C_GRID = (0.5, 2.0, 8.0)
TARGETS = {"category": CATEGORIES, "needs_info": NEEDS_INFO}


def vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True, lowercase=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=200_000, sublinear_tf=True
    )
    return word, char


def _select(rows, split_key, split, target):
    sel = [r for r in rows if r[split_key] == split and r[target] is not None]
    return [r["text"] for r in sel], [r[target] for r in sel]


class Fitted:
    """TF-IDF vectorizers and the chosen logistic regression for one target."""

    def __init__(self, word, char, model, best_c, val_scores, classes, models=None):
        self.word, self.char, self.model = word, char, model
        self.models = models or {}  # every C that was tried, for joint threshold tuning
        self.best_c, self.val_scores, self.classes = best_c, val_scores, tuple(classes)

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def proba(self, texts, c: float | None = None) -> np.ndarray:
        """Class probabilities with columns in the order of the target's class tuple.

        Uses the chosen C unless another tried C is asked for.
        """
        model = self.model if c is None else self.models[c]
        p = model.predict_proba(self.transform(texts))
        order = [list(model.classes_).index(k) for k in self.classes]
        return p[:, order]


def fit_target(rows: list[dict], split_key: str, target: str, seed: int = 0) -> Fitted:
    """Fit on train and pick C on val only."""
    classes = TARGETS[target]
    x_tr, y_tr = _select(rows, split_key, "train", target)
    x_va, y_va = _select(rows, split_key, "val", target)
    word, char = vectorizers()
    m_tr = hstack([word.fit_transform(x_tr), char.fit_transform(x_tr)]).tocsr()
    m_va = hstack([word.transform(x_va), char.transform(x_va)]).tocsr()
    val_scores = {}
    best_c, best_model, best_f1 = None, None, -1.0
    models = {}
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=400, random_state=seed)
        model.fit(m_tr, y_tr)
        f1 = f1_score(y_va, model.predict(m_va), average="macro", labels=list(classes))
        val_scores[str(c)] = round(float(f1), 4)
        models[c] = model
        if f1 > best_f1:
            best_c, best_model, best_f1 = c, model, f1
    return Fitted(word, char, best_model, best_c, val_scores, classes, models)


def predict_split(
    rows: list[dict], split_key: str = "split_repo", seed: int = 0, alt_texts: dict | None = None
) -> dict:
    """Per-issue probabilities of the same models run_split reports, for every val and test row.

    alt_texts may hold {"val": [...], "test": [...]} replacement texts (same row order); the
    fitted model scores those too, under "alt_val" and "alt_test".
    """
    val = [r["text"] for r in rows if r[split_key] == "val"]
    test = [r["text"] for r in rows if r[split_key] == "test"]
    out = {}
    for target in TARGETS:
        fit = fit_target(rows, split_key, target, seed)
        res = {"chosen_C": fit.best_c, "val": fit.proba(val), "test": fit.proba(test)}
        if alt_texts:
            res["alt_val"] = fit.proba(alt_texts["val"])
            res["alt_test"] = fit.proba(alt_texts["test"])
        if target == "needs_info":
            res["by_c"] = {
                c: {
                    "val": fit.proba(val, c),
                    "test": fit.proba(test, c),
                    **(
                        {
                            "alt_val": fit.proba(alt_texts["val"], c),
                            "alt_test": fit.proba(alt_texts["test"], c),
                        }
                        if alt_texts
                        else {}
                    ),
                }
                for c in C_GRID
            }
        out[target] = res
        print(f"baseline {target}: fitted (C={fit.best_c})", flush=True)
    return out


def run_split(rows: list[dict], split_key: str, seed: int = 0) -> dict:
    """Fit on train, pick C on val only, report on test. One result per target."""
    results: dict = {}
    for target, classes in TARGETS.items():
        x_tr, y_tr = _select(rows, split_key, "train", target)
        x_va, y_va = _select(rows, split_key, "val", target)
        x_te, y_te = _select(rows, split_key, "test", target)
        fit = fit_target(rows, split_key, target, seed)
        best_c, val_scores = fit.best_c, fit.val_scores
        pred = fit.model.predict(fit.transform(x_te))
        majority = Counter(y_tr).most_common(1)[0][0]
        maj_pred = [majority] * len(y_te)
        labels = list(classes)
        report = classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0)
        results[target] = {
            "n_train": len(y_tr),
            "n_val": len(y_va),
            "n_test": len(y_te),
            "test_class_counts": dict(Counter(y_te)),
            "chosen_C": best_c,
            "val_macro_f1_by_C": val_scores,
            "test_macro_f1": round(float(f1_score(y_te, pred, average="macro", labels=labels)), 4),
            "test_accuracy": round(float(np.mean(np.array(pred) == np.array(y_te))), 4),
            "majority_class": majority,
            "majority_macro_f1": round(
                float(f1_score(y_te, maj_pred, average="macro", labels=labels, zero_division=0)), 4
            ),
            "per_class": {
                c: {
                    "precision": round(report[c]["precision"], 4),
                    "recall": round(report[c]["recall"], 4),
                    "f1": round(report[c]["f1-score"], 4),
                    "support": int(report[c]["support"]),
                }
                for c in labels
            },
            "confusion_matrix": {
                "labels": labels,
                "rows_true_cols_pred": confusion_matrix(y_te, pred, labels=labels).tolist(),
            },
        }
        print(f"{split_key} {target}: macro-F1 {results[target]['test_macro_f1']}", flush=True)
    return results


def markdown(results: dict) -> str:
    out = []
    names = {
        "repo_split": "Repo-disjoint split (honest)",
        "in_repo": "In-repo random split (optimistic)",
    }
    for key, res in results["splits"].items():
        out.append(f"### {names[key]}\n")
        out.append(
            "| output | test n | baseline (majority) macro-F1 | TF-IDF + LR macro-F1 | accuracy |"
        )
        out.append("|---|---|---|---|---|")
        for t, r in res.items():
            out.append(
                f"| {t} | {r['n_test']} | {r['majority_macro_f1']} | {r['test_macro_f1']} "
                f"| {r['test_accuracy']} |"
            )
        out.append("")
        for t, r in res.items():
            out.append(f"{t}, per class on test:\n")
            out.append("| class | precision | recall | F1 | support |")
            out.append("|---|---|---|---|---|")
            for c, m in r["per_class"].items():
                out.append(
                    f"| {c} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |"
                )
            cm = r["confusion_matrix"]
            out.append("")
            out.append("confusion matrix (rows true, columns predicted):\n")
            out.append("| true \\ pred | " + " | ".join(cm["labels"]) + " |")
            out.append("|---|" + "---|" * len(cm["labels"]))
            for lab, row in zip(cm["labels"], cm["rows_true_cols_pred"]):
                out.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
            out.append("")
    return "\n".join(out)


def run_all(rows: list[dict], out_json: Path, out_md: Path, seed: int = 0) -> dict:
    results = {
        "seed": seed,
        "splits": {
            "repo_split": run_split(rows, "split_repo", seed),
            "in_repo": run_split(rows, "split_inrepo", seed),
        },
    }
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(markdown(results) + "\n", encoding="utf-8")
    return results
