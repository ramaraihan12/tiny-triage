"""Torch-free evaluation helpers: thresholds, bootstrap, per-repo metrics, frozen boilerplate."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .clean import build_text, clean_body, drop_lines, template_lines

THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def confusion(y_true, y_pred, k: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def _f1_from_conf(conf: np.ndarray, present_only: bool = False) -> float:
    tp = np.diag(conf).astype(float)
    support, predicted = conf.sum(1), conf.sum(0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    if present_only:
        keep = denom > 0
        return float(f1[keep].mean()) if keep.any() else 0.0
    return float(f1.mean())


def macro_f1(y_true, y_pred, k: int, present_only: bool = False) -> float:
    """Macro-F1 over class indices 0..k-1. A class with no true and no predicted item scores 0,
    unless present_only is set (used for tiny per-repo slices), which skips such classes."""
    if len(y_true) == 0:
        return float("nan")
    return _f1_from_conf(confusion(y_true, y_pred, k), present_only)


def best_threshold(y_true, p_yes, grid=THRESHOLD_GRID) -> tuple[float, float]:
    """Threshold on P(yes) with the best macro-F1; ties go to the one closest to 0.5."""
    y_true, p_yes = np.asarray(y_true), np.asarray(p_yes)
    scores = [macro_f1(y_true, (p_yes >= t).astype(int), 2) for t in grid]
    best = max(scores)
    ties = [t for t, s in zip(grid, scores) if s >= best - 1e-12]
    thr = min(ties, key=lambda t: abs(t - 0.5))
    return float(thr), float(best)


def mean_std(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
        "values": [round(float(x), 4) for x in v],
    }


def bootstrap_diff(
    y_true, pred_a, pred_b, k: int, n_boot: int = 10_000, seed: int = 0, chunk: int = 500
) -> dict:
    """Paired issue-level bootstrap of macro-F1(a) - macro-F1(b), percentile 95% CI."""
    y_true, pred_a, pred_b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    n = len(y_true)
    onehot_a = np.zeros((n, k * k), dtype=np.float32)
    onehot_b = np.zeros((n, k * k), dtype=np.float32)
    onehot_a[np.arange(n), y_true * k + pred_a] = 1
    onehot_b[np.arange(n), y_true * k + pred_b] = 1
    rng = np.random.default_rng(seed)
    diffs = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float32)
        ca, cb = w @ onehot_a, w @ onehot_b
        for a, b in zip(ca, cb):
            diffs.append(
                _f1_from_conf(a.reshape(k, k).round()) - _f1_from_conf(b.reshape(k, k).round())
            )
        done += m
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = macro_f1(y_true, pred_a, k) - macro_f1(y_true, pred_b, k)
    return {
        "diff": round(float(point), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n": int(n),
        "n_boot": n_boot,
        "level": "issue (repos are not resampled; the test set has few repos)",
    }


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "better (CI above 0)"
    if ci["ci_high"] < 0:
        return "worse (CI below 0)"
    return "not clearly different (CI includes 0)"


def per_repo(repos, y_true, preds: dict[str, np.ndarray], k: int) -> list[dict]:
    """Per-repo n and macro-F1 (over classes present in that repo) for each named prediction."""
    repos, y_true = np.asarray(repos), np.asarray(y_true)
    out = []
    for repo in sorted(set(repos.tolist())):
        sel = repos == repo
        row = {"repo": repo, "n": int(sel.sum())}
        for name, p in preds.items():
            row[name] = round(macro_f1(y_true[sel], np.asarray(p)[sel], k, present_only=True), 4)
        out.append(row)
    return out


def frozen_boilerplate(raw_by_repo: dict[str, list[dict]], train_repos) -> list[str]:
    """Union of per-repo template lines over the train repos only."""
    frozen: set[str] = set()
    for repo in sorted(train_repos):
        bodies = [clean_body(i.get("body")) for i in raw_by_repo.get(repo, [])]
        frozen |= template_lines(bodies)
    return sorted(frozen)


def deploy_text(title: str, body: str | None, frozen: set[str]) -> str:
    """Text as a deployed demo would see it: no statistics over the issue's own repo."""
    return build_text(title, drop_lines(clean_body(body), frozen))


def index_raw(raw_rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r["repo"], r["number"]): r for r in raw_rows}


def group_by_repo(raw_rows: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in raw_rows:
        by[r["repo"]].append(r)
    return dict(by)
