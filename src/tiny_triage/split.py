"""Repo-disjoint split (the honest one) and a random in-repo split (the optimistic one)."""

from __future__ import annotations

import random
from collections import Counter

from .labels import CATEGORIES

MIN_CLASS = 120  # minimum issues per category in val and test
MIN_NI = 80  # minimum yes and no for needs_info in val and test


def repo_stats(rows: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for r in rows:
        s = stats.setdefault(r["repo"], {"n": 0, "cat": Counter(), "ni": Counter()})
        s["n"] += 1
        s["cat"][r["category"]] += 1
        if r["needs_info"] is not None:
            s["ni"][r["needs_info"]] += 1
    return stats


def _totals(stats, repos):
    cat, ni = Counter(), Counter()
    for r in repos:
        cat.update(stats[r]["cat"])
        ni.update(stats[r]["ni"])
    return cat, ni


def _ok(stats, repos) -> bool:
    cat, ni = _totals(stats, repos)
    return (
        all(cat[c] >= MIN_CLASS for c in CATEGORIES) and ni["yes"] >= MIN_NI and ni["no"] >= MIN_NI
    )


def assign_repo_splits(
    stats: dict[str, dict],
    seed: int = 0,
    n_val: int = 5,
    n_test: int = 6,
    tries: int = 20000,
) -> dict[str, str]:
    """Pick disjoint train/val/test repo sets where every class appears in each.

    Random search with a fixed seed: among assignments that meet the minimum class counts,
    take the one whose val and test are closest to 15% of the issues each and whose category
    mix is closest to the overall mix.
    """
    repos = sorted(stats)
    total = sum(stats[r]["n"] for r in repos)
    all_cat, _ = _totals(stats, repos)
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(tries):
        order = repos[:]
        rng.shuffle(order)
        val, test, train = order[:n_val], order[n_val : n_val + n_test], order[n_val + n_test :]
        if not (_ok(stats, val) and _ok(stats, test) and _ok(stats, train)):
            continue
        score = 0.0
        for part in (val, test):
            size = sum(stats[r]["n"] for r in part)
            score += 4 * abs(size / total - 0.15)
            cat, _ = _totals(stats, part)
            score += sum(abs(cat[c] / size - all_cat[c] / total) for c in CATEGORIES)
        if score < best_score:
            best, best_score = (train, val, test), score
    if best is None:
        raise RuntimeError("no repo split meets the minimum class counts")
    out = {}
    for name, part in zip(("train", "val", "test"), best):
        for r in part:
            out[r] = name
    return out


def assign_inrepo_splits(keys: list, seed: int = 0, val: float = 0.1, test: float = 0.1) -> dict:
    """Random split over individual issues, so every repo shows up in train and test."""
    rng = random.Random(seed)
    out = {}
    for k in keys:
        x = rng.random()
        out[k] = "test" if x < test else "val" if x < test + val else "train"
    return out
