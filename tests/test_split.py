import random
from collections import Counter

from tiny_triage.labels import CATEGORIES
from tiny_triage.split import MIN_CLASS, assign_inrepo_splits, assign_repo_splits


def fake_stats(n_repos=14, seed=1):
    rng = random.Random(seed)
    stats = {}
    for i in range(n_repos):
        cat = Counter({c: rng.randint(MIN_CLASS, 300) for c in CATEGORIES})
        ni = Counter({"yes": rng.randint(60, 120), "no": rng.randint(100, 200)})
        stats[f"o/r{i}"] = {"n": sum(cat.values()), "cat": cat, "ni": ni}
    return stats


def test_repo_splits_are_disjoint_and_complete():
    stats = fake_stats()
    split = assign_repo_splits(stats, seed=3, n_val=3, n_test=3, tries=500)
    assert set(split) == set(stats)
    parts = {s: {r for r, v in split.items() if v == s} for s in ("train", "val", "test")}
    assert len(parts["val"]) == 3 and len(parts["test"]) == 3
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]


def test_every_class_appears_in_each_split():
    stats = fake_stats()
    split = assign_repo_splits(stats, seed=3, n_val=3, n_test=3, tries=500)
    for name in ("train", "val", "test"):
        cat = Counter()
        for r, s in split.items():
            if s == name:
                cat.update(stats[r]["cat"])
        assert all(cat[c] >= MIN_CLASS for c in CATEGORIES)


def test_split_is_deterministic():
    stats = fake_stats()
    a = assign_repo_splits(stats, seed=7, n_val=3, n_test=3, tries=300)
    b = assign_repo_splits(stats, seed=7, n_val=3, n_test=3, tries=300)
    assert a == b


def test_inrepo_split_covers_all_keys_with_expected_shares():
    keys = [("o/r", i) for i in range(5000)]
    split = assign_inrepo_splits(keys, seed=0)
    assert set(split) == set(keys)
    share = Counter(split.values())
    assert 0.75 < share["train"] / len(keys) < 0.85
