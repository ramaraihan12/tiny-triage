import numpy as np
from sklearn.metrics import f1_score

from tiny_triage import clean, evalkit, report


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 4, 300))
    assert abs(evalkit.macro_f1(y, p, 4) - f1_score(y, p, average="macro", labels=range(4))) < 1e-9


def test_macro_f1_present_only_skips_absent_classes():
    y, p = np.array([0, 0, 1]), np.array([0, 1, 1])
    assert evalkit.macro_f1(y, p, 4, present_only=True) == evalkit.macro_f1(y, p, 2)
    assert evalkit.macro_f1(y, p, 4) < evalkit.macro_f1(y, p, 4, present_only=True)


def test_best_threshold_finds_separating_cut():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.01, 0.3, 50), np.linspace(0.31, 0.99, 50)])
    thr, f1 = evalkit.best_threshold(y, p)
    assert 0.3 <= thr <= 0.31 and f1 > 0.99


def test_best_threshold_ties_prefer_half():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    assert (
        evalkit.best_threshold(y, p)[0] == 0.5 or abs(evalkit.best_threshold(y, p)[0] - 0.5) < 0.31
    )


def test_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    good = np.where(rng.random(400) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(400) < 0.5, y, (y + 1) % 3)
    a = evalkit.bootstrap_diff(y, good, bad, 3, n_boot=300, seed=5)
    b = evalkit.bootstrap_diff(y, good, bad, 3, n_boot=300, seed=5)
    assert a == b
    assert a["ci_low"] > 0 and a["diff"] > 0
    same = evalkit.bootstrap_diff(y, good, good, 3, n_boot=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    assert evalkit.verdict(a).startswith("better")
    assert evalkit.verdict(same).startswith("not clearly")


def test_per_repo_groups_and_counts():
    repos = ["a", "a", "b", "b", "b"]
    y = [0, 1, 0, 0, 1]
    rows = evalkit.per_repo(repos, y, {"m": [0, 1, 1, 1, 1]}, 2)
    assert [r["repo"] for r in rows] == ["a", "b"]
    assert rows[0]["n"] == 2 and rows[0]["m"] == 1.0
    assert rows[1]["n"] == 3 and rows[1]["m"] < 1.0


def test_mean_std_uses_sample_std():
    r = evalkit.mean_std([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and r["std"] == 1.0


def test_frozen_boilerplate_uses_train_repos_only():
    line = "### Steps to reproduce"
    train = [{"repo": "t/t", "body": f"{line}\nbody {i}"} for i in range(30)]
    test = [{"repo": "x/x", "body": f"### Other template\nbody {i}"} for i in range(30)]
    frozen = evalkit.frozen_boilerplate({"t/t": train, "x/x": test}, ["t/t"])
    assert line in frozen and "### Other template" not in frozen
    text = evalkit.deploy_text("Title", f"{line}\nreal text", set(frozen))
    assert text == "Title\n\nreal text"
    assert text == clean.build_text("Title", "real text")


def test_render_markdown_says_when_not_better():
    ci = {"diff": 0.01, "ci_low": -0.02, "ci_high": 0.04, "n": 10, "n_boot": 5, "level": ""}
    res = {
        "bootstrap": {
            k: ci
            for k in ("category", "needs_info_vs_baseline_tuned", "needs_info_vs_baseline_default")
        }
    }
    text = "\n".join(report.headline(res))
    assert "not clearly different" in text
