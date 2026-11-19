import numpy as np

from tiny_triage import baseline


def _rows():
    rng = np.random.default_rng(0)
    words = {
        "bug": "crash error exception traceback",
        "feature": "support add option proposal",
        "question": "how why help usage",
        "docs": "typo documentation readme guide",
    }
    rows = []
    for i in range(240):
        cat = list(words)[i % 4]
        split = "train" if i % 8 < 5 else ("val" if i % 8 == 5 else "test")
        text = " ".join(rng.choice(words[cat].split(), 6)) + f" item{i % 7}"
        rows.append(
            {
                "repo": "r/r",
                "number": i,
                "category": cat,
                "needs_info": ("yes" if i % 3 == 0 else "no") if i % 5 else None,
                "text": text,
                "split_repo": split,
            }
        )
    return rows


def test_predict_split_matches_run_split_choices():
    rows = _rows()
    res = baseline.predict_split(
        rows, "split_repo", 0, {"val": ["crash"] * 5, "test": ["typo"] * 3}
    )
    n_val = sum(r["split_repo"] == "val" for r in rows)
    n_test = sum(r["split_repo"] == "test" for r in rows)
    assert res["category"]["val"].shape == (n_val, 4)
    assert res["needs_info"]["test"].shape == (n_test, 2)
    assert res["category"]["alt_test"].shape == (3, 4)
    assert np.allclose(res["category"]["test"].sum(1), 1)
    ref = baseline.run_split(rows, "split_repo", 0)
    test = [r for r in rows if r["split_repo"] == "test"]
    pred = res["category"]["test"].argmax(1)
    from tiny_triage.labels import CATEGORIES

    got = np.mean([CATEGORIES[p] == r["category"] for p, r in zip(pred, test)])
    assert round(float(got), 4) == ref["category"]["test_accuracy"]
