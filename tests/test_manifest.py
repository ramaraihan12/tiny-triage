import csv
from pathlib import Path

from tiny_triage.dataset import MANIFEST_COLUMNS, build, manifest_rows, write_manifest

ROOT = Path(__file__).parents[1]
FORBIDDEN = {"text", "title", "body", "user", "author", "login", "labels"}


def test_manifest_columns_have_no_text_or_usernames():
    assert not FORBIDDEN & set(MANIFEST_COLUMNS)
    assert MANIFEST_COLUMNS == [
        "repo", "number", "category", "needs_info", "split_repo", "split_inrepo",
    ]  # fmt: skip


def test_written_manifest_never_contains_issue_text(tmp_path):
    row = {
        "repo": "o/r", "number": 5, "category": "bug", "needs_info": "no",
        "split_repo": "train", "split_inrepo": "val",
        "text": "SECRET issue text by someone", "title": "t", "body": "b",
    }  # fmt: skip
    path = tmp_path / "m.csv"
    write_manifest([row], path)
    content = path.read_text(encoding="utf-8")
    assert "SECRET" not in content
    assert list(manifest_rows([row])[0]) == MANIFEST_COLUMNS


def test_committed_manifest_is_text_free():
    path = ROOT / "data" / "manifest.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == MANIFEST_COLUMNS
        for row in reader:
            assert row["repo"].count("/") == 1
            assert row["number"].isdigit()
            assert row["category"] in {"bug", "feature", "question", "docs"}
            assert row["needs_info"] in {"", "yes", "no"}
            assert row["split_repo"] in {"train", "val", "test"}
            assert None not in row


def test_build_assigns_one_repo_split_per_repo():
    rows = []
    for r in range(14):
        for c_i, c in enumerate(("bug", "feature", "question", "docs")):
            for n in range(130):
                rows.append(
                    {
                        "repo": f"o/r{r}", "number": c_i * 1000 + n, "title": "t",
                        "body": f"body {n}", "category": c,
                        "needs_info": "yes" if n % 3 == 0 else "no",
                    }
                )  # fmt: skip
    out = build(rows, seed=1)
    by_repo: dict[str, set[str]] = {}
    for r in out:
        by_repo.setdefault(r["repo"], set()).add(r["split_repo"])
    assert all(len(v) == 1 for v in by_repo.values())
