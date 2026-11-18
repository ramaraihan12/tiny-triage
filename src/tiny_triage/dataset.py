"""Build the local dataset from raw files and write the public manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .clean import clean_repo
from .split import assign_inrepo_splits, assign_repo_splits, repo_stats

# The manifest is the only data file that is committed. These columns carry no text
# and no usernames: just where an issue lives, how it was labelled, and which split it is in.
MANIFEST_COLUMNS = ["repo", "number", "category", "needs_info", "split_repo", "split_inrepo"]


def load_raw(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def build(rows: list[dict], seed: int = 0) -> list[dict]:
    """Clean text and attach both splits. Returns rows in (repo, number) order."""
    rows = sorted(rows, key=lambda r: (r["repo"], r["number"]))
    by_repo: dict[str, list[dict]] = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)
    out = []
    for repo, items in by_repo.items():
        for issue, text in zip(items, clean_repo(items)):
            out.append(
                {
                    "repo": repo,
                    "number": issue["number"],
                    "category": issue["category"],
                    "needs_info": issue["needs_info"],
                    "text": text,
                }
            )
    repo_split = assign_repo_splits(repo_stats(out), seed=seed)
    in_split = assign_inrepo_splits([(r["repo"], r["number"]) for r in out], seed=seed)
    for r in out:
        r["split_repo"] = repo_split[r["repo"]]
        r["split_inrepo"] = in_split[(r["repo"], r["number"])]
    return out


def manifest_rows(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in MANIFEST_COLUMNS} for r in rows]


def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in manifest_rows(rows):
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_dataset(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
