"""Fetch labelled issues with the authenticated gh CLI and save them outside the repo.

The raw files hold other people's writing, so they go to a local folder you choose
and are never committed. Only the manifest (repo, number, split, labels) is public.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .labels import CATEGORIES, map_labels

PER_PAGE = 100
MAX_PAGES = 5
CAP_PER_CLASS = 400  # per repo, per category
CAP_NEEDS_INFO = 150  # per repo, how many needs_info=yes issues are kept first
MAX_REQUESTS = 1900


class Budget:
    def __init__(self, limit: int = MAX_REQUESTS):
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError(f"request budget of {self.limit} used up")
        self.used += 1


def fetch_page(repo: str, label: str, page: int) -> list[dict]:
    cmd = [
        "gh", "api", f"repos/{repo}/issues", "-X", "GET",
        "-f", f"labels={label}", "-f", "state=all", "-f", f"per_page={PER_PAGE}",
        "-f", f"page={page}", "-f", "sort=created", "-f", "direction=desc",
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(res.stdout)


def slim(item: dict) -> dict:
    """Keep only what the pipeline needs. The author is dropped on purpose."""
    return {
        "number": item["number"],
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "labels": [lb["name"] for lb in item.get("labels", [])],
        "created_at": item.get("created_at"),
    }


def collect_repo(
    repo: str,
    repo_map: dict[str, list[str]],
    budget: Budget,
    wanted: set[int] | None = None,
    fetch=fetch_page,
) -> dict[int, dict]:
    """Return {number: issue} for one repo.

    Normal mode asks for each mapped label separately (GitHub ANDs several labels, so
    one label per query), newest first, up to MAX_PAGES pages. Rebuild mode (wanted is
    given) pages until it is past the oldest wanted issue and keeps only those.
    """
    labels = [lb for k in (*CATEGORIES, "needs_info") for lb in repo_map.get(k, [])]
    found: dict[int, dict] = {}
    for label in labels:
        kept = 0
        for page in range(1, MAX_PAGES + 1):
            budget.spend()
            items = fetch(repo, label, page)
            for it in items:
                if "pull_request" in it:
                    continue
                if wanted is not None and it["number"] not in wanted:
                    continue
                if it["number"] not in found:
                    found[it["number"]] = slim(it)
                kept += 1
            if len(items) < PER_PAGE:
                break
            if wanted is not None:
                if min(i["number"] for i in items) < min(wanted):
                    break
            elif kept >= CAP_PER_CLASS:
                break
    return found


def select(repo_map: dict[str, list[str]], found: dict[int, dict]) -> list[dict]:
    """Map labels, drop ambiguous and unlabelled issues, then cap per category."""
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for num in sorted(found, reverse=True):
        issue = found[num]
        m = map_labels(repo_map, issue["labels"])
        if m.status != "ok":
            continue
        by_cat[m.category].append({**issue, "category": m.category, "needs_info": m.needs_info})
    chosen = []
    for rows in by_cat.values():
        yes = [r for r in rows if r["needs_info"] == "yes"][:CAP_NEEDS_INFO]
        rest = [r for r in rows if r["needs_info"] != "yes"]
        chosen.extend((yes + rest)[:CAP_PER_CLASS])
    return sorted(chosen, key=lambda r: r["number"])


def collect_all(
    label_map: dict,
    raw_dir: Path,
    manifest_rows: list[dict] | None = None,
    only: list[str] | None = None,
) -> int:
    """Collect every repo into raw_dir/<owner>__<name>.jsonl. Returns requests used."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, set[int]] = {}
    for r in manifest_rows or []:
        wanted.setdefault(r["repo"], set()).add(int(r["number"]))
    budget = Budget()
    for repo, repo_map in label_map.items():
        if only and repo not in only:
            continue
        out = raw_dir / (repo.replace("/", "__") + ".jsonl")
        if out.exists():
            continue
        found = collect_repo(repo, repo_map, budget, wanted.get(repo) if manifest_rows else None)
        if manifest_rows:
            rows = [
                {**found[n], **dict(zip(("category", "needs_info"), _cn(repo_map, found[n])))}
                for n in sorted(found)
            ]
            rows = [r for r in rows if r["category"] is not None]
        else:
            rows = select(repo_map, found)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"repo": repo, **row}, ensure_ascii=False) + "\n")
        print(f"{repo}: {len(rows)} issues, {budget.used} requests so far", flush=True)
    return budget.used


def _cn(repo_map, issue):
    m = map_labels(repo_map, issue["labels"])
    return m.category, m.needs_info
