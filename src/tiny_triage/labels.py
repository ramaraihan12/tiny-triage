"""Map a repo's own labels onto the tiny-triage scheme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("bug", "feature", "question", "docs")
NEEDS_INFO = ("no", "yes")


@dataclass(frozen=True)
class Mapped:
    """Result of mapping one issue's labels.

    status is "ok", "none" (no category label) or "ambiguous" (labels from two classes).
    needs_info is None when the repo has no needs_info labels, so "no" is never guessed
    for a repo that could not have said "yes".
    """

    status: str
    category: str | None
    needs_info: str | None


def load_map(path: str | Path) -> dict[str, dict[str, list[str]]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    repos = data["repos"]
    allowed = set(CATEGORIES) | {"needs_info"}
    for repo, classes in repos.items():
        unknown = set(classes) - allowed
        if unknown:
            raise ValueError(f"{repo}: unknown classes {sorted(unknown)}")
    return repos


def _fold(names) -> set[str]:
    return {n.casefold() for n in names}


def map_labels(repo_map: dict[str, list[str]], labels: list[str]) -> Mapped:
    have = _fold(labels)
    hits = [c for c in CATEGORIES if have & _fold(repo_map.get(c, []))]
    needs = repo_map.get("needs_info")
    needs_info = None
    if needs:
        needs_info = "yes" if have & _fold(needs) else "no"
    if not hits:
        return Mapped("none", None, needs_info)
    if len(hits) > 1:
        return Mapped("ambiguous", None, needs_info)
    return Mapped("ok", hits[0], needs_info)
