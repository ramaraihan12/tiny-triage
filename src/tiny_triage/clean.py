"""Turn a raw issue title and body into short plain text for the classifier."""

from __future__ import annotations

import re
from collections import Counter

MAX_BODY_CHARS = 2000
MAX_CODE_LINES = 40

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FENCE = re.compile(r"(```|~~~)[^\n]*\n(.*?)(?:\n[ \t]*\1[ \t]*(?=\n|$)|\Z)", re.S)
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"(?<![\w/])@[A-Za-z0-9][A-Za-z0-9-]*(?:/[\w-]+)?")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?(?:details|summary|img|br|p|div|span|kbd|sub|sup)\b[^>]*>", re.I)
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]")
_BLANKS = re.compile(r"\n{3,}")
_NO_RESPONSE = re.compile(r"^\s*_?no response_?\s*$", re.I)


def _shrink_code(m: re.Match) -> str:
    lines = m.group(2).count("\n") + 1
    if lines > MAX_CODE_LINES:
        return "\n[long code block removed]\n"
    return m.group(0)


def clean_body(body: str | None) -> str:
    """Strip comments, long code, urls, mentions and checklist lines. Does not truncate."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT.sub("", text)
    text = _FENCE.sub(_shrink_code, text)
    text = _IMAGE.sub("[image]", text)
    text = _URL.sub("<url>", text)
    text = _MENTION.sub("@user", text)
    text = _HTML_TAG.sub("", text)
    kept = [ln.rstrip() for ln in text.split("\n")]
    kept = [ln for ln in kept if not _CHECKBOX.match(ln) and not _NO_RESPONSE.match(ln)]
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip()


def template_lines(bodies: list[str], min_count: int = 20, min_share: float = 0.05) -> set[str]:
    """Lines that repeat across many issues of one repo are template boilerplate."""
    counts: Counter[str] = Counter()
    for body in bodies:
        counts.update({ln.strip() for ln in body.split("\n") if len(ln.strip()) > 3})
    floor = max(min_count, int(min_share * len(bodies)))
    return {ln for ln, n in counts.items() if n >= floor}


def drop_lines(body: str, boiler: set[str]) -> str:
    kept = [ln for ln in body.split("\n") if ln.strip() not in boiler]
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip()


def build_text(title: str, body: str) -> str:
    """Final classifier input: title, blank line, body cut to MAX_BODY_CHARS."""
    return f"{title.strip()}\n\n{body[:MAX_BODY_CHARS].strip()}".strip()


def clean_repo(issues: list[dict]) -> list[str]:
    """Clean every issue of one repo, removing that repo's template lines."""
    bodies = [clean_body(i.get("body")) for i in issues]
    boiler = template_lines(bodies)
    return [build_text(i["title"], drop_lines(b, boiler)) for i, b in zip(issues, bodies)]
