"""Write tests/fixtures/clean_cases.json: synthetic issues (no real issue text) and what the
Python cleaning returns for them. docs/js/clean.js has to produce the same strings
(tests/clean.test.mjs). Run: uv run python tests/make_clean_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tiny_triage.clean import build_text, clean_body, drop_lines

OUT = Path(__file__).parent / "fixtures" / "clean_cases.json"

BOILER = ["### Steps to reproduce", "**Environment**", "I tried this code:", "### Version"]

_LONG = "\n".join(f"line {i}" for i in range(45))
_EDGE = "\n".join(f"row {i}" for i in range(40))

CASES = [
    ("empty", "", ""),
    ("title only", "  Crash on start  ", ""),
    ("plain", "Add dark mode", "Please add a dark mode.\n\nIt hurts my eyes."),
    ("crlf and cr", "t", "a\r\nb\rc\r\n\r\n\r\n\r\n\r\nd"),
    ("html comment multiline", "t", "x <!-- hidden\nstuff --> y\n<!-- Describe -->\nkeep"),
    ("long fence", "t", f"crash:\n```python\n{_LONG}\n```\nthanks"),
    ("fence at the limit", "t", f"```\n{_EDGE}\n```\nafter"),
    ("fence just over the limit", "t", f"```\n{_EDGE}\nextra\n```\nafter"),
    ("short fence kept", "t", "try\n```\nfoo()\nbar()\n```\ndone"),
    ("unterminated long fence", "t", f"```\n{_LONG}\nmore"),
    ("tilde fence", "t", f"~~~sh\n{_LONG}\n~~~\ntail"),
    ("indented closing fence", "t", f"```\n{_LONG}\n   ```  \nz"),
    ("mismatched closing fence", "t", f"```\n{_LONG}\n~~~\nz\n```\nend"),
    ("image and url", "t", "![shot](http://x.io/a.png) see https://example.com/a?b=1, ok"),
    ("url ends at unicode space", "t", "go to http://a.b/c now and http://d.e　too"),
    ("url with nbsp-like and zero-width", "t", "http://a.b/c​d e"),
    ("mentions", "t", "cc @alice and @org/team-1, mail a@b.c, x@y, (@bob) /@carol"),
    ("mention after unicode letter", "t", "é@alice 5@bob _@dave 名@erin @-bad @ok-name"),
    ("mention with unicode team", "t", "@org/équipe and @org/名前 and @org/"),
    ("html tags", "t", "<details><summary>Logs</summary>x</details><br/><BR><p>y</p><img src=a>"),
    (
        "html tag word boundary",
        "t",
        "<pé>keep</pé> <pre>keep</pre> <span>a</span> <spanx>b</spanx>",
    ),
    ("checkboxes", "t", "- [x] a\n* [ ] b\n  -   [X] c\n-[ ] d\n- [] e\n- [y] f\nkeep"),
    ("no response", "t", "_No response_\n  no response  \nNo Response_\nnot no response\nkeep"),
    ("blank runs", "t", "a\n\n\n\n\nb\n\n\nc\n\nd"),
    ("trailing spaces", "t", "a   \nb\t\nc \nd　\n"),
    ("exotic whitespace", "t", "﻿a﻿\n\x1cb\x1c\n\x85c\x85\n d \n e  "),
    ("title exotic whitespace", "﻿\x1c Title \x85　", "body"),
    (
        "boilerplate drop",
        "t",
        "### Steps to reproduce\nrun it\n**Environment**\n  I tried this code: \nx",
    ),
    ("boilerplate padded", "t", "  ### Version  \n\t### Version\n###  Version\nv1"),
    ("boilerplate leaves blanks", "t", "a\n### Version\n\n### Version\n\n### Version\n\nb"),
    ("unicode text", "Ünïcode ✓ título", "日本語のテキスト\n\nемодзи 🎉🎉 done"),
    ("truncate ascii", "t", "x" * 2500),
    ("truncate astral", "t", "😀" * 1500),
    ("truncate mixed astral", "t", "a" + "😀" * 1999 + "b" * 50),
    ("truncate then strip", "t", "a" * 1998 + "  \n\n b"),
    ("truncate inside whitespace", "t", "a" * 1999 + " " + "b" * 10),
    ("code placeholder then drop", "t", f"```\n{_LONG}\n```\n### Version\nrest"),
    ("only boilerplate", "Title", "### Version\n### Steps to reproduce"),
    ("url in code block", "t", "```\nhttp://a.b\n```\n@user in code"),
]


def make() -> dict:
    boiler = set(BOILER)
    cases = []
    for name, title, body in CASES:
        b = clean_body(body)
        d = drop_lines(b, boiler)
        cases.append(
            {
                "name": name,
                "title": title,
                "body": body,
                "clean_body": b,
                "drop_lines": d,
                "build_text": build_text(title, d),
                "deploy_no_list": build_text(title, drop_lines(b, set())),
            }
        )
    return {"boilerplate": BOILER, "cases": cases}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(make(), indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
