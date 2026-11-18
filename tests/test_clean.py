from tiny_triage.clean import (
    MAX_BODY_CHARS,
    build_text,
    clean_body,
    clean_repo,
    drop_lines,
    template_lines,
)


def test_html_comments_removed_even_multiline():
    body = "before <!-- hidden\nstuff --> after\n<!-- Please describe -->\nreal text"
    out = clean_body(body)
    assert "hidden" not in out and "Please describe" not in out
    assert "real text" in out


def test_long_code_block_becomes_placeholder():
    code = "\n".join(f"line {i}" for i in range(60))
    out = clean_body(f"crash:\n```python\n{code}\n```\nthanks")
    assert "line 30" not in out
    assert "[long code block removed]" in out
    assert "thanks" in out


def test_short_code_block_is_kept():
    out = clean_body("try\n```\nfoo()\nbar()\n```\ndone")
    assert "foo()" in out and "bar()" in out


def test_unterminated_long_fence_is_cut():
    code = "\n".join(f"row {i}" for i in range(80))
    out = clean_body(f"```\n{code}")
    assert "row 70" not in out


def test_urls_mentions_and_checkboxes():
    body = "see https://example.com/x?y=1 cc @someuser\n- [x] I searched\n- [ ] I tested\nkeep"
    out = clean_body(body)
    assert "example.com" not in out and "someuser" not in out
    assert "<url>" in out and "@user" in out
    assert "I searched" not in out and "keep" in out


def test_none_body():
    assert clean_body(None) == ""


def test_truncation_applies_to_body_only():
    text = build_text("A title", "x" * 5000)
    title, body = text.split("\n\n", 1)
    assert title == "A title" and len(body) == MAX_BODY_CHARS


def test_template_lines_are_found_and_dropped():
    bodies = [f"### Steps to reproduce\nissue number {i}" for i in range(40)]
    boiler = template_lines(bodies)
    assert "### Steps to reproduce" in boiler
    assert drop_lines(bodies[0], boiler) == "issue number 0"


def test_clean_repo_keeps_title_and_removes_template():
    issues = [{"title": f"T{i}", "body": f"## Describe\nbug {i}"} for i in range(30)]
    out = clean_repo(issues)
    assert out[3].startswith("T3") and "## Describe" not in out[3] and "bug 3" in out[3]
