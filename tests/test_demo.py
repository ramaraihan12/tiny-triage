from tiny_triage import llama
from tiny_triage.demo import filter_boilerplate, keep_line


def test_form_headings_and_labels_are_kept():
    for line in (
        "### Steps to reproduce",
        "### Node.js Version",
        "**Describe the bug**",
        "**What did you do?**",
        "* Widget version:",
        "3. What workarounds are you using in the meantime?",
        "Paste the output of 'python -m sysconfig', over this line.",
    ):
        assert keep_line(line), line


def test_environment_dumps_paths_and_user_content_are_dropped():
    for line in (
        "C:\> mytool --info",
        "Build:               0123456789abcdef0123456789abcdef01234567",
        "### See https://example.com",
        "### Contact @someone",
        "### Log /etc/os-release",
        "### Build 20250101",
        "libfoo: OpenFoo 9.9.9 01 Jan 2030",
        "examplepkg           <not installed>",
        "```python",
        "<url>",
        "Every time",
        "",
    ):
        assert not keep_line(line), line


def test_filter_is_sorted_and_keeps_a_subset():
    lines = ["### B", "pass", "### A", "**Version**"]
    out = filter_boilerplate(lines)
    assert out == sorted(out) and set(out) <= set(lines)
    assert "pass" not in out


def _rows():
    rows = []
    for r in range(3):
        for c in ("bug", "feature", "question", "docs"):
            for _ in range(10):
                rows.append(
                    {"repo": f"o/r{r}", "number": len(rows), "category": c, "needs_info": None}
                )
    return rows


def test_sample_is_seeded_stratified_and_the_right_size():
    rows = _rows()
    a = llama.sample(rows, n=26)
    b = llama.sample(rows, n=26)
    assert a == b and len(a) == 26
    strata = {(r["repo"], r["category"]) for r in a}
    assert len(strata) == 12  # every (repo, category) is in there
    assert len({(r["repo"], r["number"]) for r in a}) == 26
    assert a != llama.sample(rows, seed=1, n=26)


def test_parse_accepts_the_json_object_and_flags_anything_else():
    assert llama.parse('{"category": "Bug", "confidence": 0.9, "needs_info": true}') == {
        "category": "bug",
        "needs_info": "yes",
    }
    assert llama.parse({"category": "docs", "needs_info": "no"}) == {
        "category": "docs",
        "needs_info": "no",
    }
    assert llama.parse("not json") == {"category": "unparsed", "needs_info": "unparsed"}
    assert llama.parse('{"category": "other", "needs_info": 3}') == {
        "category": "unparsed",
        "needs_info": "unparsed",
    }
    assert llama.parse("[1]") == {"category": "unparsed", "needs_info": "unparsed"}


def test_neuron_estimate_uses_both_token_counts():
    n = llama.neurons({"prompt_tokens": 1000, "completion_tokens": 100})
    assert 40 < n < 50
