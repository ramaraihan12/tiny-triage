from tiny_triage import finetune


def test_cached_output_goes_stale_when_an_input_changes(tmp_path):
    src = tmp_path / "input.bin"
    out = tmp_path / "out.json"
    src.write_bytes(b"one")
    out.write_text("{}")
    assert not finetune.fresh(out, finetune.sha(src))  # no key stored yet
    finetune.stamp(out, finetune.sha(src, "cfg"))
    assert finetune.fresh(out, finetune.sha(src, "cfg"))
    assert not finetune.fresh(out, finetune.sha(src, "other cfg"))
    src.write_bytes(b"two")
    assert not finetune.fresh(out, finetune.sha(src, "cfg"))
    out.unlink()
    assert not finetune.fresh(out, finetune.sha(src, "cfg"))
