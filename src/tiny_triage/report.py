"""Render the M2 results dict as markdown. Pure Python, no torch."""

from __future__ import annotations

from .evalkit import verdict


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _ms(d: dict) -> str:
    return f"{d['mean']:.4f} ± {d['std']:.4f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return out + [""]


def headline(res: dict) -> list[str]:
    """Plain sentences about whether the fine-tuned model beats the baseline."""
    lines = []
    boot = res["bootstrap"]
    names = {
        "category": "category",
        "needs_info_vs_baseline_tuned": "needs_info (both with val-tuned threshold)",
        "needs_info_vs_baseline_default": "needs_info (baseline at 0.5)",
    }
    for key, label in names.items():
        ci = boot[key]
        lines.append(
            f"- {label}: macro-F1 difference model - baseline = {ci['diff']:+.4f}, "
            f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]: {verdict(ci)}."
        )
    return lines


def render_markdown(res: dict) -> str:
    out = ["# M2: fine-tuned encoder vs TF-IDF baseline (repo-disjoint split)", ""]
    if res.get("smoke"):
        out += ["SMOKE RUN: tiny subset, numbers are meaningless.", ""]
    out += ["## Candidates (seed 0, validation only)", ""]
    out += table(
        [
            "model",
            "lr",
            "best epoch",
            "val category F1",
            "val needs_info F1",
            "threshold",
            "val score",
        ],
        [
            [
                c["name"],
                c["lr"],
                c["best_epoch"],
                c["val_category_f1"],
                c["val_needs_info_f1"],
                c["threshold"],
                c["val_score"],
            ]
            for c in res["candidates"]
        ],
    )
    out += [
        f"Chosen on validation: **{res['chosen']}** ({res['base_model']}, {res['license']}). "
        "Val score = mean of category macro-F1 and needs_info macro-F1 (needs_info at its "
        "val-tuned threshold). Test was only evaluated for the chosen model's 3 seeds.",
        "",
        "## Test, split_repo",
        "",
    ]
    t = res["test"]
    b = t["baseline"]
    out += table(
        ["", "category macro-F1", "needs_info macro-F1 (0.5)", "needs_info macro-F1 (val-tuned)"],
        [
            [
                "TF-IDF + LR",
                b["category"],
                b["needs_info_default"],
                f"{b['needs_info_tuned']} (C {b['tuned_C']}, thr {b['threshold']})",
            ],
            [
                f"{res['chosen']}, mean ± std of {len(res['seeds'])} seeds",
                _ms(t["model_mean"]["category"]),
                _ms(t["model_mean"]["needs_info_default"]),
                _ms(t["model_mean"]["needs_info_tuned"]),
            ],
            [
                f"{res['chosen']}, exported seed {res['exported_seed']}",
                t["model_exported"]["category"],
                t["model_exported"]["needs_info_default"],
                f"{t['model_exported']['needs_info_tuned']} "
                f"(thr {t['model_exported']['threshold']})",
            ],
        ],
    )
    rep = b.get("reproduces_m1")
    out += [
        "Std is the sample standard deviation over seeds. The baseline row with the default "
        "threshold is the M1 baseline rerun; reproduces M1 numbers "
        f"(category 0.6309, needs_info 0.524): {rep}. The tuned baseline row picks C and the "
        "threshold together on val, like the model.",
        "",
    ]
    out += ["Per seed:", ""]
    out += table(
        [
            "seed",
            "val score",
            "threshold",
            "test category",
            "test needs_info (0.5)",
            "test needs_info (tuned)",
        ],
        [
            [
                s["seed"],
                s["val_score"],
                s["threshold"],
                s["test_category_f1"],
                s["test_needs_info_default_f1"],
                s["test_needs_info_tuned_f1"],
            ]
            for s in res["seeds"]
        ],
    )
    out += [
        "## Paired bootstrap (exported seed minus baseline)",
        "",
        f"Issue-level bootstrap, {res['bootstrap']['category']['n_boot']} resamples, fixed seed. "
        "Only issues are resampled, not repos: the test set has few repos, so repo-to-repo "
        "variance is not captured and these intervals are too narrow for a new repo.",
        "",
    ]
    out += headline(res)
    out += ["", "## Per test repo (macro-F1 over classes present in that repo)", ""]
    out += table(
        [
            "repo",
            "n",
            "category base",
            "category model",
            "n labeled",
            "needs_info base (0.5)",
            "needs_info base (tuned)",
            "needs_info model (tuned)",
        ],
        [
            [
                r["repo"],
                r["n"],
                r["category_baseline"],
                r["category_model"],
                r["n_needs_info"],
                r["needs_info_baseline_default"],
                r["needs_info_baseline_tuned"],
                r["needs_info_model"],
            ]
            for r in res["per_repo"]
        ],
    )
    d = res["deploy_time"]
    out += [
        "## Deploy-time cleaning",
        "",
        f"Template lines frozen from the {d['n_train_repos']} train repos only "
        f"({d['n_frozen_lines']} lines), instead of statistics over each repo's own issues. "
        "Rows marked 'at inference only' keep the models fitted on M1 text and only rebuild "
        "the val/test text. The encoder was trained on M1 text, so for it this is a "
        "train/inference cleaning mismatch; the baseline refit row retrains TF-IDF + LR on "
        "train text rebuilt with the frozen list as well.",
        "",
    ]
    out += table(
        ["", "text", "category", "needs_info (0.5)", "needs_info (tuned)"],
        [
            [name, text, v["category"], v["needs_info_default"], v["needs_info_tuned"]]
            for name, text, v in d["rows"]
        ],
    )
    f = res["export"]
    q = f["quant"]
    fid = f["fidelity"]
    out += ["## Export, quantization, fidelity, latency", ""]
    out += [
        f"int8 chosen on validation only: **{q['chosen']}** ({q['settings'] or 'defaults'}). "
        f"Rule: {q['rule']}. Per-channel signed int8 weights with the full range collapsed the "
        "category head to one class on this model (val agreement 0.14); reduce_range or "
        "unsigned weights avoid it. The configs below were compared on val.",
        "",
    ]
    out += table(
        [
            "config",
            "size MB",
            "val category agreement",
            "val needs_info argmax",
            "val needs_info at thr",
        ],
        [
            [
                r["config"],
                r.get("size_mb", r.get("error")),
                r.get("category"),
                r.get("needs_info_argmax"),
                r.get("needs_info_threshold"),
            ]
            for r in q["configs"]
        ],
    )
    g = fid["gate"]
    out += [
        "Gate (val category agreement >= 0.98, test |macro-F1 delta| <= 0.01 per target, "
        f"size <= 50 MB): {'PASSED' if g['passed'] else 'NOT PASSED'}: {g}",
        "",
    ]
    out += table(
        ["file", "size MB"],
        [
            ["model.onnx (fp32)", fid["sizes"]["fp32_mb"]],
            ["model.int8.onnx", fid["sizes"]["int8_mb"]],
        ],
    )
    out += table(
        [
            "test comparison",
            "category argmax agreement",
            "needs_info argmax agreement",
            "needs_info at threshold",
        ],
        [
            [k, v["category"], v["needs_info_argmax"], v["needs_info_threshold"]]
            for k, v in fid["agreement"].items()
        ],
    )
    out += table(
        ["macro-F1 on full test", "PyTorch fp32", "ONNX int8", "delta"],
        [[k, v["pytorch"], v["int8"], v["delta"]] for k, v in fid["macro_f1"].items()],
    )
    out += table(
        ["CPU latency (batch 1, 1 thread, tokenizer + onnxruntime)", "n", "p50 ms", "p95 ms"],
        [[k, v["n"], v["p50_ms"], v["p95_ms"]] for k, v in fid["latency"].items()],
    )
    out += ["## Notes and caveats", ""]
    out += [f"- {n}" for n in res["notes"]]
    out += [""]
    return "\n".join(out)
