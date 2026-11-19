"""Resumable M2 driver: fine-tune candidates, pick on val, compare with the baseline, export.

python -m tiny_triage.finetune all --data DATA --runs RUNS --artifacts ART [--smoke]

Stages run in order and are skipped when their outputs exist and their inputs are unchanged
(a hash of the inputs is stored next to each cached output): frozen, candidates, select, winner,
baseline, export, quant, fidelity, deploy, report. Needs the training venv (torch, transformers,
onnx).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import baseline, dataset, evalkit, report
from .labels import CATEGORIES, NEEDS_INFO

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
STAGES = (
    "frozen",
    "candidates",
    "select",
    "winner",
    "baseline",
    "export",
    "quant",
    "fidelity",
    "deploy",
    "report",
)
INT8_MAX_MB = 50.0
SMOKE_SIZES = {"train": 400, "val": 200, "test": 200}


class Ctx:
    def __init__(self, args):
        self.args = args
        self.smoke = args.smoke
        self.data = args.data
        self.work = args.runs / "smoke" if args.smoke else args.runs
        self.art = args.artifacts / "smoke" if args.smoke else args.artifacts
        self.results = self.work / "results" if args.smoke else (args.results or ROOT / "results")
        for d in (self.work, self.art, self.results):
            d.mkdir(parents=True, exist_ok=True)
        self.max_epochs = 1 if args.smoke else args.max_epochs
        self.max_steps = 8 if args.smoke else None
        self.logfile = open(self.work / "finetune.log", "a", encoding="utf-8")
        full = dataset.read_dataset(args.data / "dataset.jsonl")
        self.full_rows = full
        self.train_repos = sorted({r["repo"] for r in full if r["split_repo"] == "train"})
        self.rows = self._subsample(full) if self.smoke else full
        self._raw = None

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logfile.write(line + "\n")
        self.logfile.flush()

    def _subsample(self, rows):
        rng = np.random.default_rng(0)
        keep = set()
        for split, n in SMOKE_SIZES.items():
            idx = [i for i, r in enumerate(rows) if r["split_repo"] == split]
            keep.update(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
        return [r for i, r in enumerate(rows) if i in keep]

    def raw(self) -> list[dict]:
        if self._raw is None:
            self._raw = dataset.load_raw(self.data / "raw")
        return self._raw

    def split_rows(self, split: str) -> list[dict]:
        return [r for r in self.rows if r["split_repo"] == split]

    def arrays(self, split: str):
        rows = self.split_rows(split)
        y_cat = np.array([CATEGORIES.index(r["category"]) for r in rows])
        y_ni = np.array(
            [-1 if r["needs_info"] is None else NEEDS_INFO.index(r["needs_info"]) for r in rows]
        )
        return [r["text"] for r in rows], y_cat, y_ni

    def frozen_set(self) -> set[str]:
        return set(json.loads((self.art / "boilerplate.json").read_text(encoding="utf-8"))["lines"])

    def deploy_rows(self, rows: list[dict]) -> list[str]:
        """Text for these rows rebuilt with the frozen list instead of per-repo statistics."""
        frozen = self.frozen_set()
        raw = evalkit.index_raw(self.raw())
        out = []
        for r in rows:
            i = raw[(r["repo"], r["number"])]
            out.append(evalkit.deploy_text(i["title"], i.get("body"), frozen))
        return out

    def deploy_texts(self, split: str) -> list[str]:
        return self.deploy_rows(self.split_rows(split))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_dir(ctx: Ctx, name: str, seed: int) -> Path:
    return ctx.work / name / f"seed{seed}"


# ------------------------------------------------------------------ cache keys


def sha(*parts) -> str:
    """Short hash of files (by content) and strings. A cached output is reused only if the
    hash of its inputs, stored next to it, still matches."""
    h = hashlib.sha1()
    for part in parts:
        if isinstance(part, Path):
            with part.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _key_file(out: Path) -> Path:
    return out.with_name(out.name + ".key")


def fresh(out: Path, key: str) -> bool:
    kf = _key_file(out)
    return out.exists() and kf.exists() and kf.read_text(encoding="utf-8").strip() == key


def stamp(out: Path, key: str) -> None:
    _key_file(out).write_text(key + "\n", encoding="utf-8")


# ------------------------------------------------------------------ stages


def stage_frozen(ctx: Ctx) -> None:
    out = ctx.art / "boilerplate.json"
    key = sha(
        ctx.data / "dataset.jsonl",
        *sorted((ctx.data / "raw").glob("*.jsonl")),
        "frozen-v1",
    )
    if fresh(out, key):
        return
    by_repo = evalkit.group_by_repo(ctx.raw())
    lines = evalkit.frozen_boilerplate(by_repo, ctx.train_repos)
    write(
        out,
        {
            "note": "template lines frequent within a single train repo; union over train repos. "
            "Contains some non-template lines: filter before shipping.",
            "n_train_repos": len(ctx.train_repos),
            "lines": lines,
        },
    )
    stamp(out, key)
    ctx.log(f"frozen boilerplate: {len(lines)} lines from {len(ctx.train_repos)} train repos")


def train_data(ctx: Ctx) -> dict:
    return {s: ctx.arrays(s) for s in ("train", "val")}


def stage_candidates(ctx: Ctx) -> None:
    from . import encoder

    data = train_data(ctx)
    for name in encoder.CANDIDATES:
        d = run_dir(ctx, name, 0)
        if (d / "done.json").exists():
            ctx.log(f"skip {name} seed0 (done)")
            continue
        ctx.log(f"train {name} seed0")
        s = encoder.train_run(name, 0, data, d, ctx.log, ctx.max_epochs, ctx.max_steps)
        ctx.log(f"done {name} seed0 val_score {s['val_score']} in {s['train_seconds']}s")


def candidate_summaries(ctx: Ctx) -> list[dict]:
    from . import encoder

    return [read(run_dir(ctx, n, 0) / "done.json") for n in encoder.CANDIDATES]


def stage_select(ctx: Ctx) -> None:
    sel = ctx.work / "select.json"
    cands = candidate_summaries(ctx)
    if not sel.exists():
        best = max(cands, key=lambda c: c["val_score"])
        write(
            sel,
            {
                "winner": best["name"],
                "candidates": [{k: v for k, v in c.items() if k != "history"} for c in cands],
            },
        )
        ctx.log(f"winner on val: {best['name']} ({best['val_score']})")
    winner = read(sel)["winner"]
    for c in cands:
        if c["name"] != winner:
            (run_dir(ctx, c["name"], 0) / "best.pt").unlink(missing_ok=True)


def winner_name(ctx: Ctx) -> str:
    return read(ctx.work / "select.json")["winner"]


def stage_winner(ctx: Ctx) -> None:
    from . import encoder

    name = winner_name(ctx)
    data = train_data(ctx)
    for seed in SEEDS[1:]:
        d = run_dir(ctx, name, seed)
        if (d / "done.json").exists():
            ctx.log(f"skip {name} seed{seed} (done)")
            continue
        ctx.log(f"train {name} seed{seed}")
        s = encoder.train_run(name, seed, data, d, ctx.log, ctx.max_epochs, ctx.max_steps)
        ctx.log(f"done {name} seed{seed} val_score {s['val_score']} in {s['train_seconds']}s")
    test_texts = ctx.arrays("test")[0]
    for seed in SEEDS:  # test is touched here, for the winner only
        d = run_dir(ctx, name, seed)
        out, ckpt = d / "test_probs.npz", d / "best.pt"
        if not ckpt.exists():
            continue  # checkpoint already pruned after export; keep the saved predictions
        key = sha(ckpt, len(test_texts))
        if fresh(out, key):
            continue
        p = encoder.predict_texts(name, ckpt, {"test": test_texts})["test"]
        np.savez(out, **p)
        stamp(out, key)
        ctx.log(f"test predictions {name} seed{seed}")


def stage_baseline(ctx: Ctx) -> None:
    out = ctx.work / "baseline_probs.npz"
    key = sha(
        ctx.data / "dataset.jsonl", ctx.art / "boilerplate.json", len(ctx.rows), "baseline-v2"
    )
    if fresh(out, key):
        return
    alt = {"val": ctx.deploy_texts("val"), "test": ctx.deploy_texts("test")}
    res = baseline.predict_split(ctx.rows, "split_repo", 0, alt)
    arrays = {}
    for target, r in res.items():
        for k in ("val", "test", "alt_val", "alt_test"):
            arrays[f"{target}_{k}"] = r[k]
    for c, r in res["needs_info"]["by_c"].items():
        for k, v in r.items():
            arrays[f"needs_info_C{c}_{k}"] = v
    # refit on train/val/test text all rebuilt with the frozen list (deploy-time pipeline)
    frozen_rows = [
        {**r, "text": t} for r, t in zip(ctx.rows, ctx.deploy_rows(ctx.rows), strict=True)
    ]
    ref = baseline.predict_split(frozen_rows, "split_repo", 0)
    for target, r in ref.items():
        for k in ("val", "test"):
            arrays[f"refit_{target}_{k}"] = r[k]
    for c, r in ref["needs_info"]["by_c"].items():
        for k, v in r.items():
            arrays[f"refit_needs_info_C{c}_{k}"] = v
    np.savez(
        out,
        chosen_C=np.array([res["category"]["chosen_C"], res["needs_info"]["chosen_C"]]),
        refit_chosen_C=np.array([ref["category"]["chosen_C"], ref["needs_info"]["chosen_C"]]),
        **arrays,
    )
    stamp(out, key)
    ctx.log("baseline probabilities saved")


def exported_seed(ctx: Ctx) -> int:
    name = winner_name(ctx)
    scores = {s: read(run_dir(ctx, name, s) / "done.json")["val_score"] for s in SEEDS}
    return max(SEEDS, key=lambda s: (scores[s], -s))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _f1s(p: dict, y_cat, y_ni, thr: float, p_tuned: dict | None = None) -> dict:
    """Macro-F1s; p_tuned (default: p) supplies the probabilities for the tuned-threshold row."""
    lab = y_ni >= 0
    pt = (p_tuned or p)["needs_info"]
    return {
        "category": evalkit.macro_f1(y_cat, p["category"].argmax(1), len(CATEGORIES)),
        "needs_info_default": evalkit.macro_f1(
            y_ni[lab], p["needs_info"][lab].argmax(1), len(NEEDS_INFO)
        ),
        "needs_info_tuned": evalkit.macro_f1(
            y_ni[lab], (pt[lab, 1] >= thr).astype(int), len(NEEDS_INFO)
        ),
    }


def exported(ctx: Ctx) -> tuple[str, int, Path]:
    name, seed = winner_name(ctx), exported_seed(ctx)
    return name, seed, run_dir(ctx, name, seed)


def stage_export(ctx: Ctx) -> None:
    """fp32 ONNX, tokenizer files and labels.json for the exported seed."""
    from . import encoder

    out = ctx.work / "export.json"
    name, seed, d = exported(ctx)
    key = sha(d / "best.pt", "export-v2")
    if fresh(out, key):
        return
    ctx.log(f"export {name} seed{seed}")
    fp32_mb = encoder.export_onnx(name, d / "best.pt", ctx.art)
    base_id, _, lic = encoder.CANDIDATES[name]
    write(
        ctx.art / "labels.json",
        {
            "categories": list(CATEGORIES),
            "needs_info": list(NEEDS_INFO),
            "needs_info_positive_class": "yes",
            "needs_info_threshold": read(d / "done.json")["threshold"],
            "max_len": encoder.MAX_LEN,
            "pooling": "mean",
            "base_model": base_id,
            "base_model_license": lic,
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["category_logits", "needs_info_logits"],
        },
    )
    write(out, {"seed": seed, "fp32_mb": fp32_mb})
    stamp(out, key)


def _agree(a: dict, b: dict, thr: float) -> dict:
    return {
        "category": round(float((a["category"].argmax(1) == b["category"].argmax(1)).mean()), 4),
        "needs_info_argmax": round(
            float((a["needs_info"].argmax(1) == b["needs_info"].argmax(1)).mean()), 4
        ),
        "needs_info_threshold": round(
            float(((a["needs_info"][:, 1] >= thr) == (b["needs_info"][:, 1] >= thr)).mean()), 4
        ),
    }


def _ort_probs(encoder, path: Path, ids, pad_id) -> dict:
    c, n = encoder.ort_logits(path, ids, pad_id)
    return {"category": softmax(c), "needs_info": softmax(n)}


def stage_quant(ctx: Ctx) -> None:
    """Compare quantization configs on VAL only (agreement with fp32 ONNX); keep the best."""
    from transformers import AutoTokenizer

    from . import encoder

    out = ctx.work / "quant.json"
    _, _, d = exported(ctx)
    fp32, int8 = ctx.art / "model.onnx", ctx.art / "model.int8.onnx"
    key = sha(fp32, json.dumps(encoder.QUANT_CONFIGS, sort_keys=True), len(ctx.rows), "quant-v1")
    if fresh(out, key) and int8.exists():
        return
    thr = read(d / "done.json")["threshold"]
    tok = AutoTokenizer.from_pretrained(ctx.art)
    ids = encoder.encode(tok, ctx.arrays("val")[0])
    ref = _ort_probs(encoder, fp32, ids, tok.pad_token_id)
    tmp = ctx.work / "quant_tmp"
    tmp.mkdir(exist_ok=True)
    table = []
    for cname in encoder.QUANT_CONFIGS:
        path = tmp / f"{cname}.onnx"
        try:
            mb = encoder.quantize(fp32, path, cname)
            row = {
                "config": cname,
                "size_mb": mb,
                **_agree(_ort_probs(encoder, path, ids, tok.pad_token_id), ref, thr),
            }
        except Exception as e:  # a config the runtime cannot build or run is reported, not fatal
            row = {"config": cname, "error": f"{type(e).__name__}: {str(e)[:80]}"}
        ctx.log(f"quant val {json.dumps(row)}")
        table.append(row)
    ok = [r for r in table if "error" not in r and r["size_mb"] <= INT8_MAX_MB]
    score = lambda r: (r["category"] + r["needs_info_argmax"] + r["needs_info_threshold"]) / 3  # noqa: E731
    best = max(ok, key=lambda r: (round(score(r), 4), -r["size_mb"]))
    shutil.copyfile(tmp / f"{best['config']}.onnx", int8)
    shutil.rmtree(tmp)
    write(
        out,
        {
            "rule": "mean of val argmax agreement (category), needs_info argmax agreement and "
            f"needs_info agreement at the threshold, vs fp32 ONNX; size <= {INT8_MAX_MB} MB",
            "configs": table,
            "chosen": best["config"],
            "settings": encoder.QUANT_CONFIGS[best["config"]],
            "int8_mb": best["size_mb"],
            "val_category_agreement": best["category"],
            "val_gate_category_ge_0.98": best["category"] >= 0.98,
        },
    )
    stamp(out, key)
    ctx.log(f"chosen quantization: {best['config']}")


def stage_fidelity(ctx: Ctx) -> None:
    """Test-set fidelity and CPU latency of the chosen int8 file (test is used for this only)."""
    from transformers import AutoTokenizer

    from . import encoder

    out = ctx.work / "fidelity.json"
    _, _, d = exported(ctx)
    fp32, int8 = ctx.art / "model.onnx", ctx.art / "model.int8.onnx"
    key = sha(fp32, int8, d / "test_probs.npz", "fidelity-v1")
    if fresh(out, key):
        return
    thr = read(d / "done.json")["threshold"]
    tok = AutoTokenizer.from_pretrained(ctx.art)
    texts, y_cat, y_ni = ctx.arrays("test")
    ids = encoder.encode(tok, texts)
    probs = {
        "pytorch": dict(np.load(d / "test_probs.npz")),
        "fp32": _ort_probs(encoder, fp32, ids, tok.pad_token_id),
        "int8": _ort_probs(encoder, int8, ids, tok.pad_token_id),
    }
    f_torch = _f1s(probs["pytorch"], y_cat, y_ni, thr)
    f_int8 = _f1s(probs["int8"], y_cat, y_ni, thr)
    macro = {
        t: {
            "pytorch": round(f_torch[t], 4),
            "int8": round(f_int8[t], 4),
            "delta": round(f_int8[t] - f_torch[t], 4),
        }
        for t in f_torch
    }
    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), min(200, len(texts)), replace=False)]
    latency = {}
    for key_, path in (("fp32", fp32), ("int8", int8)):
        latency[key_] = encoder.latency(path, tok, sample)
        ctx.log(f"latency {key_}: {latency[key_]}")
    size_mb = round(os.path.getsize(int8) / 1e6, 1)
    gate = {
        "val_category_agreement_ge_0.98": read(ctx.work / "quant.json")[
            "val_gate_category_ge_0.98"
        ],
        "test_abs_delta_category_le_0.01": abs(macro["category"]["delta"]) <= 0.01,
        "test_abs_delta_needs_info_tuned_le_0.01": abs(macro["needs_info_tuned"]["delta"]) <= 0.01,
        "size_le_50mb": size_mb <= INT8_MAX_MB,
    }
    gate["passed"] = all(gate.values())
    write(
        out,
        {
            "agreement": {
                "int8 vs ONNX fp32": _agree(probs["int8"], probs["fp32"], thr),
                "ONNX fp32 vs PyTorch": _agree(probs["fp32"], probs["pytorch"], thr),
                "int8 vs PyTorch": _agree(probs["int8"], probs["pytorch"], thr),
            },
            "macro_f1": macro,
            "latency": latency,
            "sizes": {"fp32_mb": round(os.path.getsize(fp32) / 1e6, 1), "int8_mb": size_mb},
            "gate": gate,
        },
    )
    stamp(out, key)
    ctx.log(f"fidelity gate: {gate}")
    keep = exported_seed(ctx)
    name = winner_name(ctx)
    for s in SEEDS:  # only the exported checkpoint is kept
        if s != keep:
            (run_dir(ctx, name, s) / "best.pt").unlink(missing_ok=True)


def stage_deploy(ctx: Ctx) -> None:
    from . import encoder

    out = ctx.work / "deploy_model.npz"
    name, _, d = exported(ctx)
    key = sha(d / "best.pt", ctx.art / "boilerplate.json", len(ctx.rows))
    if fresh(out, key):
        return
    p = encoder.predict_texts(
        name,
        d / "best.pt",
        {"val": ctx.deploy_texts("val"), "test": ctx.deploy_texts("test")},
    )
    np.savez(
        out,
        **{f"{split}_{t}": p[split][t] for split in p for t in ("category", "needs_info")},
    )
    stamp(out, key)
    ctx.log("deploy-time model predictions saved")


# ------------------------------------------------------------------ report


def build_notes(res: dict) -> list[str]:
    notes = []
    top = sorted(res["candidates"], key=lambda c: -c["val_score"])[:2]
    if len(top) == 2 and top[0]["val_score"] - top[1]["val_score"] <= 0.005:
        a, b = top
        best_cat = max(top, key=lambda c: c["val_category_f1"])["name"]
        best_ni = max(top, key=lambda c: c["val_needs_info_f1"])["name"]
        notes.append(
            f"Validation scores of {a['name']} ({a['val_score']}) and {b['name']} "
            f"({b['val_score']}) are practically tied: {best_cat} is better on category, "
            f"{best_ni} on needs_info. The pick between them is not a strong result."
        )
    runs = res["candidates"] + res["seeds"]
    epochs = sorted({r["best_epoch"] for r in runs})
    run_epochs = sorted({r["epochs_run"] for r in runs})
    notes.append(
        f"Best epoch per run: {epochs}; epochs actually run: {run_epochs} (patience 1). No run "
        "improved after epoch 2, so the 4-epoch budget was never used and longer schedules "
        "or other learning rates were not explored."
    )
    thr = [s["threshold"] for s in res["seeds"]]
    ex = res["test"]["model_exported"]
    rel = "lower" if ex["needs_info_tuned"] < ex["needs_info_default"] else "not lower"
    notes.append(
        f"The val-tuned needs_info thresholds vary a lot across seeds ({thr}) and do not "
        f"transfer to the test repos: for the exported seed the tuned threshold gives "
        f"{ex['needs_info_tuned']} on test, {rel} than the {ex['needs_info_default']} at 0.5."
    )
    notes.append(
        "The encoder was trained on M1 text (template lines removed with statistics over each "
        "repo's own issues), but at deploy time only the frozen list can be applied: a "
        "train/inference cleaning mismatch. The deploy-time rows show its effect; the baseline "
        "refit row shows what training on the same cleaning would give."
    )
    notes.append(
        "artifacts/boilerplate.json contains some non-template lines, so it has to be filtered "
        "before it ships in the M3 demo. It is not published."
    )
    return notes


def stage_report(ctx: Ctx) -> None:
    from . import encoder

    name, seed_star, _ = exported(ctx)
    base_id, _, lic = encoder.CANDIDATES[name]
    sel = read(ctx.work / "select.json")
    test_rows = ctx.split_rows("test")
    _, y_cat, y_ni = ctx.arrays("test")
    lab = y_ni >= 0
    kc, kn = len(CATEGORIES), len(NEEDS_INFO)
    _, v_cat, v_ni = ctx.arrays("val")
    vlab = v_ni >= 0

    seeds, model_f1 = [], {"category": [], "needs_info_default": [], "needs_info_tuned": []}
    probs_by_seed = {}
    for s in SEEDS:
        d = run_dir(ctx, name, s)
        done, p = read(d / "done.json"), dict(np.load(d / "test_probs.npz"))
        probs_by_seed[s] = (p, done["threshold"])
        f = _f1s(p, y_cat, y_ni, done["threshold"])
        for k in model_f1:
            model_f1[k].append(f[k])
        seeds.append(
            {
                "seed": s,
                "val_score": done["val_score"],
                "threshold": done["threshold"],
                "best_epoch": done["best_epoch"],
                "epochs_run": done["epochs_run"],
                "test_category_f1": round(f["category"], 4),
                "test_needs_info_default_f1": round(f["needs_info_default"], 4),
                "test_needs_info_tuned_f1": round(f["needs_info_tuned"], 4),
            }
        )
    mp, thr_m = probs_by_seed[seed_star]

    # baseline: needs_info "tuned" row picks C and threshold together on val
    bz = np.load(ctx.work / "baseline_probs.npz")

    def ni_c(prefix, c, k):
        return bz[f"{prefix}needs_info_C{c}_{k}"]

    best = None
    for c in baseline.C_GRID:
        t_c, f_c = evalkit.best_threshold(v_ni[vlab], ni_c("", c, "val")[vlab, 1])
        if best is None or f_c > best[2]:
            best = (c, t_c, f_c)
    c_star, thr_b, val_ni_b = best
    base_test = {"category": bz["category_test"], "needs_info": bz["needs_info_test"]}
    base_tuned = {"category": bz["category_test"], "needs_info": ni_c("", c_star, "test")}
    f_base = _f1s(base_test, y_cat, y_ni, thr_b, base_tuned)
    f_mod = _f1s(mp, y_cat, y_ni, thr_m)

    reproduces = None
    m1 = ROOT / "results" / "baseline.json"
    if m1.exists():
        ref = read(m1)["splits"]["repo_split"]
        reproduces = {
            "category": round(f_base["category"], 4) == ref["category"]["test_macro_f1"],
            "needs_info": round(f_base["needs_info_default"], 4)
            == ref["needs_info"]["test_macro_f1"],
        }
    if not ctx.smoke and not (reproduces and all(reproduces.values())):
        raise RuntimeError(f"baseline no longer reproduces M1: {reproduces}")

    pred = {
        "base_cat": base_test["category"].argmax(1),
        "model_cat": mp["category"].argmax(1),
        "base_ni": base_test["needs_info"].argmax(1),
        "base_ni_tuned": (base_tuned["needs_info"][:, 1] >= thr_b).astype(int),
        "model_ni": (mp["needs_info"][:, 1] >= thr_m).astype(int),
    }
    boot = {
        "category": evalkit.bootstrap_diff(y_cat, pred["model_cat"], pred["base_cat"], kc),
        "needs_info_vs_baseline_tuned": evalkit.bootstrap_diff(
            y_ni[lab], pred["model_ni"][lab], pred["base_ni_tuned"][lab], kn
        ),
        "needs_info_vs_baseline_default": evalkit.bootstrap_diff(
            y_ni[lab], pred["model_ni"][lab], pred["base_ni"][lab], kn
        ),
    }
    repos = np.array([r["repo"] for r in test_rows])
    cat_rows = evalkit.per_repo(
        repos,
        y_cat,
        {"category_baseline": pred["base_cat"], "category_model": pred["model_cat"]},
        kc,
    )
    ni_rows = {
        r["repo"]: r
        for r in evalkit.per_repo(
            repos[lab],
            y_ni[lab],
            {
                "needs_info_baseline_default": pred["base_ni"][lab],
                "needs_info_baseline_tuned": pred["base_ni_tuned"][lab],
                "needs_info_model": pred["model_ni"][lab],
            },
            kn,
        )
    }
    per_repo = []
    for r in cat_rows:
        extra = ni_rows.get(r["repo"], {})
        per_repo.append(
            {
                **r,
                "n_needs_info": extra.get("n", 0),
                "needs_info_baseline_default": extra.get("needs_info_baseline_default"),
                "needs_info_baseline_tuned": extra.get("needs_info_baseline_tuned"),
                "needs_info_model": extra.get("needs_info_model"),
            }
        )

    # deploy-time cleaning: same fitted models and thresholds, new val/test text
    dz = np.load(ctx.work / "deploy_model.npz")
    dm = {s: {t: dz[f"{s}_{t}"] for t in ("category", "needs_info")} for s in ("val", "test")}

    def base_alt(split, tuned):
        return {
            "category": bz[f"category_alt_{split}"],
            "needs_info": ni_c("", c_star, f"alt_{split}")
            if tuned
            else bz[f"needs_info_alt_{split}"],
        }

    def rd(f):
        return {k: round(v, 4) for k, v in f.items()}

    base_val = {"category": bz["category_val"], "needs_info": bz["needs_info_val"]}
    base_val_tuned = {"category": bz["category_val"], "needs_info": ni_c("", c_star, "val")}
    dep_rows = []
    for split, ys, std, std_t in (
        ("test", (y_cat, y_ni), base_test, base_tuned),
        ("val", (v_cat, v_ni), base_val, base_val_tuned),
    ):
        dep_rows.append((f"baseline, {split}", "M1 text", rd(_f1s(std, *ys, thr_b, std_t))))
        dep_rows.append(
            (
                f"baseline, {split}",
                "frozen list at inference only",
                rd(_f1s(base_alt(split, False), *ys, thr_b, base_alt(split, True))),
            )
        )
        if split == "test":
            dep_rows.append((f"model, {split}", "M1 text", rd(_f1s(mp, *ys, thr_m))))
        dep_rows.append(
            (f"model, {split}", "frozen list at inference only", rd(_f1s(dm[split], *ys, thr_m)))
        )
    # baseline refit on train text rebuilt with the frozen list, thresholds tuned on its own val
    best_r = None
    for c in baseline.C_GRID:
        t_c, f_c = evalkit.best_threshold(v_ni[vlab], ni_c("refit_", c, "val")[vlab, 1])
        if best_r is None or f_c > best_r[2]:
            best_r = (c, t_c, f_c)
    rc, rthr, _ = best_r
    refit = {"category": bz["refit_category_test"], "needs_info": bz["refit_needs_info_test"]}
    refit_t = {"category": bz["refit_category_test"], "needs_info": ni_c("refit_", rc, "test")}
    dep_rows.insert(
        2,
        (
            "baseline refit, test",
            "frozen list in train and test",
            rd(_f1s(refit, y_cat, y_ni, rthr, refit_t)),
        ),
    )
    deploy = {
        "n_train_repos": len(ctx.train_repos),
        "n_frozen_lines": len(ctx.frozen_set()),
        "rows": [[a, b, c] for a, b, c in dep_rows],
        "note": "encoder trained on M1 text; thresholds and C chosen on M1 val text and kept",
    }

    exp = {
        "fp32_mb": read(ctx.work / "export.json")["fp32_mb"],
        "quant": read(ctx.work / "quant.json"),
        "fidelity": read(ctx.work / "fidelity.json"),
    }
    result = {
        "smoke": ctx.smoke,
        "candidates": sel["candidates"],
        "chosen": name,
        "base_model": base_id,
        "license": lic,
        "exported_seed": seed_star,
        "seeds": seeds,
        "test": {
            "n_category": int(len(y_cat)),
            "n_needs_info": int(lab.sum()),
            "baseline": {
                "category": round(f_base["category"], 4),
                "needs_info_default": round(f_base["needs_info_default"], 4),
                "needs_info_tuned": round(f_base["needs_info_tuned"], 4),
                "threshold": thr_b,
                "tuned_C": c_star,
                "val_needs_info_f1_tuned": round(val_ni_b, 4),
                "reproduces_m1": reproduces,
            },
            "model_mean": {k: evalkit.mean_std(v) for k, v in model_f1.items()},
            "model_exported": {
                "category": round(f_mod["category"], 4),
                "needs_info_default": round(f_mod["needs_info_default"], 4),
                "needs_info_tuned": round(f_mod["needs_info_tuned"], 4),
                "threshold": thr_m,
            },
        },
        "bootstrap": boot,
        "per_repo": per_repo,
        "deploy_time": deploy,
        "export": exp,
    }
    result["notes"] = build_notes(result)
    write(ctx.results / "finetune.json", result)
    (ctx.results / "finetune.md").write_text(
        report.render_markdown(result) + "\n", encoding="utf-8"
    )
    write_predictions(ctx, test_rows, base_test, mp, pred)
    ctx.log(f"results written to {ctx.results}")


def write_predictions(ctx: Ctx, rows, base, mod, pred) -> None:
    cols = ["repo", "number", "true_category", "true_needs_info"]
    cols += ["baseline_category", "baseline_needs_info", "baseline_needs_info_tuned"]
    cols += ["model_category", "model_needs_info"]
    cols += [f"baseline_p_{c}" for c in CATEGORIES] + ["baseline_p_needs_info_yes"]
    cols += [f"model_p_{c}" for c in CATEGORIES] + ["model_p_needs_info_yes"]
    with (ctx.results / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for i, r in enumerate(rows):
            w.writerow(
                [r["repo"], r["number"], r["category"], r["needs_info"] or ""]
                + [
                    CATEGORIES[pred["base_cat"][i]],
                    NEEDS_INFO[pred["base_ni"][i]],
                    NEEDS_INFO[pred["base_ni_tuned"][i]],
                    CATEGORIES[pred["model_cat"][i]],
                    NEEDS_INFO[pred["model_ni"][i]],
                ]
                + [f"{x:.4f}" for x in base["category"][i]]
                + [f"{base['needs_info'][i, 1]:.4f}"]
                + [f"{x:.4f}" for x in mod["category"][i]]
                + [f"{mod['needs_info'][i, 1]:.4f}"]
            )


# ------------------------------------------------------------------ main


def stage_bench(ctx: Ctx) -> None:
    from . import encoder

    texts = [r["text"] for r in ctx.rows if r["split_repo"] == "train"][:2000]
    for name in ("minilm-l12", "granite-30m", "electra-small"):
        ctx.log(f"bench {json.dumps(encoder.bench(name, texts))}")


RUNNERS = {
    "frozen": stage_frozen,
    "candidates": stage_candidates,
    "select": stage_select,
    "winner": stage_winner,
    "baseline": stage_baseline,
    "export": stage_export,
    "quant": stage_quant,
    "fidelity": stage_fidelity,
    "deploy": stage_deploy,
    "report": stage_report,
}


def run_baseline_elsewhere(ctx: Ctx, args) -> None:
    import subprocess
    import sys

    cmd = [str(args.baseline_python), "-m", "tiny_triage.finetune", "baseline"]
    cmd += ["--data", str(args.data), "--runs", str(args.runs), "--artifacts", str(args.artifacts)]
    cmd += ["--smoke"] * args.smoke
    ctx.log("baseline stage in " + str(args.baseline_python))
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tiny_triage.finetune")
    p.add_argument("stage", choices=("all", "bench", *STAGES), help="'all' runs every stage")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--results", type=Path, help="default: the repo's results/ folder")
    p.add_argument("--hf-home", type=Path, help="Hugging Face cache (sets HF_HOME)")
    p.add_argument(
        "--baseline-python",
        type=Path,
        help="interpreter for the baseline stage (the project's numpy 2 env reproduces M1 "
        "exactly; torch 2.2 forces numpy<2, which shifts LR convergence slightly)",
    )
    p.add_argument("--max-epochs", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="tiny subset, 1 short epoch, own folders")
    args = p.parse_args(argv)
    os.environ.setdefault("USE_TF", "0")  # keep a global tensorflow install out of transformers
    os.environ.setdefault("USE_FLAX", "0")
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
    ctx = Ctx(args)
    if args.stage == "bench":
        stage_bench(ctx)
        return
    for stage in STAGES if args.stage == "all" else (args.stage,):
        ctx.log(f"== stage {stage}")
        if stage == "baseline" and args.baseline_python:
            run_baseline_elsewhere(ctx, args)
        else:
            RUNNERS[stage](ctx)
    ctx.log("finished")


if __name__ == "__main__":
    main()
