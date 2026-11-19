"""M3 demo cleaning: which template-line list the browser demo applies, chosen on validation.

python -m tiny_triage.demo STAGE --data DATA --runs RUNS --artifacts ART

Stages (each is skipped when its output exists and its inputs are unchanged):
  lists     filter the frozen M2 list down to clearly template-like lines (torch-free)
  baseline  fit TF-IDF + LR on train, score val/test under each cleaning variant (project env)
  model     int8 ONNX, batch 1, val/test under each variant (needs onnxruntime + transformers)
  report    pick the variant on validation, write results/demo.json and results/demo.md

Variants: "none" (clean_body only), "full" (the whole frozen list), "filtered" (the filtered list).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from . import dataset, evalkit
from .finetune import fresh, sha, stamp
from .labels import CATEGORIES, NEEDS_INFO
from .report import table

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("none", "full", "filtered")
SHIPPABLE = ("none", "filtered")
TIE = 0.002

# Lines kept from the frozen list: form headings, bold or bulleted field labels, and a few
# template prompts. Everything else in the frozen list is a pasted environment dump, config
# or code line that only looked repetitive.
_HEADING = re.compile(r"^#{1,4} \S")
_BOLD_LABEL = re.compile(r"^\*\*.+\*\*:?$")
_BULLET_LABEL = re.compile(r"^\* [A-Za-z][A-Za-z ()/]*:$")
_NUMBERED_PROMPT = re.compile(r"^\d\. (?:What|How|Why)\b.*\?$")
_PROMPT_PREFIXES = (
    "Add more configuration information here",
    "Don't forget to mention",
    "Paste the output of",
    "I tried this code",
)
# anything that can carry user content or an identifying detail is never shipped
_NEVER = re.compile(r"<url>|https?:|@|\\|\d{4,}|\bC:|/(?:etc|usr|home|bin|var)\b", re.I)


def keep_line(line: str) -> bool:
    if not line or _NEVER.search(line):
        return False
    if _HEADING.match(line) or _BOLD_LABEL.match(line) or _BULLET_LABEL.match(line):
        return True
    if _NUMBERED_PROMPT.match(line):
        return True
    return line.startswith(_PROMPT_PREFIXES)


def filter_boilerplate(lines: list[str]) -> list[str]:
    return sorted(ln for ln in lines if keep_line(ln))


# ------------------------------------------------------------------ helpers


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


class Demo:
    def __init__(self, args):
        self.data, self.art = args.data, args.artifacts
        self.work = args.runs / "demo"
        self.work.mkdir(parents=True, exist_ok=True)
        self.results = args.results or ROOT / "results"
        self._rows = None
        self._raw = None

    def rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = dataset.read_dataset(self.data / "dataset.jsonl")
        return self._rows

    def raw(self) -> dict:
        if self._raw is None:
            self._raw = evalkit.index_raw(dataset.load_raw(self.data / "raw"))
        return self._raw

    def split(self, name: str) -> list[dict]:
        return [r for r in self.rows() if r["split_repo"] == name]

    def lists(self) -> dict[str, set[str]]:
        full = read(self.art / "boilerplate.json")["lines"]
        return {
            "none": set(),
            "full": set(full),
            "filtered": set(read(self.art / "boilerplate.filtered.json")["lines"]),
        }

    def texts(self, variant: str, split: str) -> list[str]:
        boiler = self.lists()[variant]
        raw = self.raw()
        out = []
        for r in self.split(split):
            i = raw[(r["repo"], r["number"])]
            out.append(evalkit.deploy_text(i["title"], i.get("body"), boiler))
        return out

    def arrays(self, split: str):
        rows = self.split(split)
        y_cat = np.array([CATEGORIES.index(r["category"]) for r in rows])
        y_ni = np.array(
            [-1 if r["needs_info"] is None else NEEDS_INFO.index(r["needs_info"]) for r in rows]
        )
        return y_cat, y_ni


# ------------------------------------------------------------------ stages


def stage_lists(d: Demo) -> None:
    out = d.art / "boilerplate.filtered.json"
    src = d.art / "boilerplate.json"
    key = sha(
        src,
        "filter-v2",
        *_PROMPT_PREFIXES,
        _NEVER.pattern,
        _HEADING.pattern,
        _BOLD_LABEL.pattern,
        _BULLET_LABEL.pattern,
        _NUMBERED_PROMPT.pattern,
    )
    if fresh(out, key):
        return
    full = read(src)["lines"]
    kept = filter_boilerplate(full)
    write(
        out,
        {
            "note": "frozen template lines from the 22 train repos, filtered to headings, "
            "field labels and template prompts",
            "n_full": len(full),
            "lines": kept,
        },
    )
    stamp(out, key)
    print(f"filtered list: {len(kept)} of {len(full)} lines")


def stage_baseline(d: Demo) -> None:
    from . import baseline

    out = d.work / "baseline_probs.npz"
    key = sha(
        d.data / "dataset.jsonl",
        d.art / "boilerplate.json",
        d.art / "boilerplate.filtered.json",
        "demo-baseline-v1",
    )
    if fresh(out, key):
        return
    rows = d.rows()
    arrays = {}
    for target in baseline.TARGETS:
        fit = baseline.fit_target(rows, "split_repo", target, 0)
        arrays[f"{target}_chosen_C"] = np.array([fit.best_c])
        for variant in VARIANTS:
            for split in ("val", "test"):
                arrays[f"{target}_{variant}_{split}"] = fit.proba(d.texts(variant, split))
        print(f"baseline {target}: C={fit.best_c}", flush=True)
    np.savez(out, **arrays)
    stamp(out, key)


def stage_model(d: Demo, variant: str, threads: int) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    out = d.work / f"model_probs_{variant}.npz"
    model = d.art / "model.int8.onnx"
    key = sha(
        model,
        d.data / "dataset.jsonl",
        d.art / "boilerplate.json",
        d.art / "boilerplate.filtered.json",
        variant,
        "demo-model-v1",
    )
    if fresh(out, key):
        return
    tok = AutoTokenizer.from_pretrained(d.art)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    max_len = read(d.art / "labels.json")["max_len"]
    arrays = {}
    for split in ("val", "test"):
        texts = d.texts(variant, split)
        cat = np.zeros((len(texts), 4), dtype=np.float32)
        ni = np.zeros((len(texts), 2), dtype=np.float32)
        for i, t in enumerate(texts):
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
            feed = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
            c, n = sess.run(None, feed)
            cat[i], ni[i] = softmax(c)[0], softmax(n)[0]
            if i % 1000 == 0:
                print(f"{variant} {split} {i}/{len(texts)}", flush=True)
        arrays[f"category_{split}"], arrays[f"needs_info_{split}"] = cat, ni
    np.savez(out, **arrays)
    stamp(out, key)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _scores(cat_p, ni_p, y_cat, y_ni, thr: float) -> dict:
    lab = y_ni >= 0
    return {
        "category": round(evalkit.macro_f1(y_cat, cat_p.argmax(1), 4), 4),
        "needs_info": round(evalkit.macro_f1(y_ni[lab], (ni_p[lab, 1] >= thr).astype(int), 2), 4),
    }


def stage_report(d: Demo) -> None:
    labels = read(d.art / "labels.json")
    thr = labels["needs_info_threshold"]
    base = np.load(d.work / "baseline_probs.npz")
    y = {s: d.arrays(s) for s in ("val", "test")}
    res: dict = {"needs_info_threshold": thr, "variants": {}}
    for v in VARIANTS:
        mod = np.load(d.work / f"model_probs_{v}.npz")
        row = {}
        for split in ("val", "test"):
            yc, yn = y[split]
            lab = yn >= 0
            b_ni = base[f"needs_info_{v}_{split}"]
            row[split] = {
                "model": _scores(mod[f"category_{split}"], mod[f"needs_info_{split}"], yc, yn, thr),
                "baseline": _scores(base[f"category_{v}_{split}"], b_ni, yc, yn, 0.5),
            }
            row[split]["model"]["needs_info_0.5"] = round(
                evalkit.macro_f1(
                    yn[lab], (mod[f"needs_info_{split}"][lab, 1] >= 0.5).astype(int), 2
                ),
                4,
            )
            row[split]["model_vs_baseline_category_agree"] = round(
                float(
                    (
                        mod[f"category_{split}"].argmax(1)
                        == base[f"category_{v}_{split}"].argmax(1)
                    ).mean()
                ),
                4,
            )
        m = row["val"]["model"]
        row["val_score"] = round((m["category"] + m["needs_info"]) / 2, 4)
        res["variants"][v] = row
    # the full list is reported for reference but cannot ship (it holds non-template lines),
    # so only none and filtered compete; a validation gap under 0.002 is a tie, and a tie
    # goes to none because it needs no list in the page
    val = {v: res["variants"][v]["val_score"] for v in SHIPPABLE}
    chosen = "filtered" if val["filtered"] - val["none"] >= TIE else "none"
    res["chosen"] = chosen
    res["rule"] = (
        "highest validation score for the int8 model among the variants that can ship (none, "
        "filtered; the full list holds non-template lines): mean of category macro-F1 and "
        f"needs_info macro-F1 at the labels.json threshold {thr}. A gap under {TIE} counts as "
        "a tie and goes to none, which needs no list"
    )
    res["n"] = {s: int(len(y[s][0])) for s in y}
    res["n_needs_info_labelled"] = {s: int((y[s][1] >= 0).sum()) for s in y}
    res["lines"] = {
        "full": len(d.lists()["full"]),
        "filtered": len(d.lists()["filtered"]),
    }
    write(d.results / "demo.json", res)
    (d.results / "demo.md").write_text(render(res), encoding="utf-8")
    print(f"chosen: {chosen}")


def render(res: dict) -> str:
    out = [
        "# M3: cleaning for the browser demo",
        "",
        "The M1 text cleaning removes template lines using statistics over each repo's own "
        "issues, which a pasted issue in the demo cannot do. The alternative is a frozen list "
        "of template lines built from the 22 train repos. Three variants, all scored "
        "with the exported int8 model at batch 1 and with the TF-IDF baseline (fitted on the "
        "M1 train text, so the cleaning here only changes what it is asked to score):",
        "",
        "- none: `clean_body` only, no template list",
        f"- full: all {res['lines']['full']} frozen lines (not published, it has non-template "
        "lines)",
        f"- filtered: {res['lines']['filtered']} lines kept, headings, field labels and template "
        "prompts only",
        "",
        f"Rule: {res['rule']}. Chosen on validation: **{res['chosen']}**. needs_info for the "
        "model uses the threshold the demo uses; the baseline uses 0.5. Macro-F1, repo-disjoint "
        f"split; val n={res['n']['val']} (needs_info {res['n_needs_info_labelled']['val']}), "
        f"test n={res['n']['test']} (needs_info {res['n_needs_info_labelled']['test']}).",
        "",
    ]
    for split in ("val", "test"):
        out.append(f"## {split}")
        out.append("")
        rows = []
        for v, r in res["variants"].items():
            s = r[split]
            rows.append(
                [
                    v,
                    s["model"]["category"],
                    s["model"]["needs_info"],
                    s["model"]["needs_info_0.5"],
                    s["baseline"]["category"],
                    s["baseline"]["needs_info"],
                ]
            )
        out += table(
            [
                "variant",
                "model category",
                "model needs_info (demo thr)",
                "model needs_info (0.5)",
                "baseline category",
                "baseline needs_info (0.5)",
            ],
            rows,
        )
    scores = ", ".join(f"{v} {r['val_score']}" for v, r in res["variants"].items())
    out.append(f"Validation score of the model (mean of its two macro-F1s): {scores}.")
    out.append("")
    out.append(
        "The three variants are within 0.001 of each other on validation, so the choice is "
        "close to a coin flip. Test is shown for all three for completeness; only validation "
        "was used to choose."
    )
    out.append("")
    if res["chosen"] == "none":
        out.append(
            "The demo therefore applies no template list: it runs `clean_body` and cuts the "
            "text, nothing else."
        )
    else:
        out.append(f"The demo applies the {res['chosen']} list.")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ site data

MODEL_FILES = (
    "model.int8.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
# What the page fetches from jsDelivr on the first run (transformers.web.min.js, ort.wasm.min.mjs,
# ort-wasm-simd-threaded.mjs and .wasm), kB as transferred with compression, measured with the
# end-to-end test at the pinned versions. Uncompressed they are about 11.7 MB.
RUNTIME_KB = 1937


def _pred(cat_p, ni_p, i, thr, labelled):
    return {
        "category": CATEGORIES[int(cat_p[i].argmax())],
        "needs_info": NEEDS_INFO[int(ni_p[i, 1] >= thr)],
    }


def _matches(rows, key, field):
    pairs = [
        (r[key][field], r["maintainer"][field])
        for r in rows
        if r["maintainer"][field] is not None and r[key][field] != "unparsed"
    ]
    return sum(a == b for a, b in pairs)


def stage_site(d: Demo, docs: Path) -> None:
    res = read(d.results / "demo.json")
    chosen = res["chosen"]
    labels = read(d.art / "labels.json")
    thr = labels["needs_info_threshold"]
    ft = read(d.results / "finetune.json")
    base_json = read(d.results / "baseline.json")["splits"]["repo_split"]

    # model folder
    mdir = docs / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copyfile(d.art / name, mdir / name)
    quant = ft["export"]["quant"]
    labels["quantization"] = {
        "chosen": quant["chosen"],
        **quant["settings"],
        "chosen_on": "validation: agreement with the fp32 ONNX model, "
        f"{quant['val_category_agreement']} on category",
        "runtime": "onnxruntime-web wasm, one thread",
    }
    cleaning = {
        "variant": chosen,
        "steps": "clean_body, then drop_lines with the list (if any), then title + body cut "
        "to max_body_chars; see docs/js/clean.js",
        "max_body_chars": 2000,
        "boilerplate": None,
        "chosen_on": "validation, see results/demo.md",
    }
    if chosen == "filtered":
        cleaning["boilerplate"] = "boilerplate.json"
        shutil.copyfile(d.art / "boilerplate.filtered.json", mdir / "boilerplate.json")
    labels["cleaning"] = cleaning
    write(mdir / "labels.json", labels)

    # table
    from . import llama

    sample = read(d.work.parent / llama.DEFAULT_DIR / "sample.json")
    cache = d.work.parent / llama.DEFAULT_DIR / "responses"
    boiler = d.lists()[chosen]
    test_rows = d.split("test")
    index = {(r["repo"], r["number"]): i for i, r in enumerate(test_rows)}
    mod = np.load(d.work / f"model_probs_{chosen}.npz")
    base = np.load(d.work / "baseline_probs.npz")
    raw = d.raw()
    rows = []
    for it in sample["issues"]:
        i = index[(it["repo"], it["number"])]
        key = f"{it['repo'].replace('/', '__')}__{it['number']}"
        rec = json.loads((cache / f"{key}.json").read_text(encoding="utf-8"))
        src = raw[(it["repo"], it["number"])]
        text = evalkit.deploy_text(src["title"], src.get("body"), boiler)
        # the Llama answers must come from the text the shipped cleaning produces now
        assert rec["variant"] == chosen, (key, rec["variant"], chosen)
        assert rec["input_sha256"] == llama.text_sha(text), f"{key}: Llama input text differs"
        answer = rec["parsed"]
        rows.append(
            {
                "repo": it["repo"],
                "number": it["number"],
                "url": f"https://github.com/{it['repo']}/issues/{it['number']}",
                "title": raw[(it["repo"], it["number"])]["title"],
                "maintainer": {"category": it["category"], "needs_info": it["needs_info"]},
                "baseline": _pred(
                    base[f"category_{chosen}_test"], base[f"needs_info_{chosen}_test"], i, 0.5, 1
                ),
                "model": _pred(mod["category_test"], mod["needs_info_test"], i, thr, 1),
                "llama": answer,
            }
        )
    n_ni = sum(r["maintainer"]["needs_info"] is not None for r in rows)
    cats = Counter(r["maintainer"]["category"] for r in rows)
    nis = Counter(r["maintainer"]["needs_info"] for r in rows if r["maintainer"]["needs_info"])
    top_cat, top_ni = cats.most_common(1)[0], nis.most_common(1)[0]
    summ = {
        "majority": {
            "category": {"class": top_cat[0], "count": top_cat[1]},
            "needs_info": {"class": top_ni[0], "count": top_ni[1]},
        },
        "category": {k: _matches(rows, k, "category") for k in ("baseline", "model", "llama")},
        "needs_info": {
            "n": n_ni,
            **{k: _matches(rows, k, "needs_info") for k in ("baseline", "model", "llama")},
        },
        "llama_unparsed": {
            f: sum(r["llama"][f] == "unparsed" for r in rows) for f in ("category", "needs_info")
        },
    }
    write(
        docs / "data" / "table.json",
        {
            "n": len(rows),
            "seed": sample["seed"],
            "method": sample["method"],
            "cleaning": chosen,
            "llama_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "needs_info_thresholds": {"tiny-triage": thr, "baseline": 0.5},
            "rows": rows,
            "summary": summ,
        },
    )

    # summary
    all_rows = d.rows()
    repos_by = {
        s: {r["repo"] for r in all_rows if r["split_repo"] == s} for s in ("train", "val", "test")
    }
    boot = ft["bootstrap"]
    t = ft["test"]
    ch = res["variants"][chosen]["test"]
    model_mb = round((d.art / "model.int8.onnx").stat().st_size / 1e6, 1)
    total_kb = sum((d.art / n).stat().st_size for n in MODEL_FILES) / 1e3 + RUNTIME_KB
    write(
        docs / "data" / "summary.json",
        {
            "source": "results/finetune.json, results/baseline.json, results/demo.json",
            "repos": {
                "total": sum(len(v) for v in repos_by.values()),
                **{k: len(v) for k, v in repos_by.items()},
            },
            "issues_total": len(all_rows),
            "test": {
                "n_category": t["n_category"],
                "n_needs_info": t["n_needs_info"],
                "category": {
                    "model": t["model_exported"]["category"],
                    "baseline": t["baseline"]["category"],
                    "majority": base_json["category"]["majority_macro_f1"],
                    "diff": boot["category"]["diff"],
                    "ci_low": boot["category"]["ci_low"],
                    "ci_high": boot["category"]["ci_high"],
                },
                "needs_info": {
                    "model": t["model_exported"]["needs_info_tuned"],
                    "baseline": t["baseline"]["needs_info_tuned"],
                    "model_default": t["model_exported"]["needs_info_default"],
                    "baseline_default": t["baseline"]["needs_info_default"],
                    "majority": base_json["needs_info"]["majority_macro_f1"],
                    "diff": boot["needs_info_vs_baseline_tuned"]["diff"],
                    "ci_low": boot["needs_info_vs_baseline_tuned"]["ci_low"],
                    "ci_high": boot["needs_info_vs_baseline_tuned"]["ci_high"],
                },
            },
            "demo": {
                "cleaning": chosen,
                "category": {
                    "model": ch["model"]["category"],
                    "baseline": ch["baseline"]["category"],
                },
                "needs_info": {
                    "model": ch["model"]["needs_info"],
                    "baseline": ch["baseline"]["needs_info"],
                },
            },
            "needs_info_threshold": thr,
            "model_mb": model_mb,
            "download_mb": math.ceil(total_kb / 1e3),
        },
    )
    print(f"site data written under {docs} (cleaning {chosen}, {math.ceil(total_kb / 1e3)} MB)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tiny_triage.demo")
    p.add_argument("stage", choices=("lists", "baseline", "model", "report", "site"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--results", type=Path)
    p.add_argument("--variant", choices=VARIANTS, help="model stage: one variant (default all)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--docs", type=Path, default=ROOT / "docs")
    args = p.parse_args(argv)
    d = Demo(args)
    if args.stage == "lists":
        stage_lists(d)
    elif args.stage == "baseline":
        stage_baseline(d)
    elif args.stage == "model":
        for v in [args.variant] if args.variant else VARIANTS:
            stage_model(d, v, args.threads)
    elif args.stage == "report":
        stage_report(d)
    else:
        stage_site(d, args.docs)


if __name__ == "__main__":
    main()
