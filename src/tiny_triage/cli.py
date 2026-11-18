"""Command line: collect, build, baseline.

tiny-triage collect  --raw DIR [--manifest data/manifest.csv] [--only owner/name]
tiny-triage build    --raw DIR --out DIR      (writes dataset.jsonl and data/manifest.csv)
tiny-triage baseline --data DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import baseline, collect, dataset, labels

ROOT = Path(__file__).resolve().parents[2]
LABEL_MAP = ROOT / "data" / "label_map.yml"
MANIFEST = ROOT / "data" / "manifest.csv"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tiny-triage")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="fetch labelled issues with gh into a local folder")
    c.add_argument("--raw", type=Path, required=True)
    c.add_argument("--manifest", type=Path, help="rebuild exactly the issues in this manifest")
    c.add_argument("--only", action="append")
    b = sub.add_parser("build", help="clean text, split, write dataset and manifest")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("baseline", help="TF-IDF + logistic regression on both splits")
    m.add_argument("--data", type=Path, required=True)
    args = p.parse_args(argv)

    if args.cmd == "collect":
        lm = labels.load_map(LABEL_MAP)
        rows = dataset.read_manifest(args.manifest) if args.manifest else None
        used = collect.collect_all(lm, args.raw, rows, args.only)
        print(f"{used} API requests")
    elif args.cmd == "build":
        rows = dataset.build(dataset.load_raw(args.raw), seed=args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        dataset.write_dataset(rows, args.out / "dataset.jsonl")
        dataset.write_manifest(rows, MANIFEST)
        print(f"{len(rows)} issues -> {MANIFEST}")
    else:
        rows = dataset.read_dataset(args.data / "dataset.jsonl")
        (ROOT / "results").mkdir(exist_ok=True)
        baseline.run_all(rows, ROOT / "results" / "baseline.json", ROOT / "results" / "baseline.md")
