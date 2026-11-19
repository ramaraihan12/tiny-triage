"""Torch side of M2: encoder + mean pooling + two heads, training loop, ONNX export.

Heavy imports (torch, transformers, onnx) live only in this module. CI never imports it.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import evalkit
from .labels import CATEGORIES, NEEDS_INFO

MAX_LEN = 256
BATCH = 32
WEIGHT_DECAY = 0.01
WARMUP = 0.06
MAX_EPOCHS = 4
PATIENCE = 1

# Fixed learning rate per model, not tuned: the small ELECTRA is usually fine-tuned with a
# higher rate (1e-4) than the BERT-style sentence encoders (5e-5).
CANDIDATES = {
    "electra-small": ("google/electra-small-discriminator", 1e-4, "Apache-2.0"),
    "minilm-l12": ("sentence-transformers/all-MiniLM-L12-v2", 5e-5, "Apache-2.0"),
    "granite-30m": ("ibm-granite/granite-embedding-30m-english", 5e-5, "Apache-2.0"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_model(model_id: str) -> str:
    """Download only config, tokenizer and one weights file; return the local folder."""
    from huggingface_hub import list_repo_files, snapshot_download

    try:
        top = {f for f in list_repo_files(model_id) if "/" not in f}
    except Exception:  # offline or flaky network: use whatever is already cached
        top = None
    if top is None:
        return snapshot_download(model_id, local_files_only=True)
    weights = "model.safetensors" if "model.safetensors" in top else "pytorch_model.bin"
    patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        weights,
    ]
    return snapshot_download(model_id, allow_patterns=patterns)


class Classifier(nn.Module):
    """Encoder, attention-masked mean pooling, a category head and a needs_info head."""

    def __init__(self, base_path: str, dropout: float = 0.1, attn: str | None = None):
        super().__init__()
        kwargs = {"attn_implementation": attn} if attn else {}
        self.encoder = AutoModel.from_pretrained(base_path, **kwargs)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.category = nn.Linear(hidden, len(CATEGORIES))
        self.needs_info = nn.Linear(hidden, len(NEEDS_INFO))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        pooled = self.drop(pooled)
        return self.category(pooled), self.needs_info(pooled)


def encode(tok, texts: list[str], max_len: int = MAX_LEN) -> list[list[int]]:
    return tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]


def collate(batch, pad_id: int):
    width = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, *_rest) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    cat = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ni = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return ids, mask, cat, ni


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    """Balanced weights n / (k * count), the same rule as sklearn's class_weight='balanced'."""
    y = y[y >= 0]
    counts = np.bincount(y, minlength=k).astype(float)
    return torch.tensor(len(y) / (k * np.maximum(counts, 1)), dtype=torch.float32)


@torch.no_grad()
def predict(model: Classifier, ids: list[list[int]], pad_id: int, device, batch: int = 64):
    """fp32 softmax probabilities (category, needs_info) in the input order."""
    model.eval()
    order = np.argsort([len(x) for x in ids])
    cat = np.zeros((len(ids), len(CATEGORIES)), dtype=np.float32)
    ni = np.zeros((len(ids), len(NEEDS_INFO)), dtype=np.float32)
    for start in range(0, len(ids), batch):
        sel = order[start : start + batch]
        x, m, _, _ = collate([(ids[i], 0, 0) for i in sel], pad_id)
        c, n = model(x.to(device), m.to(device))
        cat[sel] = torch.softmax(c.float(), -1).cpu().numpy()
        ni[sel] = torch.softmax(n.float(), -1).cpu().numpy()
    return cat, ni


def val_metrics(cat_p, ni_p, y_cat, y_ni) -> dict:
    """Val score = mean of category macro-F1 and needs_info macro-F1 at the val-tuned threshold."""
    cat_f1 = evalkit.macro_f1(y_cat, cat_p.argmax(1), len(CATEGORIES))
    lab = y_ni >= 0
    thr, ni_f1 = evalkit.best_threshold(y_ni[lab], ni_p[lab, 1])
    return {
        "val_category_f1": round(cat_f1, 4),
        "val_needs_info_f1": round(ni_f1, 4),
        "threshold": thr,
        "val_score": round((cat_f1 + ni_f1) / 2, 4),
    }


def train_run(
    name: str,
    seed: int,
    data: dict,
    run_dir: Path,
    log,
    max_epochs: int = MAX_EPOCHS,
    max_steps: int | None = None,
) -> dict:
    """Train one candidate with one seed. Saves best.pt and val_probs.npz, returns a summary.

    data: {"train": (texts, y_cat, y_ni), "val": (...)} with integer labels, -1 for null.
    """
    base_id, lr, _ = CANDIDATES[name]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    path = fetch_model(base_id)
    tok = AutoTokenizer.from_pretrained(path)
    model = Classifier(path).to(device)
    pad_id = tok.pad_token_id
    x_tr, c_tr, n_tr = data["train"]
    x_va, c_va, n_va = data["val"]
    ids_tr, ids_va = encode(tok, x_tr), encode(tok, x_va)
    w_cat = class_weights(c_tr, len(CATEGORIES)).to(device)
    w_ni = class_weights(n_tr, len(NEEDS_INFO)).to(device)

    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        list(zip(ids_tr, c_tr.tolist(), n_tr.tolist())),
        batch_size=BATCH,
        shuffle=True,
        generator=gen,
        num_workers=0,
        collate_fn=lambda b: collate(b, pad_id),
    )
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    total = len(loader) * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ce_cat = nn.CrossEntropyLoss(weight=w_cat)
    ce_ni = nn.CrossEntropyLoss(weight=w_ni, ignore_index=-1)

    best, best_epoch, bad, history = None, 0, 0, []
    t0 = time.time()
    step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for ids, mask, y_c, y_n in loader:
            ids, mask, y_c, y_n = (t.to(device) for t in (ids, mask, y_c, y_n))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lc, ln = model(ids, mask)
            loss = ce_cat(lc.float(), y_c)
            if (y_n >= 0).any():
                loss = loss + ce_ni(ln.float(), y_n)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                log(f"{name} seed{seed} epoch {epoch} step {step}/{total} loss {loss.item():.4f}")
            if max_steps and step >= max_steps:
                break
        cat_p, ni_p = predict(model, ids_va, pad_id, device)
        m = val_metrics(cat_p, ni_p, c_va, n_va)
        m.update(epoch=epoch, train_loss=round(running / max(1, len(loader)), 4))
        history.append(m)
        log(f"{name} seed{seed} epoch {epoch}: {json.dumps(m)}")
        if best is None or m["val_score"] > best["val_score"]:
            best, best_epoch, bad = m, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            np.savez(run_dir / "val_probs.npz", category=cat_p, needs_info=ni_p)
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"{name} seed{seed}: early stop after epoch {epoch}")
                break
        if max_steps and step >= max_steps:
            break
    summary = {
        "name": name,
        "model_id": base_id,
        "seed": seed,
        "lr": lr,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "train_seconds": round(time.time() - t0, 1),
        **{k: best[k] for k in ("val_category_f1", "val_needs_info_f1", "threshold", "val_score")},
    }
    (run_dir / "done.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del model, opt
    torch.cuda.empty_cache()
    return summary


def load_trained(name: str, ckpt: Path, device, attn: str | None = None):
    base_id = CANDIDATES[name][0]
    path = fetch_model(base_id)
    tok = AutoTokenizer.from_pretrained(path)
    model = Classifier(path, attn=attn)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval(), tok, path


def predict_texts(name: str, ckpt: Path, texts_by_key: dict[str, list[str]]) -> dict:
    """fp32 probabilities from a checkpoint for several named text lists."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_trained(name, ckpt, device)
    out = {}
    for key, texts in texts_by_key.items():
        c, n = predict(model, encode(tok, texts), tok.pad_token_id, device)
        out[key] = {"category": c, "needs_info": n}
    del model
    torch.cuda.empty_cache()
    return out


def bench(name: str, texts: list[str], steps: int = 30) -> dict:
    """Seconds per training step and peak VRAM at batch 32 / max_len 256 on real texts."""
    device = torch.device("cuda")
    set_seed(0)
    path = fetch_model(CANDIDATES[name][0])
    tok = AutoTokenizer.from_pretrained(path)
    model = Classifier(path).to(device).train()
    ids = encode(tok, texts)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    times = []
    rng = np.random.default_rng(0)
    for i in range(steps):
        sel = rng.choice(len(ids), BATCH, replace=False)
        x, m, y_c, _ = collate([(ids[j], j % 4, 0) for j in sel], tok.pad_token_id)
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            lc, ln = model(x.to(device), m.to(device))
        loss = nn.functional.cross_entropy(lc.float(), y_c.to(device))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(time.time() - t)
    return {
        "model": name,
        "s_per_step": round(float(np.mean(times)), 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


# ---------------------------------------------------------------- ONNX export


class _Export(nn.Module):
    def __init__(self, model: Classifier):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


def export_onnx(name: str, ckpt: Path, out_dir: Path) -> float:
    """Write model.onnx (fp32, opset 17, no external data) and the tokenizer files."""
    import onnx

    model, tok, _ = load_trained(name, ckpt, torch.device("cpu"), attn="eager")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model.onnx"
    ids = torch.randint(5, 1000, (3, 24))
    mask = torch.ones_like(ids)
    mask[0, 15:] = 0
    with torch.no_grad():
        torch.onnx.export(
            _Export(model).eval(),
            (ids, mask),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["category_logits", "needs_info_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "category_logits": {0: "batch"},
                "needs_info_logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    proto = onnx.load(str(fp32), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in proto.graph.initializer):
        raise RuntimeError("ONNX file uses external data")
    tok.save_pretrained(out_dir)
    for extra in ("vocab.txt", "vocab.json", "merges.txt"):
        (out_dir / extra).unlink(missing_ok=True)
    return round(os.path.getsize(fp32) / 1e6, 1)


# Dynamic-quantization settings compared on validation. Findings on this model: per-channel
# weights with the full int8 range collapse the category head to one class (consistent with
# 8-bit saturation in the u8 x s8 CPU kernels); reduce_range or unsigned weights avoid it.
QUANT_CONFIGS = {
    "qint8": {},
    "qint8_reduce": {"reduce_range": True},
    "qint8_perchannel": {"per_channel": True},
    "qint8_perchannel_reduce": {"per_channel": True, "reduce_range": True},
    "quint8": {"weight_type": "QUInt8"},
    "quint8_reduce": {"weight_type": "QUInt8", "reduce_range": True},
    "quint8_perchannel": {"weight_type": "QUInt8", "per_channel": True},
    "quint8_perchannel_reduce": {
        "weight_type": "QUInt8",
        "per_channel": True,
        "reduce_range": True,
    },
    "qint8_matmul_only": {"op_types_to_quantize": ["MatMul"]},
    "qint8_reduce_matmul_only": {"reduce_range": True, "op_types_to_quantize": ["MatMul"]},
    "quint8_reduce_matmul_only": {
        "weight_type": "QUInt8",
        "reduce_range": True,
        "op_types_to_quantize": ["MatMul"],
    },
}


def quantize(fp32: Path, out: Path, name: str) -> float:
    """Quantize with one named config from QUANT_CONFIGS; returns the file size in MB."""
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    kw = dict(QUANT_CONFIGS[name])
    kw["weight_type"] = QuantType[kw.get("weight_type", "QInt8")]
    logging.getLogger().setLevel(logging.ERROR)  # silences the pre-processing hint
    quantize_dynamic(str(fp32), str(out), **kw)
    return round(os.path.getsize(out) / 1e6, 1)


def _session(path: Path, threads: int | None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(path: Path, ids: list[list[int]], pad_id: int, batch: int = 32) -> tuple:
    sess = _session(path, None)
    order = np.argsort([len(x) for x in ids])
    cat = np.zeros((len(ids), len(CATEGORIES)), dtype=np.float32)
    ni = np.zeros((len(ids), len(NEEDS_INFO)), dtype=np.float32)
    for start in range(0, len(ids), batch):
        sel = order[start : start + batch]
        x, m, _, _ = collate([(ids[i], 0, 0) for i in sel], pad_id)
        c, n = sess.run(None, {"input_ids": x.numpy(), "attention_mask": m.numpy()})
        cat[sel], ni[sel] = c, n
    return cat, ni


def _feed(enc) -> dict:
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def latency(path: Path, tok, texts: list[str], max_len: int = MAX_LEN) -> dict:
    """Batch 1, one intra-op thread, tokenizer plus onnxruntime, milliseconds per issue."""
    sess = _session(path, 1)
    for t in texts[:5]:  # warm-up
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        sess.run(None, _feed(enc))
    times = []
    for t in texts:
        t0 = time.perf_counter()
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        sess.run(None, _feed(enc))
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "n": len(times),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
    }
