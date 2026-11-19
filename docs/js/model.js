// Loads the tokenizer (transformers.js) and the int8 ONNX file (onnxruntime-web, wasm, one thread)
// and classifies one text at a time. Both libraries come in through the import map in index.html,
// so there is a single copy of onnxruntime-web.
import { AutoTokenizer, env as hfEnv } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";

export const ORT_VERSION = "1.22.0-dev.20250409-89f8206ba4";
const WASM_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

hfEnv.allowRemoteModels = false; // the tokenizer comes from docs/model/, never from the Hub
hfEnv.allowLocalModels = true;
hfEnv.useBrowserCache = false;
ort.env.wasm.numThreads = 1; // GitHub Pages cannot send the COOP/COEP headers threads need
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = WASM_DIR;

const softmax = (x) => {
  const m = Math.max(...x);
  const e = Array.from(x, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

export class Classifier {
  constructor(tokenizer, session, labels) {
    this.tokenizer = tokenizer;
    this.session = session;
    this.labels = labels;
    // an empty text encodes to just [CLS] [SEP]
    [this.cls, this.sep] = Array.from(tokenizer("").input_ids.data, Number);
  }

  // modelUrl: folder holding model.int8.onnx, tokenizer.json, labels.json (same origin)
  static async load(modelUrl, onProgress) {
    const base = new URL(modelUrl, document.baseURI);
    const t0 = performance.now();
    hfEnv.localModelPath = new URL("../", base).href;
    const folder = base.pathname.replace(/\/+$/, "").split("/").pop();
    const [labels, tokenizer, bytes] = await Promise.all([
      fetch(new URL("labels.json", base)).then((r) => r.json()),
      AutoTokenizer.from_pretrained(folder),
      fetchWithProgress(new URL("model.int8.onnx", base), onProgress),
    ]);
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const clf = new Classifier(tokenizer, session, labels);
    clf.loadMs = performance.now() - t0;
    return clf;
  }

  // [CLS] tokens [SEP], cut to max_len like Python's truncation=True. transformers.js adds the
  // special tokens first and then cuts the tail, which would drop the closing [SEP] on long
  // issues, so the special tokens are added here instead.
  tokenize(text) {
    // transformers.js lowercases a word-final capital sigma to the final form, Python to the
    // plain one, so map it first
    const enc = this.tokenizer(text.replaceAll("Σ", "σ"), { add_special_tokens: false });
    const ids = Array.from(enc.input_ids.data, Number).slice(0, this.labels.max_len - 2);
    return [this.cls, ...ids, this.sep];
  }

  async classifyIds(ids) {
    const n = ids.length;
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
    };
    const out = await this.session.run(feeds);
    const cat = softmax(out.category_logits.data);
    const ni = softmax(out.needs_info_logits.data);
    const yes = ni[this.labels.needs_info.indexOf(this.labels.needs_info_positive_class)];
    return {
      tokens: n,
      categoryProbs: Object.fromEntries(this.labels.categories.map((c, i) => [c, cat[i]])),
      category: this.labels.categories[cat.indexOf(Math.max(...cat))],
      needsInfoProb: yes,
      needsInfo: yes >= this.labels.needs_info_threshold ? "yes" : "no",
    };
  }

  async classify(text) {
    const t0 = performance.now();
    const ids = this.tokenize(text);
    const res = await this.classifyIds(ids);
    res.ms = performance.now() - t0;
    return res;
  }
}
