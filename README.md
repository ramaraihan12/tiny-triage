# tiny-triage

A tiny issue classifier trained on what maintainers of real projects actually did: which label they put on an issue, and whether they asked for more information. Milestone M1 is the dataset and an honest TF-IDF baseline; milestone M2 fine-tunes a small encoder and compares it with that baseline. Milestone M3 is a browser demo of the fine-tuned model.

**Live demo: https://ramaraihan.is-a.dev/tiny-triage/** (the model runs in your browser, a one-time download of about 17 MB).

Small sibling of [issue-triage-action](https://github.com/ramaraihan12/issue-triage-action), the GitHub Action that triages new issues with an LLM.

## Why

Triage costs maintainer time. The routine part is mostly "is this a bug, a feature request, a question or a docs problem, and is there enough to act on?". In issue-triage-action I answer that with an LLM call per issue, which is slow and costs money (or free-tier quota) every time. I want to know whether a model small enough to run in the browser can do the routine part well enough to be useful, with the LLM kept for the hard cases.

The plan for the whole project:

1. M1 (done): dataset from public repos, plus a plain TF-IDF baseline so I know what "better than nothing" means.
2. M2 (done, below): fine-tune a small Apache-2.0 encoder locally on the same split, compare it with the baseline, export it to ONNX.
3. M3 (done, below): run it in the browser as a demo, and put one LLM next to it on 50 issues.

## Labels

Two outputs per issue:

- `category`: `bug`, `feature`, `question` or `docs`, the same four classes issue-triage-action uses.
- `needs_info`: `yes` if maintainers asked for more information or a reproduction, `no` otherwise.

Every repo names its labels differently (`Type: Bug`, `kind/bug`, `C-bug`, `>bug`, `issue::bug`). [data/label_map.yml](data/label_map.yml) maps each repo's labels onto the scheme. I wrote it by reading each repo's real label list. Rules:

- An issue is kept only if its labels hit exactly one category class. Bug plus feature, or bug plus docs, is skipped as ambiguous. Issues with no category label are skipped.
- Pull requests are skipped.
- `needs_info` is only defined for repos that have such a label (20 of 33). For the other repos it is left empty, not set to `no`, because a repo without the label could never have said `yes`.

## Data

33 public repos, picked for popularity, active triage and consistent labels: JS/TS (react, vue, eslint, axios, material-ui, svelte, TypeScript, vscode), Python (pandas, flask, scikit-learn, pytest, poetry, pydantic, httpx, numpy), Go (kubernetes, hugo, gh cli, traefik, prometheus), Rust (cargo, tokio), Java (spring-boot, elasticsearch), PHP (symfony), devtools (docker compose, ruff, deno), data/ML (transformers, airflow, polars, langchain).

I fetched at most about 400 issues per class per repo, newest first, with the authenticated `gh` CLI (one label per query, no search API). That is 29,054 issues:

| class | issues |
|---|---|
| bug | 10,218 |
| feature | 9,897 |
| question | 5,988 |
| docs | 2,951 |

`needs_info`: 2,063 yes, 16,898 no, 10,093 undefined.

Text is title plus body. I strip HTML comments, replace code blocks longer than 40 lines with a placeholder, drop checklist lines, replace URLs and @mentions, remove lines that repeat across many issues of one repo (template boilerplate), and cut the body to 2,000 characters.

### Raw text is not in this repo

Issue text is other people's writing, so I don't republish it. The repo has the collection script, the label map, and [data/manifest.csv](data/manifest.csv): repo, issue number, category, needs_info and which split. No text, no usernames (the collector drops the author too). To rebuild the dataset yourself:

```bash
uv sync
uv run tiny-triage collect --raw ./raw --manifest data/manifest.csv   # needs an authenticated gh, ~600 requests
uv run tiny-triage build --raw ./raw --out ./data-local
uv run tiny-triage baseline --data ./data-local
```

Leave out `--manifest` to collect fresh issues instead of the ones in the manifest. A rebuild can differ a little: maintainers relabel and edit issues after I collected (2026-11-18).

## Why the split is by repo

If I split issues at random, the same repo appears in train and test. A model can then pick up each project's own template and vocabulary, and the score says little about a repo it has never seen. The use case is a maintainer dropping this into a new repo, so the number that matters is the one on repos the model has not seen.

The honest split puts whole repos in one side only (22 train, 5 val, 6 test), chosen by a seeded search so that every category, and both `needs_info` values, appear in each side with a reasonable count:

- val: ruff, eslint, transformers, langchain, spring-boot
- test: gh cli, elasticsearch, httpx, vscode, flask, pydantic

For contrast I also report an in-repo split: random 80/10/10 over all issues. It is optimistic and I label it that way everywhere.

| split | train | val | test |
|---|---|---|---|
| repo-disjoint | 20,257 | 4,552 | 4,245 |
| in-repo random | 23,272 | 2,871 | 2,911 |

## Baseline

TF-IDF on words (1 to 2 grams) and characters (3 to 5 grams) into a class-weighted logistic regression, one model per output. I picked the regularisation strength (C in 0.5, 2, 8) on the validation set only. Majority class means always predicting the most common training class. All numbers are on the test set; the full per-class tables and confusion matrices are in [results/baseline.md](results/baseline.md) and [results/baseline.json](results/baseline.json).

| split | output | test n | majority macro-F1 | TF-IDF + LR macro-F1 |
|---|---|---|---|---|
| repo-disjoint (honest) | category | 4,245 | 0.129 | **0.631** |
| repo-disjoint (honest) | needs_info | 2,462 | 0.460 | **0.524** |
| in-repo random (optimistic) | category | 2,911 | 0.130 | 0.793 |
| in-repo random (optimistic) | needs_info | 1,891 | 0.474 | 0.738 |

Per class on the repo-disjoint test set (category):

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| bug | 0.60 | 0.71 | 0.65 | 1,475 |
| feature | 0.74 | 0.80 | 0.77 | 1,484 |
| question | 0.58 | 0.36 | 0.44 | 854 |
| docs | 0.74 | 0.60 | 0.66 | 432 |

What I take from it:

- Category carries over to unseen repos reasonably well, and the gap to the in-repo number (0.79) is the repo-specific vocabulary a random split lets leak in.
- `question` is the weak class. Half of them are predicted as `bug`, which is what a stack trace and "it doesn't work" look like in both.
- `needs_info` barely carries over: the `yes` class has F1 0.17 on unseen repos, against 0.53 in-repo. Whether a maintainer asks for more info depends a lot on the project's habits. The 0.52 macro-F1 is mostly the `no` class.

## M2: fine-tuned encoder

I fine-tuned three small Apache-2.0 encoders on the same repo-disjoint split: `google/electra-small-discriminator` (14M parameters), `sentence-transformers/all-MiniLM-L12-v2` (33M) and `ibm-granite/granite-embedding-30m-english` (30M). Each is the encoder, mean pooling over the tokens, and two linear heads (category and needs_info), trained together with class-balanced cross-entropy; needs_info is left out of the loss where the label is undefined. Plain PyTorch, fp16, batch 32, 256 tokens, AdamW, a fixed learning rate per model (1e-4 for electra-small, 5e-5 for the others, not tuned), at most 4 epochs with early stopping on validation.

Choosing: one seed per candidate, scored on the validation repos only (mean of the two macro-F1 scores, with the needs_info threshold on P(yes) tuned on validation). electra-small won with 0.680, but MiniLM was at 0.678, so that pick is a coin flip: MiniLM is better on category and electra on needs_info. Test was never looked at for the losers. The winner was then trained with two more seeds (0, 1, 2 in all) and the exported model is the seed with the best validation score (seed 2). No run improved after epoch 2.

Test, repo-disjoint split (macro-F1; the first row is the M1 baseline again, reproduced exactly):

| | category | needs_info, threshold 0.5 | needs_info, val-tuned threshold |
|---|---|---|---|
| TF-IDF + LR | 0.631 | 0.524 | 0.549 |
| electra-small, mean ± std over 3 seeds | **0.700 ± 0.012** | 0.520 ± 0.019 | 0.498 ± 0.019 |
| electra-small, exported seed | **0.714** | 0.524 | 0.477 |

The plain summary: it beats TF-IDF on category and does not on needs_info. Paired issue-level bootstrap (10,000 resamples) for the exported seed minus the baseline: category +0.083 (95% CI +0.067 to +0.099); needs_info -0.072 (CI -0.097 to -0.048) when both use a val-tuned threshold, and -0.047 (CI -0.070 to -0.025) against the baseline at 0.5. These intervals resample issues only. The test set is six repos, so repo-to-repo variance is not in them, and they are too tight for a new repo.

The val-tuned needs_info threshold does not transfer: it was 0.46, 0.68 and 0.75 for the three seeds, and for the exported seed it is worse on test than 0.5. Category is better than the baseline in four of the six test repos (cli, vscode, pydantic, flask) and about equal in the other two. needs_info is worse in cli and elasticsearch and level in vscode. Per-repo tables are in [results/finetune.md](results/finetune.md).

Deploy-time cleaning: M1 removes template lines using statistics over each repo's own issues, which a pasted issue in a demo cannot do. I froze a list of template lines from the train repos only and rebuilt the validation and test text with it. Same models, only the inference text changes: category 0.714 to 0.724 and needs_info 0.524 to 0.505 for the encoder, 0.631 to 0.652 and 0.524 to 0.516 for TF-IDF. The encoder was trained on M1 text, so this is a train/inference mismatch; retraining TF-IDF on text cleaned the same way gives 0.651 and 0.517. So the frozen list costs nothing on category here. That list still contains some non-template lines and has to be filtered before it ships, so I have not published it.

Export: the exported model is one ONNX file (opset 17, inputs `input_ids` and `attention_mask`, outputs `category_logits` and `needs_info_logits`), 54 MB in fp32 and 14 MB after int8 dynamic quantization. Quantization needed care. Per-channel signed int8 weights collapsed the category head to one class (val agreement with fp32 of 0.14), so I compared 11 settings on validation and picked unsigned per-channel weights. The int8 model agrees with fp32 on 98.0% of test category predictions and 98.7% of needs_info predictions, and its macro-F1 differs by less than 0.004 on both outputs. On one CPU thread, tokenizer plus onnxruntime takes about 46 ms per issue (p95 52 ms) for int8, against 58 ms for fp32. That was measured with onnxruntime on a laptop CPU; M3 checks it in the browser. The int8 weights ship in `docs/model/` (M3).

Reproduce (needs a GPU with torch 2.2 and CUDA preinstalled; see [requirements-train.txt](requirements-train.txt)):

```bash
python -m tiny_triage.finetune all --data ./data-local --runs ./runs --artifacts ./artifacts --baseline-python path/to/uv-env/python
```

The baseline stage runs in the project's own environment because torch 2.2 needs numpy below 2, which shifts the logistic regression slightly (category 0.634 instead of 0.631). Results go to `results/finetune.json`, `results/finetune.md` and `results/test_predictions.csv` (per-issue predictions and probabilities, no text).

## Label noise I saw

Maintainer labels are real but not clean. Examples, by repo and issue number only:

- Near-empty issues labelled bug: cli/cli 11822, 11847, 11861.
- Labelled bug but reads like a how-do-I question: cli/cli 8972, pytest-dev/pytest 6464, python-poetry/poetry 6963 and 10156, tokio-rs/tokio 6844.
- Labelled feature but starts like an error report: eslint/eslint 18926, 19153, 19361; facebook/react 9832.
- Labelled docs but contains a traceback: astral-sh/ruff 18653, 18851, 19132.
- `needs_info = no` means "the label is absent", not "nothing was missing". Maintainers forget the label, or add it late.
- Some repos barely use their needs_info label (vue 5, langchain 3 and transformers 1 yes issues), so they add almost no signal there.

## Honest limits

- Labels come from large, active, mostly English-language open source projects. Small repos with other habits may differ a lot.
- I collected newest first, so the data leans to recent issues and to issue forms, which probably make the task easier than free text.
- Only issues that already have a category label are in the data, so the model never sees the unlabelled majority, which may be harder.
- Per-repo caps stop one project from dominating, but they also change the class mix compared with a real tracker.
- The test set is six repos. Three of them define `needs_info`, so that number is shaky.
- The baseline does not use the repo's templates, comments or author history. A real tool might.
- The LLM comparison is 50 issues (M3), which is an anecdote, not a benchmark.
- The encoder numbers are one training recipe with one fixed learning rate per model; a longer search could move them.

## M3: browser demo

[ramaraihan.is-a.dev/tiny-triage](https://ramaraihan.is-a.dev/tiny-triage/) is a static page served from `docs/` by GitHub Pages. You paste an issue, or a public GitHub issue URL (fetched from `api.github.com` without a token, so 60 requests an hour per IP), and press "Run in my browser". The int8 model, its tokenizer and the WebAssembly runtime download only after that click, about 17 MB once (14 MB is the model). The tokenizer is [transformers.js](https://github.com/huggingface/transformers.js) 3.8.1 and inference is onnxruntime-web 1.22.0-dev.20250409-89f8206ba4 (the version transformers.js depends on, so the page holds one copy of the runtime), wasm backend, one thread, because Pages cannot send the headers threads need. Both come from jsDelivr at pinned versions; the model files are served from the same origin. The page shows the load time and the time per issue: in headless Chromium on my laptop the median is 125 ms (p95 129 ms) per test issue, which averages 204 tokens, for tokenizer and model together (native onnxruntime took about 46 ms in M2). The aggregate browser numbers in this section are in [results/browser.json](results/browser.json); they were measured with a Playwright script that is not in the repo, and no text is in that file. Everything runs locally; the only request that leaves the page is the optional GitHub fetch.

`needs_info` is shown with an "experimental" badge: on unseen repos it is worse than the TF-IDF baseline, so the page says so next to the output.

**Same output as Python.** The cleaning (`clean_body`, `drop_lines`, `build_text`) is ported to `docs/js/clean.js`. Synthetic fixtures generated by the Python code (`tests/fixtures/clean_cases.json`, no real issue text) are checked by `node --test` in CI, and I compared the two on every raw issue of the val and test repos (8,797 issues): identical output for all of them. Token ids from transformers.js match Python `tokenizers` for 600 of 600 test issues, after two fixes in `docs/js/model.js`: transformers.js adds the special tokens before truncating, which drops the closing `[SEP]` on long issues, so the page truncates the tokens itself; and transformers.js lowercases a word-final capital sigma to the final form where Python gives the plain one, so the page maps it first (6 of 6 Greek test strings now match). Two cleaning differences are known and not handled, both far rarer than the 8,797-issue check can see: Python's case-insensitive match also treats U+0130 and U+0131 as `i`, and Python 3.11 uses Unicode 14 for `\w` while JavaScript uses newer tables. The int8 model in onnxruntime-web (Chromium) against native onnxruntime int8, batch 1, on random test issues: category 98.3% to 99.3% agreement over two draws of 600, needs_info 99.3% at 0.5 and 99.8% to 100% at the demo threshold. Against the fp32 model the WebAssembly int8 agrees on 99.0% to 99.5% of categories, a bit more than native int8 does (98.3% to 99.3%), so the difference comes from native int8 arithmetic rather than from the browser. I also tried re-quantizing with `reduce_range` (`quint8_perchannel_reduce`) because agreement was under 0.99 on one draw; on 600 validation issues it agreed less with fp32 than the shipped file (category 0.990 against 0.995), so I kept the M2 choice.

**Cleaning in the demo.** Template lines are normally found per repo, which a pasted issue cannot do. I compared three variants on validation with the int8 model at batch 1: no list, the full frozen list from M2, and a filtered list (232 of the 549 lines, only form headings, field labels and template prompts, nothing with paths, URLs, handles or long digit runs). All three are within 0.001 on the validation score, so it is close to a coin flip; the full list cannot be published, and a gap under 0.002 goes to the option that needs no list. The demo applies no list, only `clean_body` (which also strips comments, long code, links and checklist lines), so an issue that carries a form template reaches the model with the template headings in. Test for that choice: category 0.730 (baseline 0.651), needs_info 0.512 at the threshold 0.75 (baseline 0.539). Tables for all three variants, val and test, are in [results/demo.md](results/demo.md); the code is `src/tiny_triage/demo.py`.

**Llama 3.3 on 50 issues.** The page's table lists 50 test issues next to what the maintainers did, the baseline, the model, and Llama 3.3 70B (`@cf/meta/llama-3.3-70b-instruct-fp8-fast` on Cloudflare Workers AI). The sample was drawn before any call: seed 0, test split, 2 issues for each repo and category, the last places filled from random strata (`src/tiny_triage/llama.py`). The prompt is the system prompt of issue-triage-action with labels, duplicates and missing_info replaced by one yes/no `needs_info` field, JSON output as there, temperature 0; the input is the same cleaned text the model sees, sent as `NEW ISSUE` followed by that text (title, blank line, body) in one user message, not issue-triage-action's `Title:`/`Body:` layout, and with no label or candidate lists. The system prompt and this message are the only two changes to its prompt. Each cached answer stores the cleaning variant and the sha256 of its input, and building the page data checks both against the shipped cleaning. 50 calls, no retries (the sample was run twice: the first run used the same text but did not record the input hash, so I redid all 50 calls once with it, and every answer came out the same), about 770 neurons by the list price (the free allowance is 10,000 a day), 0 replies that could not be parsed.

| agreement with the maintainer, 50 issues | TF-IDF | tiny-triage | Llama 3.3 |
|---|---|---|---|
| category | 31 | 38 | 38 |
| needs_info (24 issues whose repo has the label) | 18 | 20 | 18 |

Always answering the most common maintainer class would match 17 of 50 on category (`bug`) and 21 of 24 on needs_info (`no`), which beats all three systems on needs_info; only 3 of the 24 labels are `yes`, so this table says little about needs_info. This is 50 issues, so a few issues either way are noise, so it says "the small model is in the same range as a 70B LLM on category here", not that either is better. The page shows only repo, number, title and labels or predictions; no issue body text is published. Llama 3.3 is licensed under the Llama 3.3 Community License, Copyright (c) Meta Platforms, Inc., and the page says "Built with Llama".

The files in `docs/model/` are Apache-2.0 (own `LICENSE` and a model card); the rest of the repo is MIT.

Rebuild the demo data (needs the local data and the artifacts from M2):

```bash
python -m tiny_triage.demo lists    --data ./data-local --runs ./runs --artifacts ./artifacts
python -m tiny_triage.demo baseline --data ./data-local --runs ./runs --artifacts ./artifacts   # project env
python -m tiny_triage.demo model    --data ./data-local --runs ./runs --artifacts ./artifacts   # training env
python -m tiny_triage.demo report   --data ./data-local --runs ./runs --artifacts ./artifacts
python -m tiny_triage.llama sample  --data ./data-local --runs ./runs --artifacts ./artifacts
CF_API_TOKEN=... CF_ACCOUNT_ID=... python -m tiny_triage.llama run --variant none --data ./data-local --runs ./runs --artifacts ./artifacts
python -m tiny_triage.demo site     --data ./data-local --runs ./runs --artifacts ./artifacts
```

## Next

The needs_info output does not beat TF-IDF yet. The open question is whether it is worth keeping, or whether a per-repo threshold or more `needs_info`-labelled repos would fix it.

## Development

```bash
uv sync
uv run ruff check . && uv run pytest
```

Tests use no network and do not need torch. CI runs the same two commands (plus `ruff format --check`) and `node --test "tests/*.test.mjs"` for the JavaScript port of the cleaning.

MIT licence, except `docs/model/` (the model files), which is Apache-2.0. The code and the manifest are mine; issue text belongs to its authors and stays out of this repo.
