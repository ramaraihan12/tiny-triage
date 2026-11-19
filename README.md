# tiny-triage

A tiny issue classifier trained on what maintainers of real projects actually did: which label they put on an issue, and whether they asked for more information. Milestone M1 is the dataset and an honest TF-IDF baseline; milestone M2 fine-tunes a small encoder and compares it with that baseline. The browser demo (M3) is not built yet.

Small sibling of [issue-triage-action](https://github.com/ramaraihan12/issue-triage-action), the GitHub Action that triages new issues with an LLM.

## Why

Triage costs maintainer time. The routine part is mostly "is this a bug, a feature request, a question or a docs problem, and is there enough to act on?". In issue-triage-action I answer that with an LLM call per issue, which is slow and costs money (or free-tier quota) every time. I want to know whether a model small enough to run in the browser can do the routine part well enough to be useful, with the LLM kept for the hard cases.

The plan for the whole project:

1. M1 (done): dataset from public repos, plus a plain TF-IDF baseline so I know what "better than nothing" means.
2. M2 (done, below): fine-tune a small Apache-2.0 encoder locally on the same split, compare it with the baseline, export it to ONNX.
3. M3: run it in the browser as a demo.

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

Export: the exported model is one ONNX file (opset 17, inputs `input_ids` and `attention_mask`, outputs `category_logits` and `needs_info_logits`), 54 MB in fp32 and 14 MB after int8 dynamic quantization. Quantization needed care. Per-channel signed int8 weights collapsed the category head to one class (val agreement with fp32 of 0.14), so I compared 11 settings on validation and picked unsigned per-channel weights. The int8 model agrees with fp32 on 98.0% of test category predictions and 98.7% of needs_info predictions, and its macro-F1 differs by less than 0.004 on both outputs. On one CPU thread, tokenizer plus onnxruntime takes about 46 ms per issue (p95 52 ms) for int8, against 58 ms for fp32. That was measured with onnxruntime on a laptop CPU; I still have to check it in the browser in M3. The weights are not in this repo yet.

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
- I have not compared with the LLM in issue-triage-action on this data yet.
- The encoder numbers are one training recipe with one fixed learning rate per model; a longer search could move them.

## Next

M3: run the exported model in the browser with onnxruntime-web, check that the int8 file behaves the same there, and decide how to handle the needs_info output, which does not beat TF-IDF yet.

## Development

```bash
uv sync
uv run ruff check . && uv run pytest
```

Tests use no network and do not need torch. CI runs the same two commands (plus `ruff format --check`).

MIT licence. The code and the manifest are mine; issue text belongs to its authors and stays out of this repo.
