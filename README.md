# tiny-triage

A tiny issue classifier trained on what maintainers of real projects actually did: which label they put on an issue, and whether they asked for more information. Everything here is milestone M1: the dataset and an honest baseline. There is no neural model yet.

Small sibling of [issue-triage-action](https://github.com/ramaraihan12/issue-triage-action), the GitHub Action that triages new issues with an LLM.

## Why

Triage costs maintainer time. The routine part is mostly "is this a bug, a feature request, a question or a docs problem, and is there enough to act on?". In issue-triage-action I answer that with an LLM call per issue, which is slow and costs money (or free-tier quota) every time. I want to know whether a model small enough to run in the browser can do the routine part well enough to be useful, with the LLM kept for the hard cases.

The plan for the whole project:

1. M1 (this repo now): dataset from public repos, plus a plain TF-IDF baseline so I know what "better than nothing" means.
2. M2: fine-tune a small Apache-2.0 encoder locally on the same split and compare it with the baseline.
3. M3: export it and run it in the browser as a demo.

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

## Next

Fine-tune a small Apache-2.0 encoder locally on the same repo-disjoint split, compare it with these numbers, and only then think about a browser demo. If it does not clearly beat TF-IDF on unseen repos, I will say so here.

## Development

```bash
uv sync
uv run ruff check . && uv run pytest
```

Tests use no network. CI runs the same two commands.

MIT licence. The code and the manifest are mine; issue text belongs to its authors and stays out of this repo.
