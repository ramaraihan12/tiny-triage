# M2: fine-tuned encoder vs TF-IDF baseline (repo-disjoint split)

## Candidates (seed 0, validation only)

| model | lr | best epoch | val category F1 | val needs_info F1 | threshold | val score |
|---|---|---|---|---|---|---|
| electra-small | 0.0001 | 1 | 0.7121 | 0.6479 | 0.46 | 0.68 |
| minilm-l12 | 5e-05 | 1 | 0.7456 | 0.611 | 0.25 | 0.6783 |
| granite-30m | 5e-05 | 1 | 0.7431 | 0.52 | 0.32 | 0.6315 |

Chosen on validation: **electra-small** (google/electra-small-discriminator, Apache-2.0). Val score = mean of category macro-F1 and needs_info macro-F1 (needs_info at its val-tuned threshold). Test was only evaluated for the chosen model's 3 seeds.

## Test, split_repo

|  | category macro-F1 | needs_info macro-F1 (0.5) | needs_info macro-F1 (val-tuned) |
|---|---|---|---|
| TF-IDF + LR | 0.6309 | 0.524 | 0.5494 (C 0.5, thr 0.44) |
| electra-small, mean ± std of 3 seeds | 0.7000 ± 0.0121 | 0.5204 ± 0.0190 | 0.4982 ± 0.0185 |
| electra-small, exported seed 2 | 0.7136 | 0.5244 | 0.4769 (thr 0.75) |

Std is the sample standard deviation over seeds. The baseline row with the default threshold is the M1 baseline rerun; reproduces M1 numbers (category 0.6309, needs_info 0.524): {'category': True, 'needs_info': True}. The tuned baseline row picks C and the threshold together on val, like the model.

Per seed:

| seed | val score | threshold | test category | test needs_info (0.5) | test needs_info (tuned) |
|---|---|---|---|---|---|
| 0 | 0.68 | 0.46 | 0.6905 | 0.4997 | 0.5079 |
| 1 | 0.7015 | 0.68 | 0.6958 | 0.5372 | 0.5099 |
| 2 | 0.7046 | 0.75 | 0.7136 | 0.5244 | 0.4769 |

## Paired bootstrap (exported seed minus baseline)

Issue-level bootstrap, 10000 resamples, fixed seed. Only issues are resampled, not repos: the test set has few repos, so repo-to-repo variance is not captured and these intervals are too narrow for a new repo.

- category: macro-F1 difference model - baseline = +0.0827, 95% CI [+0.0665, +0.0990]: better (CI above 0).
- needs_info (both with val-tuned threshold): macro-F1 difference model - baseline = -0.0724, 95% CI [-0.0967, -0.0480]: worse (CI below 0).
- needs_info (baseline at 0.5): macro-F1 difference model - baseline = -0.0471, 95% CI [-0.0698, -0.0247]: worse (CI below 0).

## Per test repo (macro-F1 over classes present in that repo)

| repo | n | category base | category model | n labeled | needs_info base (0.5) | needs_info base (tuned) | needs_info model (tuned) |
|---|---|---|---|---|---|---|---|
| cli/cli | 870 | 0.4885 | 0.527 | 870 | 0.4803 | 0.5437 | 0.4122 |
| elastic/elasticsearch | 392 | 0.588 | 0.5903 | 392 | 0.602 | 0.7662 | 0.5451 |
| encode/httpx | 275 | 0.6965 | 0.694 | 0 | n/a | n/a | n/a |
| microsoft/vscode | 1200 | 0.4404 | 0.5602 | 1200 | 0.5344 | 0.5108 | 0.5147 |
| pallets/flask | 201 | 0.3475 | 0.3865 | 0 | n/a | n/a | n/a |
| pydantic/pydantic | 1307 | 0.6152 | 0.6806 | 0 | n/a | n/a | n/a |

## Deploy-time cleaning

Template lines frozen from the 22 train repos only (549 lines), instead of statistics over each repo's own issues. Rows marked 'at inference only' keep the models fitted on M1 text and only rebuild the val/test text. The encoder was trained on M1 text, so for it this is a train/inference cleaning mismatch; the baseline refit row retrains TF-IDF + LR on train text rebuilt with the frozen list as well.

|  | text | category | needs_info (0.5) | needs_info (tuned) |
|---|---|---|---|---|
| baseline, test | M1 text | 0.6309 | 0.524 | 0.5494 |
| baseline, test | frozen list at inference only | 0.6522 | 0.5163 | 0.5401 |
| baseline refit, test | frozen list in train and test | 0.6505 | 0.5166 | 0.5569 |
| model, test | M1 text | 0.7136 | 0.5244 | 0.4769 |
| model, test | frozen list at inference only | 0.7242 | 0.505 | 0.4772 |
| baseline, val | M1 text | 0.6562 | 0.5436 | 0.5549 |
| baseline, val | frozen list at inference only | 0.6809 | 0.5553 | 0.5595 |
| model, val | frozen list at inference only | 0.6937 | 0.6008 | 0.7126 |

## Export, quantization, fidelity, latency

int8 chosen on validation only: **quint8_perchannel** ({'weight_type': 'QUInt8', 'per_channel': True}). Rule: mean of val argmax agreement (category), needs_info argmax agreement and needs_info agreement at the threshold, vs fp32 ONNX; size <= 50.0 MB. Per-channel signed int8 weights with the full range collapsed the category head to one class on this model (val agreement 0.14); reduce_range or unsigned weights avoid it. The configs below were compared on val.

| config | size MB | val category agreement | val needs_info argmax | val needs_info at thr |
|---|---|---|---|---|
| qint8 | 14.1 | 0.9708 | 0.978 | 0.9934 |
| qint8_reduce | 14.1 | 0.9396 | 0.9547 | 0.9892 |
| qint8_perchannel | 14.2 | 0.1437 | 0.8157 | 0.9359 |
| qint8_perchannel_reduce | 14.2 | 0.9848 | 0.9886 | 0.9914 |
| quint8 | 14.1 | 0.9829 | 0.9793 | 0.9901 |
| quint8_reduce | 14.1 | 0.9629 | 0.9695 | 0.9899 |
| quint8_perchannel | 14.2 | 0.987 | 0.9877 | 0.9941 |
| quint8_perchannel_reduce | 14.2 | 0.9859 | 0.987 | 0.9914 |
| qint8_matmul_only | 26 | 0.9703 | 0.9804 | 0.9943 |
| qint8_reduce_matmul_only | 26 | 0.94 | 0.9558 | 0.9899 |
| quint8_reduce_matmul_only | 26 | 0.9627 | 0.9708 | 0.9917 |

Gate (val category agreement >= 0.98, test |macro-F1 delta| <= 0.01 per target, size <= 50 MB): PASSED: {'val_category_agreement_ge_0.98': True, 'test_abs_delta_category_le_0.01': True, 'test_abs_delta_needs_info_tuned_le_0.01': True, 'size_le_50mb': True, 'passed': True}

| file | size MB |
|---|---|
| model.onnx (fp32) | 54.2 |
| model.int8.onnx | 14.2 |

| test comparison | category argmax agreement | needs_info argmax agreement | needs_info at threshold |
|---|---|---|---|
| int8 vs ONNX fp32 | 0.9802 | 0.987 | 0.996 |
| ONNX fp32 vs PyTorch | 1 | 1 | 1 |
| int8 vs PyTorch | 0.9802 | 0.987 | 0.996 |

| macro-F1 on full test | PyTorch fp32 | ONNX int8 | delta |
|---|---|---|---|
| category | 0.7136 | 0.7128 | -0.0008 |
| needs_info_default | 0.5244 | 0.5261 | 0.0017 |
| needs_info_tuned | 0.4769 | 0.4804 | 0.0035 |

| CPU latency (batch 1, 1 thread, tokenizer + onnxruntime) | n | p50 ms | p95 ms |
|---|---|---|---|
| fp32 | 200 | 58.2 | 63.8 |
| int8 | 200 | 46.4 | 51.6 |

## Notes and caveats

- Validation scores of electra-small (0.68) and minilm-l12 (0.6783) are practically tied: minilm-l12 is better on category, electra-small on needs_info. The pick between them is not a strong result.
- Best epoch per run: [1, 2]; epochs actually run: [2, 3] (patience 1). No run improved after epoch 2, so the 4-epoch budget was never used and longer schedules or other learning rates were not explored.
- The val-tuned needs_info thresholds vary a lot across seeds ([0.46, 0.68, 0.75]) and do not transfer to the test repos: for the exported seed the tuned threshold gives 0.4769 on test, lower than the 0.5244 at 0.5.
- The encoder was trained on M1 text (template lines removed with statistics over each repo's own issues), but at deploy time only the frozen list can be applied: a train/inference cleaning mismatch. The deploy-time rows show its effect; the baseline refit row shows what training on the same cleaning would give.
- artifacts/boilerplate.json contains some non-template lines, so it has to be filtered before it ships in the M3 demo. It is not published.

