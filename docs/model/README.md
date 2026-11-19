# tiny-triage model card

Issue classifier with two outputs: `category` (bug, feature, question, docs) and `needs_info` (would maintainers ask for more information). Used by the [browser demo](https://ramaraihan.is-a.dev/tiny-triage/). Code and write-up: https://github.com/ramaraihan12/tiny-triage

## Files

- `model.int8.onnx`: the model, one ONNX file (opset 17), dynamic int8 quantization, 14.2 MB. Inputs `input_ids` and `attention_mask`, outputs `category_logits` (4) and `needs_info_logits` (2).
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: the WordPiece tokenizer of the base model.
- `labels.json`: class order, maximum length 256, mean pooling, the needs_info threshold (0.75), quantization and cleaning settings.

## Model

Fine-tuned from [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator) (14M parameters, Apache-2.0): the encoder, mean pooling over tokens, and two linear heads trained together with class-balanced cross-entropy. One seed (seed 2 of three) was exported, chosen on validation. electra-small and MiniLM-L12 were within 0.002 on validation, so that pick between them is close to a coin flip.

## Data

Titles and bodies of 29,054 labelled issues from 33 public GitHub repos. The label is whatever the maintainers of that repo put on the issue, mapped onto the four categories; `needs_info` comes from the repo's own "needs more info" style labels and exists for 20 of the 33 repos. Training used 22 repos, validation 5, test 6; repos never cross splits. The text is not redistributed (see the repository README).

## Results

Test, six repos the model never saw, macro-F1 (PyTorch fp32, the exported seed): category 0.714 against 0.631 for a TF-IDF + logistic regression baseline; needs_info 0.477 at the shipped threshold of 0.75 against 0.549 for the baseline with its own tuned threshold (0.524 against 0.524 at the default 0.5). The int8 file changes macro-F1 by less than 0.004 and agrees with fp32 on 98.0% of category predictions. Per-repo numbers, bootstrap intervals and the other candidates are in `results/` of the repository.

## Limits

- `needs_info` is experimental: on unseen repos it is worse than the baseline, and its threshold, tuned on validation repos, does not transfer well.
- Category is weakest for `question`, which often looks like a bug report.
- Trained on large, active, mostly English-language projects that use issue forms; other kinds of repos may differ a lot. Only issues that already had a category label were used.
- The input is the title plus the first 2,000 characters of the body, cut to 256 tokens.
- Maintainer labels are noisy; the scores above measure agreement with them, not with a ground truth.

## License

The model weights and tokenizer files in this folder are derived from google/electra-small-discriminator and are released under the Apache License 2.0 (see `LICENSE`). The rest of the repository is MIT.
