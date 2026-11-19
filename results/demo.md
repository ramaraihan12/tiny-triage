# M3: cleaning for the browser demo

The M1 text cleaning removes template lines using statistics over each repo's own issues, which a pasted issue in the demo cannot do. The alternative is a frozen list of template lines built from the 22 train repos. Three variants, all scored with the exported int8 model at batch 1 and with the TF-IDF baseline (fitted on the M1 train text, so the cleaning here only changes what it is asked to score):

- none: `clean_body` only, no template list
- full: all 549 frozen lines (not published, it has non-template lines)
- filtered: 232 lines kept, headings, field labels and template prompts only

Rule: highest validation score for the int8 model among the variants that can ship (none, filtered; the full list holds non-template lines): mean of category macro-F1 and needs_info macro-F1 at the labels.json threshold 0.75. A gap under 0.002 counts as a tie and goes to none, which needs no list. Chosen on validation: **none**. needs_info for the model uses the threshold the demo uses; the baseline uses 0.5. Macro-F1, repo-disjoint split; val n=4552 (needs_info 3421), test n=4245 (needs_info 2462).

## val

| variant | model category | model needs_info (demo thr) | model needs_info (0.5) | baseline category | baseline needs_info (0.5) |
|---|---|---|---|---|---|
| none | 0.6932 | 0.7124 | 0.6077 | 0.6855 | 0.5617 |
| full | 0.6961 | 0.7097 | 0.6027 | 0.6809 | 0.5553 |
| filtered | 0.6941 | 0.7098 | 0.609 | 0.6794 | 0.5601 |

## test

| variant | model category | model needs_info (demo thr) | model needs_info (0.5) | baseline category | baseline needs_info (0.5) |
|---|---|---|---|---|---|
| none | 0.73 | 0.5119 | 0.5301 | 0.6514 | 0.5387 |
| full | 0.7251 | 0.4803 | 0.5024 | 0.6522 | 0.5163 |
| filtered | 0.728 | 0.4807 | 0.4991 | 0.6494 | 0.5116 |

Validation score of the model (mean of its two macro-F1s): none 0.7028, full 0.7029, filtered 0.702.

The three variants are within 0.001 of each other on validation, so the choice is close to a coin flip. Test is shown for all three for completeness; only validation was used to choose.

The demo therefore applies no template list: it runs `clean_body` and cuts the text, nothing else.
