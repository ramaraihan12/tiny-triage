### Repo-disjoint split (honest)

| output | test n | baseline (majority) macro-F1 | TF-IDF + LR macro-F1 | accuracy |
|---|---|---|---|---|
| category | 4245 | 0.1289 | 0.6309 | 0.6612 |
| needs_info | 2462 | 0.4602 | 0.524 | 0.7827 |

category, per class on test:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| bug | 0.5993 | 0.7119 | 0.6508 | 1475 |
| feature | 0.739 | 0.8032 | 0.7698 | 1484 |
| question | 0.5803 | 0.3595 | 0.444 | 854 |
| docs | 0.735 | 0.5972 | 0.659 | 432 |

confusion matrix (rows true, columns predicted):

| true \ pred | bug | feature | question | docs |
|---|---|---|---|---|
| bug | 1050 | 260 | 131 | 34 |
| feature | 194 | 1192 | 54 | 44 |
| question | 425 | 107 | 307 | 15 |
| docs | 83 | 54 | 37 | 258 |

needs_info, per class on test:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| no | 0.859 | 0.8914 | 0.8749 | 2099 |
| yes | 0.1972 | 0.1543 | 0.1731 | 363 |

confusion matrix (rows true, columns predicted):

| true \ pred | no | yes |
|---|---|---|
| no | 1871 | 228 |
| yes | 307 | 56 |

### In-repo random split (optimistic)

| output | test n | baseline (majority) macro-F1 | TF-IDF + LR macro-F1 | accuracy |
|---|---|---|---|---|
| category | 2911 | 0.1302 | 0.7928 | 0.8028 |
| needs_info | 1891 | 0.4741 | 0.7376 | 0.9069 |

category, per class on test:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| bug | 0.819 | 0.839 | 0.8289 | 1025 |
| feature | 0.8398 | 0.8126 | 0.826 | 987 |
| question | 0.717 | 0.7421 | 0.7294 | 601 |
| docs | 0.8063 | 0.7685 | 0.7869 | 298 |

confusion matrix (rows true, columns predicted):

| true \ pred | bug | feature | question | docs |
|---|---|---|---|---|
| bug | 860 | 62 | 81 | 22 |
| feature | 89 | 802 | 79 | 17 |
| question | 78 | 61 | 446 | 16 |
| docs | 23 | 30 | 16 | 229 |

needs_info, per class on test:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| no | 0.9484 | 0.9484 | 0.9484 | 1705 |
| yes | 0.5269 | 0.5269 | 0.5269 | 186 |

confusion matrix (rows true, columns predicted):

| true \ pred | no | yes |
|---|---|---|
| no | 1617 | 88 |
| yes | 88 | 98 |

