# WAKER: Weight Attribution based on Kolmogorov-Arnold Networks for Entity Resolution

Code accompanying the paper:

> Lamisse F. Bouabdelli, Fatma Abdelhedi, Slimane Hammoudi, Allel Hadjali.
> **WAKER: Weight Attribution based on Kolmogorov-Arnold Networks for Entity Resolution.**
> 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026), Procedia Computer Science.

## Overview

Entity Resolution (ER) is the task of identifying records that refer to the
same real-world entity across heterogeneous, independently designed
datasets. Not all attributes are equally discriminative for this decision,
and treating them uniformly leads to suboptimal matching performance.

This repository implements an ER pipeline that computes attribute-level
similarities and compares three ways of aggregating them into a
match / non-match decision:

- **Uniform weighting** every attribute contributes equally to the score.
- **Expert-defined (manual) weighting** importance scores assigned from
  domain knowledge.
- **KAN-learned weighting** a Kolmogorov-Arnold Network (KAN) is trained
  on labeled pairs, and per-attribute importance is derived from it via a
  leave-one-out analysis; those weights are then used to compute the score.


## The four experimental setups

| # | Setup | Score used |
|---|-------|------------|
| 0 | No KAN, no weights | `mean(sim_i)` over all attributes |
| 1 | Manual weights | `sum(sim_i * w_i) / sum(w_i)`, expert-defined `w_i` |
| 2 | KAN-deduced weights | same weighted formula, `w_i` derived from KAN via leave-one-out importance |
| 3 | KAN-only classifier | the KAN's predicted probability directly |

For setups (0), (1) and (2), the decision threshold is tuned on the
validation set for the resulting score. For setup (3), the threshold is
tuned on the validation set for the KAN probability. All setups are then
evaluated on the test set using the threshold selected on validation.



## Dataset format

The script expects a folder containing five CSV files (the classic
[DeepMatcher](https://github.com/anhaidgroup/deepmatcher) / Magellan
benchmark layout, e.g. iTunes-Amazon):

- `tableA.csv`, `tableB.csv` — one row per record, with at least these
  columns: `Song_Name`, `Artist_Name`, `Album_Name`, `Genre`, `Price`,
  `CopyRight`, `Time`, `Released`.
- `train.csv`, `valid.csv`, `test.csv`  pair labels, with columns
  `ltable_id`, `rtable_id` (row indices into `tableA` / `tableB`) and
  `label` (`0`/`1`).

Adjust `attribute_list` / `TEXT_ATTRS` / `NUM_ATTRS` in
`kan_er_experiment.py` to match another dataset's schema.

## Requirements

Python 3.9+ is recommended.

Install the required dependencies: `pip install -r requirements.txt`

---

## Repository structure

```
.
├── WAKER_experiment.py   # end-to-end pipeline (preprocessing, embeddings,
│                           #   KAN training, weight extraction, 4 evaluations)
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bouabdelli2026waker,
  title     = {WAKER: Weight Attribution based on Kolmogorov-Arnold Networks for Entity Resolution},
  author    = {Bouabdelli, Lamisse F. and Abdelhedi, Fatma and Hammoudi, Slimane and Hadjali, Allel},
  booktitle = {30th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES 2026)},
  series    = {Procedia Computer Science},
  year      = {2026},
  publisher = {Elsevier}
}
```
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

This work was carried out with the support of the [ANRT](https://www.anrt.asso.fr) through the CIFRE (Industrial Agreements for Training through Research) program.

The project is supported by:

* [Trimane](https://trimane.fr/en/)
* [LIAS laboratory](https://www.lias-lab.fr/) (ISAE-ENSMA)
