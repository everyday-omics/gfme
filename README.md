# GFME: Genomic Foundation Model Ensemble

An ensemble of three DNA foundation models (AlphaGenome, Evo2, GPN-MSA) for variant pathogenicity prediction. GFME scores both SNVs and indels across all variant consequence categories, with particular strength in non-missense variants where existing tools are scarce.

## Citing

If you use GFME in your work, please cite the following papers:

> Rabinowitz T, Baris Feldman H. An Ensemble of Genomic Foundation Models Accurately Predicts Variant Pathogenicity. 2026 (submitted).

<details>
<summary>BibTeX</summary>

```bibtex
@article{rabinowitz2026gfme,
  title={An Ensemble of Genomic Foundation Models Accurately Predicts Variant Pathogenicity},
  author={Rabinowitz, Tom and Baris Feldman, Hagit},
  journal={submitted},
  year={2026}
}
```
</details>

GFME is built on three foundation models. Please also cite:
- Avsec Ž, et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature* 649, 1206–1218 (2026). https://doi.org/10.1038/s41586-025-10014-0
- Brixi G, et al. Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361 (2026). https://doi.org/10.1038/s41586-026-10176-5
- Benegas G, et al. A DNA language model based on multispecies alignment predicts the effects of genome-wide variants. *Nature Biotechnology* 43, 1960–1965 (2025). https://doi.org/10.1038/s41587-024-02511-w

## Installation

```bash
git clone https://github.com/everyday-omics/gfme.git
cd gfme
pip install -e ".[test]"
make test          # unit tests and the golden-number regression, about a minute
```

On a fresh clone `make test` runs the unit tests and skips the golden-number checks, which read `results/` and the AlphaGenome matrices; both appear once the analysis has been run.

## Scoring variants

The tool runs as a hosted Space at https://everyday-omics-gfme.hf.space/. It scores novel variants through the AlphaGenome API and NVIDIA NIM (Evo2) with your own keys, and answers instantly for the ClinVar variants it holds pre-scored.

## Reproducing the analysis

```bash
make all           # the whole analysis, scripts 02 through 18
make figures       # figures
```

The reproduction path **starts at scored data**. Running the foundation models themselves — AlphaGenome through its API, Evo2-7B on GPU, GPN-MSA, Borzoi, SpliceAI, and the CADD/phyloP/GERP++/phastCons/DANN lookups — takes days and needs API keys, GPUs and annotation databases. How each was scored is described in the supplement of the article; their outputs are the inputs here.

The variant tables and every baseline and single-score column ship in `data/`, so any comparison against CADD, phyloP, GERP++, phastCons, DANN, GPN-MSA or Evo2 runs as it stands. Refitting the ensemble additionally needs the AlphaGenome matrices and foundation model blocks, 2.17 GB attached to the [latest release](https://github.com/everyday-omics/gfme/releases/latest). Unpack them and point `GFME_DATA` at the result; see [data/README.md](data/README.md).

| script | what it produces |
|---|---|
| `02_representation.py` | S4.1 — the AlphaGenome representation search |
| `03_fm_combination.py` | S4.2 — all 511 subsets of the nine foundation models |
| `04_classifier_comparison.py` | S4.3 — classifier families under nested cross-validation |
| `05_hyperparameters.py` | S4.4 — the CatBoost grid |
| `06_per_tool_calibration.py` | S4.5 — does supervision lift any single tool? |
| `07_train_final_model.py` | the frozen ensemble and its out-of-fold predictions |
| `08_evaluate.py` | Tables 2, 3, 4, 5 — per-category AUROC on each cohort |
| `09_screening.py` | Tables 1 and S1 — the screening comparison |
| `10_evo2_size.py` | S5 — Evo2 1B against 7B, and the ablation |
| `11_acmg_calibration.py` | S8 — PP3/BP4 evidence tiers, `--arm indel` for the indel tiers |
| `12_reclassification.py` | the VUS and conflicting cohorts |
| `13_figures.py` | every figure |
| `15_predictive_values.py` | S6 — PPV and NPV against an assumed prevalence |
| `16_shap.py` | S10 — per-variant attribution |
| `17_importance.py` | gain importance by block |
| `18_acmg_criteria.py` | S8.1 — evidence tiers against the ClinVar classification |

Steps 02 to 05 are model selection and touch the training partition only. Step 07 freezes one model, and everything after it applies that model rather than refitting.

## Repository contents

| Directory | Contents |
|-----------|----------|
| `config/default.yaml` | every constant the analysis depends on; nothing is hardcoded elsewhere |
| `gfme/` | the library: one loader, one feature builder, one metric implementation |
| `scripts/` | the analysis, in order; each writes to `results/` |
| `data/variants/` | thirteen cohorts: one row per variant, with label and consequence |
| `data/scores/` | pre-computed foundation model and baseline scores, one column per tool |
| `models/` | the SNV and indel ensembles, with their ACMG/AMP evidence tiers |
| `tests/` | unit tests, leakage assertions, and the golden numbers |

## Design rules

`gfme` is deliberately small. `io.load_dataset` is the only function that opens a data file; `features.build` is the only place the AlphaGenome aggregation exists; `metrics.per_category` is the only implementation of the headline metric. Anything that reads a data file some other way is a bug.

**One loader.** Every data file is opened by `gfme.io`. Joins are by variant key, never by row position.

**One feature builder.** `features.build` produces the AlphaGenome block for any of the ten representation arms the search compares, including the one the pipeline uses. The arm that wins the search is therefore the arm that runs.

**Config, not constants.** Fold counts, hyperparameters, category lists and seeds live in `config/default.yaml`. Every result records the fingerprint of the configuration that produced it, and a saved model refuses to be applied under a different one.

**Fail loudly.** Shapes and row alignment are asserted at every join. An unrecognised consequence term raises rather than falling into a residual bucket.

**Deterministic.** Fixed seeds throughout. CatBoost is deterministic for a fixed column order but its fit changes when columns are permuted, so column order is part of the contract and is asserted against the saved model's feature list.

## License

| Component | License | Commercial use |
|-----------|---------|---------------|
| Code (`gfme/`, `scripts/`, `tests/`) | [Apache 2.0](LICENSE) | Yes |
| Trained models (`models/*.cbm`) | [AlphaGenome Model Parameters ToU](https://deepmind.google.com/science/alphagenome/model-terms) | No |
| AlphaGenome features (release assets) | [AlphaGenome Model Parameters ToU](https://deepmind.google.com/science/alphagenome/model-terms) | No |
| GPN-MSA scores (`gpnmsa` column) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | No |
| Evo2 scores (`evo2` column) | [Apache 2.0](https://github.com/ArcInstitute/evo2/blob/main/LICENSE) | Yes |

The trained CatBoost models were built on AlphaGenome outputs and inherit its non-commercial terms. See [NOTICE.md](NOTICE.md) for full details.

**Not validated for clinical use.** Do not use GFME scores as the sole basis for clinical variant interpretation.
