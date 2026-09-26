# Models

The two models the article reports, and the evidence tiers calibrated on them. Nothing here is
refitted by the analysis; step 07 produces them and every later step applies them.

| file | what it is |
|---|---|
| `gfme_snv.cbm` | the SNV ensemble, CatBoost, 750 features: 747 AlphaGenome activity columns, the GPN-MSA score, the Evo2 score and the consequence encoding |
| `gfme_indel.cbm` | the indel ensemble, 751 features: the same AlphaGenome block, the Evo2 score, MSA gap fraction, indel length and the consequence encoding |
| `*.manifest.json` | the feature names in order, and the fingerprint of the configuration the model was fitted under |
| `acmg_thresholds_snv.json` | PP3 and BP4 score thresholds per consequence category, with the likelihood ratio each interval achieved on the test set |
| `acmg_thresholds_indel.json` | the same for indels |

Column order is part of the contract. CatBoost is deterministic for a fixed order but its fit
changes when columns are permuted, so `models.load` refuses a matrix whose feature names do not
match the manifest.

The thresholds are fitted on out-of-fold predictions from the training partition and checked
against the held-out test set; a tier whose interval held no benign variant is marked unvalidated
in the JSON and should not be used as evidence.
