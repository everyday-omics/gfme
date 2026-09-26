# Data

Everything the analysis reads. Foundation model inference is not reproduced here; these are its
outputs. `manifest.json` records, per dataset, which file every column came from and how it was
joined.

## Layout

```
variants/<name>.parquet     one row per variant: key, chrom, pos, ref, alt, label, consequence
alphagenome/<name>.npz      the 2,536 raw AlphaGenome columns, keyed
scores/<name>.parquet       one column per side tool and baseline, keyed, NaN where unscored
blocks/<name>.<model>.npz   the multi-column foundation model outputs (Borzoi, the embedding models)
```

A key is `chrom:pos:ref:alt` with the chromosome carrying its `chr` prefix. Every join in the
code is by key. Coverage is a reported quantity, so a tool that could not score a variant leaves
NaN and is never filled in.

## Datasets

| name | n | what it is |
|---|---|---|
| `clinvar_train` | 7,577 | ClinVar ≥1★, submitted and classified before 2023, balanced 1:1 pathogenic:benign within each of the seven consequence categories, with the external-cohort overlap removed |
| `clinvar_train_excluded` | 1,194 | the variants removed from training because they also appear in VKGL or MGeND. Identity only, no features — they are present in `external` |
| `clinvar_test` | 40,179 | ClinVar 2023 onwards. Ten majority-class variants per minority-class variant within each category, missense minority capped at 1,000 |
| `external` | 37,745 | VKGL and MGeND, downsampled the same way. Held out from *our training*, not absent from ClinVar |
| `indel_train` / `indel_test` | 6,480 / 4,428 | the indel arm, ≤50 bp, six categories, balanced 1:1 within each |
| `external_indel` | 14,573 | VKGL and MGeND indels, not rebalanced — AUROC does not depend on prevalence |
| `clinvar_pre2023` | 8,771 | the whole pre-2023 partition, before the external-overlap removal, used by the screening comparison |
| `clinvar_screen10k` | 10,000 | a random draw from the pre-2023 snapshot at its natural composition, 1,548 pathogenic and 8,452 benign, which is the sample Table S1 reports |
| `reclass_vus` / `reclass_conflicting` | 20,171 / 7,259 | SNVs uncertain or conflicting in January 2023 and definitively classified since |
| `reclass_vus_indel` / `reclass_conflicting_indel` | 1,910 / 643 | the same, for indels |

The training partition is 7,577 rather than 8,771 because the 1,194 variants shared with the
external cohort are removed **before** any model selection runs, so nothing in the paper is
selected on a variant the external evaluation later scores.

## What is not in the tree

`alphagenome/` and `blocks/` hold the raw model outputs, 2.17 GB across twenty files, which is
past what a git tree should carry. They are attached to the repository's latest release instead,
one asset per cohort, so you download only the cohorts you need:

    https://github.com/everyday-omics/gfme/releases/latest

Unpack them anywhere and point `GFME_DATA` at the directory that contains `variants/`,
`alphagenome/`, `scores/` and `blocks/`:

    export GFME_DATA=/path/to/gfme-data

Without them the repository still answers most questions: the variant tables name every variant
in every cohort with its label and consequence, and `scores/` carries the baseline and
single-score foundation model values, so every comparison against CADD, phyloP, GERP++,
phastCons, DANN, GPN-MSA and Evo2 runs as it stands. What needs the matrices is anything that
refits the ensemble.

## Sources

| source | release | used for |
|---|---|---|
| ClinVar VCF, GRCh38 | `##fileDate=2026-02-01` | the internal training and test partitions, and the reclassification cohorts |
| ClinVar VCF, GRCh38 | January 2023 | the 2023 snapshot that defines which variants were VUS or conflicting at the time |
| VKGL | consensus release, 153,168 SNVs | the external cohort |
| MGeND | 13,774 SNVs | the external cohort |

Labels are ClinVar's clinical significance collapsed to pathogenic (P, LP) and benign (B, LB);
conflicting and uncertain records are excluded from the labelled sets and form the
reclassification cohorts instead. Review status must be at least one star.

## Consequence categories

Seven for SNVs: intronic, synonymous, non-coding, splice, LoF, other, missense. Six for indels:
in-frame, LoF, other, splice, intronic, non-coding.

Assignment is a lookup against an explicit term table, held in `gfme/consequences.py`, with one
table per annotation vocabulary — ClinVar's `MC` field for the internal cohorts and Ensembl VEP
for the external ones. The two tables agree term for term, so the per-category rows of the
internal and external tables describe the same classes.

Two assignments are worth naming because they look like errors and are not:

* RNA-gene variants (ClinVar `non-coding_transcript_variant`, VEP `non_coding_transcript_variant`)
  are **other**, not non-coding. `non-coding` here means UTR, upstream, downstream and intergenic:
  variants at a protein-coding locus that do not change the protein. A variant in an RNA gene is
  a different object.
* Start-loss (ClinVar `initiator_codon_variant`, VEP `start_lost`) is **other**, not LoF.

`splice` is the canonical ±1,2 acceptor and donor positions only, which is all ClinVar's MC field
marks; every extended splice term is intronic.

About one ClinVar record in five carries more than one MC term, so a precedence order decides the
category for those. The order is stated in `gfme/consequences.py` and in the Methods. It is a
convention, not a biological claim.

## Foundation models and tools

| | version / settings |
|---|---|
| AlphaGenome | API, 2,536 output columns per variant, raw and quantile layers |
| Evo2 | 7B for the analysis, 1B compared in S5; negated log-likelihood ratio |
| GPN-MSA | negated score; does not score indels, where the MSA gap fraction stands in |
| Borzoi | Flashzoi, 524,288 bp context, 7,611 tracks, log2 fold change at the maximally changed bin |
| DNABERT-2, NT-v2, HyenaDNA, GROVER, Caduceus-Ph | embedding difference vectors; L2 distance as the zero-shot score |
| CADD | v1.7, pre-scored SNVs by tabix; indels scored locally through the full pipeline |
| phyloP, phastCons | UCSC hg38 100-way bigWig |
| GERP++ | hg19 bigWig with liftover |
| DANN | ANNOVAR hg19 database with liftover |
| SpliceAI | `spliceai` package, distance parameter 50, maximum of the four delta channels |
| REVEL, AlphaMissense | missense only, uncalibrated |

## Terms

The code in this repository is Apache 2.0. The inputs are not, and two of them restrict
commercial use, which carries over to anything derived from them:

| | terms |
|---|---|
| AlphaGenome | non-commercial, under the AlphaGenome model terms. `models/*.cbm` are fitted on its output and inherit this |
| GPN-MSA | CC BY-NC-ND 4.0. Its scores are a column in `scores/` |
| Evo2 | Apache 2.0 |
| CADD | free for non-commercial use; commercial use requires a licence |
| ClinVar, UCSC phyloP and phastCons | public domain |
| VKGL, MGeND, DANN, Borzoi, DNABERT-2, NT-v2, HyenaDNA, GROVER, Caduceus-Ph, SpliceAI, REVEL, AlphaMissense | each under its own terms, which this repository does not relicense |

So the practical position is that the models and the score tables here are for non-commercial
use, whatever the licence on the code that produced them.
