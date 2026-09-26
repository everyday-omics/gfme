"""Discrimination metrics.

Macro-AUROC is the headline: the unweighted mean of the per-category AUROCs. It is used instead
of pooled AUROC because the categories differ enormously in pathogenic prevalence — in ClinVar,
splice-site variants are pathogenic in about 98% of classified submissions and intronic variants
in under 1% — so a pooled figure mostly measures whether a score has learned which category a
variant belongs to. Assigning every variant its category's prevalence alone reaches a pooled
AUROC of 0.962 while separating nothing within any category.

Two rules make the per-tool comparison fair to tools that do not address every category:

* a tool's macro is averaged over the categories it actually scores, so a missense-only predictor
  is not charged for the six categories it never claimed;
* a (tool, category) cell with fewer than `metrics.min_cell` scored variants, or with only one
  class present, is reported as absent rather than as a number.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true, scores):
    """AUROC over the finite scores, or None if the cell cannot support one."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() < 2 or len(np.unique(y_true[ok])) < 2:
        return None
    return float(roc_auc_score(y_true[ok], scores[ok]))


def per_category(y_true, scores, consequence, categories, min_cell=20):
    """{category: AUROC} for the cells that qualify, plus the macro over them.

    Returns (per_category, macro, n_scored). `macro` is None when no cell qualifies.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    consequence = np.asarray(consequence)
    out, counts = {}, {}
    for cat in categories:
        mask = (consequence == cat) & np.isfinite(scores)
        n = int(mask.sum())
        if n >= min_cell and len(np.unique(y_true[mask])) == 2:
            out[cat] = float(roc_auc_score(y_true[mask], scores[mask]))
            counts[cat] = n
    macro = float(np.mean(list(out.values()))) if out else None
    return out, macro, counts


def macro_auroc(y_true, scores, consequence, categories, min_cell=20):
    return per_category(y_true, scores, consequence, categories, min_cell)[1]


def bootstrap_ci(y_true, scores, consequence, categories, min_cell=20,
                 iterations=2000, level=0.95, seed=42):
    """Percentile confidence interval for macro-AUROC, resampling variants within category.

    Resampling within category rather than over the pooled set keeps each bootstrap replicate at
    the same category composition, so the interval describes uncertainty in the AUROCs and not in
    the category mix, which is a design choice of the benchmark and not a random quantity.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    consequence = np.asarray(consequence)
    index_by_cat = {c: np.flatnonzero(consequence == c) for c in categories}

    replicates = []
    for _ in range(iterations):
        take = np.concatenate([rng.choice(idx, size=len(idx), replace=True)
                               for idx in index_by_cat.values() if len(idx)])
        macro = per_category(y_true[take], scores[take], consequence[take],
                             categories, min_cell)[1]
        if macro is not None:
            replicates.append(macro)
    if not replicates:
        return None, None
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(replicates, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def evaluate(y_true, scores_by_tool, consequence, categories, min_cell=20):
    """Per-category and macro AUROC for a whole table's worth of tools."""
    return {tool: dict(zip(('per_category', 'macro', 'n_scored'),
                           per_category(y_true, s, consequence, categories, min_cell)))
            for tool, s in scores_by_tool.items()}
