"""Partitions and cross-validation folds.

The train/test split is temporal, not random: training is ClinVar variants submitted and
classified before 2023, testing is 2023 onwards. A random split over a database that grows by
resubmission would let a variant's own later reclassification inform its training label, and would
also reward a tool that had already been used as PP3/BP4 evidence on the very variants it is being
scored on. The temporal split makes the evaluation pseudo-prospective instead.

Cross-validation inside the training partition is stratified jointly on label and consequence, so
every fold carries the same category mix — with categories as small as 168 variants, plain
label-stratification leaves folds that differ in composition enough to move macro-AUROC on its own.

Nothing here ever reads the test partition. `assert_no_leakage` is the check that says so.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _strata(labels, consequence):
    return np.array([f'{c}|{y}' for c, y in zip(np.asarray(consequence), np.asarray(labels))])


def folds(labels, consequence, k, seed=42):
    """`k` stratified folds as (train_index, test_index) pairs. Deterministic for a given seed."""
    strata = _strata(labels, consequence)
    # A stratum with fewer members than k cannot be spread across the folds; fall back to the
    # label alone for those rows rather than letting scikit-learn drop or warn silently.
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < k])
    if rare:
        strata = np.array([s if s not in rare else f'rare|{s.rsplit("|", 1)[1]}' for s in strata])
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strata)), strata))


def nested_folds(labels, consequence, outer_k, inner_k, seed=42):
    """Outer folds, each carrying its own inner folds over the outer training rows.

    Model selection happens in the inner loop and is scored on the outer test fold, so the
    estimate is not contaminated by the choice it is estimating.
    """
    out = []
    for outer_train, outer_test in folds(labels, consequence, outer_k, seed):
        inner = folds(np.asarray(labels)[outer_train], np.asarray(consequence)[outer_train],
                      inner_k, seed)
        out.append({
            'train': outer_train,
            'test': outer_test,
            'inner': [(outer_train[a], outer_train[b]) for a, b in inner],
        })
    return out


def sample_weights(labels, consequence, categories):
    """Weights that give every category the same total influence.

    Without them a category with 3,265 training variants counts twenty times a category with 168,
    while macro-AUROC weights them equally — so the objective and the metric would disagree.
    """
    consequence = np.asarray(consequence)
    weights = np.ones(len(consequence), dtype=float)
    present = [c for c in categories if (consequence == c).sum()]
    for cat in present:
        mask = consequence == cat
        weights[mask] = len(consequence) / (len(present) * mask.sum())
    return weights


def assert_no_leakage(train_keys, test_keys, name=''):
    """No variant may appear in both partitions."""
    overlap = set(map(str, train_keys)) & set(map(str, test_keys))
    if overlap:
        raise AssertionError(f'{name}: {len(overlap):,} variants are in both partitions, '
                             f'e.g. {sorted(overlap)[:5]}')


def assert_excluded(train_keys, excluded_keys, name=''):
    """The external-overlap exclusion list must not appear in training."""
    leaked = set(map(str, train_keys)) & set(map(str, excluded_keys))
    if leaked:
        raise AssertionError(f'{name}: {len(leaked):,} excluded variants are still in training')


def assert_fits_in_train(fold_pairs, n_train, name=''):
    """Every fold index must address a training row. Catches folds built over the wrong table."""
    for i, (a, b) in enumerate(fold_pairs):
        if len(a) == 0 or len(b) == 0:
            raise AssertionError(f'{name}: fold {i} is empty')
        if max(a.max(), b.max()) >= n_train:
            raise AssertionError(f'{name}: fold {i} indexes row {max(a.max(), b.max())} '
                                 f'of a {n_train}-row partition')
