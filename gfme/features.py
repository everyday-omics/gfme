"""Feature construction. The only place the AlphaGenome aggregation exists.

AlphaGenome emits 2,536 columns per variant, named `<scorer>__<biosample>` (a handful of
scorer-level columns carry no biosample). AG-747 collapses that into

    19 per-scorer columns    signed max over every biosample, on raw output
  + 728 per-biosample columns signed max over the seven activity-level scorers, on raw output

Both blocks are raw. `build()` takes the arm as a parameter rather than hardcoding the choice,
because the representation search in scripts/02_representation.py compares ten arms and every one
of them must come out of this function — otherwise the arm that wins the search is not the arm the
rest of the pipeline runs.

Nine of the 19 per-scorer columns are algebraically redundant (each is the elementwise max of its
own scorer's biosample columns, and `contact_contact` is a verbatim duplicate of a biosample
column). They are kept because the published feature vector contains them and the frozen model was
fitted with them; `tests/test_features.py` pins the redundancy so it stays a known property rather
than a discovery.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

N_AG747 = 747


def _group(feature_names):
    """Column indices grouped by scorer and by biosample, both in sorted name order."""
    by_scorer, by_biosample = defaultdict(list), defaultdict(list)
    for i, name in enumerate(feature_names):
        by_scorer[name.split('__', 1)[0]].append(i)
        if '__' in name:
            by_biosample[name.rsplit('__', 1)[1]].append(i)
    return by_scorer, by_biosample


def _aggregate(matrix, columns, how, n_rows):
    """Signed max or mean over `columns`, ignoring non-finite cells. All-missing becomes 0.0.

    Signed, not absolute: the direction of an AlphaGenome effect is informative, and taking the
    magnitude discards it. A row with no finite cell in the group scores 0.0 — the same value a
    variant with no predicted effect gets — which is the published behaviour.
    """
    if not len(columns):
        return np.zeros(n_rows, np.float32)
    sub = matrix[:, columns]
    finite = np.isfinite(sub)
    if how == 'max':
        out = np.where(finite, sub, -np.inf).max(axis=1)
    elif how == 'mean':
        count = finite.sum(axis=1)
        out = np.where(finite, sub, 0.0).sum(axis=1) / np.maximum(count, 1)
        out[count == 0] = 0.0
    else:
        raise ValueError(f'unknown aggregation {how!r}')
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _scorer_set(which, scorers, activity_scorers):
    if which == 'activity':
        return set(activity_scorers)
    if which == 'all':
        return set(scorers)
    if which == 'change':
        # The per-scorer columns that measure a change between alleles rather than a level.
        return {s for s in scorers if s.endswith('_diff_log2_sum') or s == 'rnaseq_gene_lfc'}
    raise ValueError(f'unknown scorer set {which!r}')


def build(raw, feature_names, arm, quantile=None, activity_scorers=()):
    """AlphaGenome feature block for one arm. Returns (matrix, column names).

    `arm` is an entry from `alphagenome.arms` in the config. `quantile` is required only by arms
    whose biosample block reads the AlphaGenome quantile layer.
    """
    names = [str(x) for x in feature_names]
    raw = np.asarray(raw, dtype=np.float32)
    n = raw.shape[0]
    by_scorer, by_biosample = _group(names)
    scorers, biosamples = sorted(by_scorer), sorted(by_biosample)

    if arm.get('scorer') == 'none' and arm.get('biosample') == 'none':
        return raw, names                                   # the no-aggregation reference arm

    columns, column_names = [], []

    if arm.get('scorer') != 'none':
        # The scorer block is raw and max in every arm; it is the biosample block that varies.
        for s in scorers:
            columns.append(_aggregate(raw, by_scorer[s], 'max', n))
            column_names.append(f'scorer__{s}')

    if arm.get('biosample') != 'none':
        if arm['biosample'] == 'quantile':
            if quantile is None:
                raise ValueError(f'arm {arm.get("label")!r} needs the quantile layer')
            source = np.asarray(quantile, dtype=np.float32)
        else:
            source = raw
        keep = _scorer_set(arm.get('scorers', 'activity'), scorers, activity_scorers)
        keep_columns = {i for s in scorers if s in keep for i in by_scorer[s]}
        block, block_names = [], []
        for b in biosamples:
            cols = [i for i in by_biosample[b] if i in keep_columns]
            block.append(_aggregate(source, cols, arm.get('agg', 'max'), n))
            block_names.append(f'activity__{b}')
        block = np.column_stack(block)

        if arm.get('normalise') == 'row':
            # Keeps the pattern across biosamples and discards the overall level.
            mean = block.mean(axis=1, keepdims=True)
            sd = block.std(axis=1, keepdims=True)
            sd[sd < 1e-9] = 1.0
            block = ((block - mean) / sd).astype(np.float32)
        if arm.get('collapse') == 'mean':
            # Keeps the overall level and discards the pattern.
            block = block.mean(axis=1, keepdims=True)
            block_names = ['activity__mean']

        columns.extend(block.T)
        column_names.extend(block_names)

    matrix = np.column_stack(columns).astype(np.float32)
    assert matrix.shape[1] == len(column_names), (matrix.shape, len(column_names))
    return matrix, column_names


def build_selected(raw, feature_names, config, quantile=None):
    """The AlphaGenome block the pipeline actually uses.

    On real AlphaGenome output — 2,536 columns — this is AG-747, and the width is asserted. The
    function itself is not tied to that number: it aggregates whatever scorer and biosample
    vocabulary it is given, so the tests can exercise it on a small matrix.
    """
    arm = config['alphagenome']['arms'][config['alphagenome']['selected']]
    matrix, names = build(raw, feature_names, arm, quantile,
                          config['alphagenome']['activity_scorers'])
    if raw.shape[1] == config['alphagenome']['n_raw_columns']:
        assert matrix.shape[1] == N_AG747, \
            f'{config["alphagenome"]["n_raw_columns"]} AlphaGenome columns should reduce to ' \
            f'{N_AG747}, got {matrix.shape[1]}'
    return matrix, names


def assemble(ag_block, ag_names, side, side_names, cons_num, config):
    """The model input matrix: [AlphaGenome | side features | consequence].

    Column order is part of the contract — CatBoost is deterministic for a fixed order but its
    fit changes under a permutation — so the order is asserted here rather than trusted to the
    order arguments happen to arrive in.
    """
    assert tuple(config['block_order']) == ('alphagenome', 'side', 'consequence'), \
        config['block_order']
    n = ag_block.shape[0]
    side = np.asarray(side, dtype=np.float64).reshape(n, -1)
    cons_num = np.asarray(cons_num, dtype=np.float64).reshape(n, 1)
    assert side.shape[1] == len(side_names), (side.shape, side_names)

    matrix = np.hstack([ag_block, side, cons_num])
    # Missing scores become 0.0 rather than being dropped: a variant a side tool could not score
    # still has AlphaGenome features and still belongs in the evaluation.
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    names = list(ag_names) + list(side_names) + ['cons_num']
    assert matrix.shape[1] == len(names)
    return matrix, names


def apply_directions(scores, config, variant_type='snv'):
    """Negate the side scores distributed as higher-is-more-benign, so every column points the
    same way. Trees are sign invariant, but the sign is part of the published contract."""
    out = {}
    for spec in config['side_features'][variant_type]:
        name = spec['name']
        if name not in scores:
            continue
        out[name] = -np.asarray(scores[name], dtype=float) if spec['negate'] \
            else np.asarray(scores[name], dtype=float)
    return out


# ----------------------------------------------------------------------------------------------
# Borzoi
# ----------------------------------------------------------------------------------------------
#: Shipped with borzoi-pytorch; one row per output track, with the assay and the biological
#: sample encoded in its description.
def _borzoi_targets():
    """The targets file shipped inside borzoi-pytorch, wherever it is installed."""
    override = os.environ.get('BORZOI_TARGETS')
    if override:
        return override
    import importlib.util
    spec = importlib.util.find_spec('borzoi_pytorch')
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError('borzoi_pytorch is not installed; set BORZOI_TARGETS')
    return str(Path(spec.submodule_search_locations[0]) / 'precomputed' / 'targets.txt')


BORZOI_TARGETS = _borzoi_targets

N_BORZOI_TISSUE = 2423


def borzoi_tissue_groups(targets=None):
    targets = targets or BORZOI_TARGETS()
    """Track indices grouped by tissue, parsed from the Borzoi target descriptions.

    A description reads `ASSAY:tissue:...`, except for ChIP tracks, which read `CHIP:mark:tissue`
    and so put the tissue one field later. Every one of the 7,611 tracks lands in exactly one
    group, and there are 2,423 groups.
    """
    import pandas as pd

    table = pd.read_table(targets)
    table.columns = ['index'] + list(table.columns[1:])
    groups = defaultdict(list)
    for i, description in enumerate(table['description'].values):
        parts = str(description).split(':')
        if parts[0] == 'CHIP' and len(parts) >= 3:
            tissue = parts[2].strip()
        elif len(parts) >= 2:
            tissue = parts[1].strip()
        else:
            tissue = 'unknown'
        groups[tissue].append(i)
    return {k: groups[k] for k in sorted(groups)}


def borzoi_tissue(block, groups=None):
    """The Borzoi representation the manuscript uses: max |log2 fold change| within each tissue.

    The raw block is one signed log2 fold change per track, taken at the bin of largest absolute
    alt--ref difference. Two reductions are applied here and both matter. The absolute value is
    taken first, so that a track moving up and a track moving down do not cancel when the maximum
    is taken across the assays of a tissue. The maximum is then taken within the tissue, which
    turns 7,611 tracks into 2,423 columns.

    Passing the raw 7,611 tracks instead costs Borzoi a great deal in any comparison against a
    reduced representation, which is why the reduction lives here rather than in a script: every
    consumer of the Borzoi block gets the same one.
    """
    groups = groups if groups is not None else borzoi_tissue_groups()
    block = np.abs(np.asarray(block, dtype=np.float32))
    out = np.zeros((block.shape[0], len(groups)), dtype=np.float32)
    for j, columns in enumerate(groups.values()):
        out[:, j] = np.nanmax(block[:, columns], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
