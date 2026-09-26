"""Data access. The only module that opens a data file.

Every dataset is three things, and nothing reads them any other way:

    data/variants/<name>.parquet      one row per variant: key, locus, label, consequence
    data/alphagenome/<name>*.npz      the 2,536 raw AlphaGenome columns, keyed
    data/scores/<name>.parquet        one column per side tool and baseline, keyed

Joins are by variant key, never by row position. Positional alignment between a variant table and
a score array is only correct while every file is built in the same order by the same script, and
it fails silently the moment one of them is rebuilt — the failure is a scrambled feature matrix
that still has the right shape. `align()` therefore takes keys and returns a mask of what was
found, and every caller is expected to look at it.

A key is `chrom:pos:ref:alt` with the chromosome carrying its `chr` prefix.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import features

ROOT = Path(__file__).resolve().parent.parent
#: The AlphaGenome matrices are the bulk of the repository. Set GFME_DATA to keep them elsewhere.
DATA = Path(os.environ.get('GFME_DATA') or ROOT / 'data')

#: name -> (variant table, AlphaGenome stem, score table, variant type)
DATASETS = {
    'clinvar_train':          ('clinvar_train',          'clinvar_train',    'clinvar_train',    'snv'),
    'clinvar_pre2023':        ('clinvar_pre2023',        'clinvar_pre2023',  'clinvar_pre2023',  'snv'),
    'clinvar_train_excluded': ('clinvar_train_excluded', None,               None,               'snv'),
    'clinvar_test':           ('clinvar_test',           'clinvar_test',     'clinvar_test',     'snv'),
    'external':               ('external',               'external',         'external',         'snv'),
    'clinvar_screen10k':      ('clinvar_screen10k',      'clinvar_screen10k', 'clinvar_screen10k', 'snv'),
    'reclass_vus':            ('reclass_vus',            'reclass_vus',      'reclass_vus',      'snv'),
    'reclass_conflicting':    ('reclass_conflicting',    'reclass_conflicting', 'reclass_conflicting', 'snv'),
    'reclass_vus_indel':      ('reclass_vus_indel',      'reclass_vus_indel', 'reclass_vus_indel', 'indel'),
    'reclass_conflicting_indel': ('reclass_conflicting_indel', 'reclass_conflicting_indel',
                                 'reclass_conflicting_indel', 'indel'),
    'indel_train':            ('indel_train',            'indel_train',      'indel_train',      'indel'),
    'indel_test':             ('indel_test',             'indel_test',       'indel_test',       'indel'),
    'external_indel':         ('external_indel',         'external_indel',   'external_indel',   'indel'),
}

REQUIRED_COLUMNS = ('key', 'chrom', 'pos', 'ref', 'alt', 'label', 'consequence')


class MissingData(FileNotFoundError):
    pass


def normalise_key(key) -> str:
    key = str(key)
    return key if key.startswith('chr') else 'chr' + key


def variant_type(name: str) -> str:
    return DATASETS[name][3]


# ----------------------------------------------------------------------------------------------
# Variant tables
# ----------------------------------------------------------------------------------------------
def load_dataset(name: str) -> pd.DataFrame:
    """The variant table for `name`. The only entry point for variant data."""
    if name not in DATASETS:
        raise KeyError(f'unknown dataset {name!r}; known: {sorted(DATASETS)}')
    path = DATA / 'variants' / f'{DATASETS[name][0]}.parquet'
    if not path.exists():
        raise MissingData(f'{path} — see data/README.md')
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f'{path.name} is missing {missing}')
    df['key'] = df['key'].map(normalise_key)
    if df['key'].duplicated().any():
        dup = df.loc[df['key'].duplicated(), 'key'].head().tolist()
        raise AssertionError(f'{path.name}: duplicate keys, e.g. {dup}')
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# AlphaGenome
# ----------------------------------------------------------------------------------------------
def _alphagenome_shards(stem: str):
    paths = sorted(glob.glob(str(DATA / 'alphagenome' / f'{stem}.npz'))) or \
        sorted(glob.glob(str(DATA / 'alphagenome' / f'{stem}.shard*.npz')))
    if not paths:
        raise MissingData(f'no AlphaGenome npz for {stem!r} in {DATA / "alphagenome"}')
    return paths


def load_alphagenome(name: str, keys=None, layer='raw'):
    """Raw AlphaGenome columns for `keys`. Returns (matrix, feature_names, found_mask).

    Rows whose key has no AlphaGenome record come back as NaN and are flagged in `found_mask`;
    they are not dropped, because dropping them here would silently change the denominator of
    every downstream metric.
    """
    stem = DATASETS[name][1]
    if stem is None:
        raise MissingData(f'{name} ships variant identity only, no features')

    lookup, blocks, feature_names, offset = {}, [], None, 0
    for path in _alphagenome_shards(stem):
        z = np.load(path, allow_pickle=True)
        array = z[layer] if layer in z.files else z['ag_features']
        names = [str(x) for x in z['feature_names']]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise AssertionError(f'{path}: feature_names differ from the first shard')
        for i, k in enumerate(z['keys']):
            lookup.setdefault(normalise_key(k), offset + i)
        blocks.append(np.asarray(array, dtype=np.float32))
        offset += len(z['keys'])
    stacked = np.vstack(blocks)

    if keys is None:
        return stacked, feature_names, np.ones(len(stacked), bool)

    keys = [normalise_key(k) for k in keys]
    matrix = np.full((len(keys), stacked.shape[1]), np.nan, np.float32)
    found = np.array([k in lookup for k in keys])
    rows = np.flatnonzero(found)
    if len(rows):
        matrix[rows] = stacked[[lookup[keys[i]] for i in rows]]
    return matrix, feature_names, found


# ----------------------------------------------------------------------------------------------
# Side tools and baselines
# ----------------------------------------------------------------------------------------------
def load_scores(name: str, keys=None, tools=None) -> pd.DataFrame:
    """Side-tool and baseline scores, indexed by key.

    One tidy table per dataset, one column per tool, NaN where a tool could not score a variant.
    Coverage is a reported quantity, so it is never filled in or interpolated.
    """
    stem = DATASETS[name][2]
    if stem is None:
        raise MissingData(f'{name} ships variant identity only, no scores')
    path = DATA / 'scores' / f'{stem}.parquet'
    if not path.exists():
        raise MissingData(f'{path} — see data/README.md')
    df = pd.read_parquet(path)
    df['key'] = df['key'].map(normalise_key)
    df = df.drop_duplicates('key').set_index('key')
    if tools is not None:
        absent = [t for t in tools if t not in df.columns]
        if absent:
            raise AssertionError(f'{path.name} has no column for {absent}; '
                                 f'available: {sorted(df.columns)}')
        df = df[list(tools)]
    if keys is not None:
        df = df.reindex([normalise_key(k) for k in keys])
    return df


def load_block(name: str, model: str, keys=None, raw_borzoi=False):
    """A foundation model's multi-column output for a cohort.

    Borzoi emits a 7,611-track log2 fold change and the five embedding models an allele-difference
    vector. The combination search consumes those blocks; the comparison tables use their scalar
    reductions, which live in the score table instead.

    The Borzoi block is returned in the reduced per-tissue representation the manuscript
    describes, because that is the representation Borzoi is entitled to be judged on --- handing a
    Random Forest 7,611 unreduced tracks against 747 reduced AlphaGenome columns measures the
    reduction, not the model. `raw_borzoi=True` returns the tracks for the callers that want the
    unreduced input.
    """
    path = DATA / 'blocks' / f'{name}.{model}.npz'
    if not path.exists():
        raise MissingData(f'{path} — no {model} block for {name}')
    z = np.load(str(path), allow_pickle=True)
    matrix = np.asarray(z['block'], dtype=np.float32)
    if model.startswith('borzoi') and not raw_borzoi:
        from . import features
        matrix = features.borzoi_tissue(matrix)
    if keys is None:
        return matrix, [normalise_key(k) for k in z['keys']]
    lookup = {normalise_key(k): i for i, k in enumerate(z['keys'])}
    rows = np.array([lookup.get(normalise_key(k), -1) for k in keys])
    out = np.full((len(keys), matrix.shape[1]), np.nan, np.float32)
    found = rows >= 0
    out[found] = matrix[rows[found]]
    return out, list(keys)


def coverage(scores: pd.DataFrame) -> dict:
    """Fraction of rows each tool actually scored — reported alongside every table."""
    return {c: float(np.isfinite(scores[c].to_numpy(dtype=float)).mean()) for c in scores.columns}


# ----------------------------------------------------------------------------------------------
# Assembled model input
# ----------------------------------------------------------------------------------------------
def load_matrix(name: str, config, arm=None):
    """The model input matrix for a dataset: [AlphaGenome | side | consequence].

    This is the single path from files to a fitted model's input, so the training matrix and
    every evaluation matrix are built by the same code and cannot drift apart.
    """
    from . import consequences

    df = load_dataset(name)
    vtype = variant_type(name)
    keys = df['key'].tolist()

    raw, feature_names, found = load_alphagenome(name, keys)
    if not found.all():
        print(f'[io] {name}: {(~found).sum():,} of {len(found):,} variants have no '
              f'AlphaGenome record', flush=True)

    arm_spec = arm or config['alphagenome']['arms'][config['alphagenome']['selected']]
    quantile = None
    if arm_spec.get('biosample') == 'quantile':
        quantile, _, _ = load_alphagenome(name, keys, layer='quantile')
    if arm is None:
        ag, ag_names = features.build_selected(raw, feature_names, config, quantile)
    else:
        ag, ag_names = features.build(raw, feature_names, arm, quantile,
                                      config['alphagenome']['activity_scorers'])

    specs = config['side_features'][vtype]
    wanted = [s['name'] for s in specs]
    available = load_scores(name, keys)
    side_raw = {}
    for spec in specs:
        col = spec['name']
        if col in available.columns:
            side_raw[col] = available[col].to_numpy(dtype=float)
        elif col == 'indel_size' and 'indel_size' in df.columns:
            side_raw[col] = df['indel_size'].to_numpy(dtype=float)
        else:
            raise AssertionError(f'{name}: no scores for side feature {col!r}')
    directed = features.apply_directions(side_raw, config, vtype)
    side = np.column_stack([directed[c] for c in wanted])
    side_names = [f'{c}_neg' if s['negate'] else c for c, s in zip(wanted, specs)]

    cons_num = np.array([consequences.encode(c, vtype) for c in df['consequence']], dtype=float)
    matrix, names = features.assemble(ag, ag_names, side, side_names, cons_num, config)
    assert len(matrix) == len(df), (matrix.shape, df.shape)
    return matrix, names, df


# ----------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------
def results_dir(step: str) -> Path:
    path = ROOT / 'results' / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(step: str, name: str, payload: dict, config=None) -> Path:
    """Write a result with the config fingerprint that produced it."""
    import json

    from . import config as config_module

    body = dict(payload)
    if config is not None:
        body['_config_fingerprint'] = config_module.fingerprint(config)
    path = results_dir(step) / f'{name}.json'
    with open(path, 'w') as fh:
        json.dump(body, fh, indent=2, sort_keys=False, default=float)
    return path
