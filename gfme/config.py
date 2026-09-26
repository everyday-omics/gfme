"""Configuration loading.

`config/default.yaml` holds every constant the analysis depends on. Loading it returns a
deeply-immutable mapping, so a script cannot reach in and change a fold count for its own run and
leave the file describing something that never happened.

Each run records the hash of the settings that produced it. A checkpoint or cached result whose
hash does not match the current config is rejected rather than resumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / 'config' / 'default.yaml'


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


def _validate(cfg):
    cats = cfg['categories']['snv']
    assert len(set(cats)) == len(cats) == 7, f'expected seven SNV categories, got {cats}'
    assert 'missense' in cats, 'missense is one of the seven categories, not a separate arm'
    assert len(set(cfg['categories']['indel'])) == 6

    ag = cfg['alphagenome']
    assert ag['selected'] in ag['arms'], f'{ag["selected"]!r} is not one of the arms'
    assert len(ag['activity_scorers']) == 7

    assert cfg['block_order'] == ['alphagenome', 'side', 'consequence']
    n_fm = len(cfg['foundation_models'])
    assert n_fm == 9, f'the combination search is 2^{n_fm}-1 subsets; the paper reports 511'
    assert set(cfg['selected_combination']) <= set(cfg['foundation_models'])

    for k in ('outer_k', 'inner_k', 'oof_k'):
        assert cfg['cv'][k] >= 2, f'cv.{k} must be at least 2'
    assert cfg['metrics']['min_cell'] >= 1
    return cfg


def load(path=None):
    """Load, validate and freeze the configuration."""
    path = Path(path) if path else DEFAULT
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return _freeze(_validate(cfg))


def override(cfg, assignments):
    """Apply `dotted.key=value` overrides and revalidate.

    Used to run one script across several settings — the representation arms, weighting on and
    off — without editing the file, so every variant is still described by a config that can be
    fingerprinted and recorded with its result.
    """
    body = _thaw(cfg)
    for assignment in assignments:
        path, _, raw = assignment.partition('=')
        if not _:
            raise ValueError(f'expected dotted.key=value, got {assignment!r}')
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = body
        parts = path.split('.')
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f'{path}: no section {part!r}')
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f'{path}: no setting {parts[-1]!r}')
        node[parts[-1]] = value
    return _freeze(_validate(body))


def describe(cfg):
    """The short label identifying a run's settings in logs and result names."""
    return (f'{cfg["alphagenome"]["selected"]}'
            f'/{"weighted" if cfg["model"]["sample_weights"] == "by_category" else "unweighted"}')


def fingerprint(cfg, *keys):
    """Stable hash of the whole config, or of the named top-level sections.

    Results and checkpoints are keyed by this, so a run cannot silently absorb rows produced
    under different settings.
    """
    subset = _thaw(cfg) if not keys else {k: _thaw(cfg[k]) for k in sorted(keys)}
    blob = json.dumps(subset, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def as_dict(cfg):
    """A mutable deep copy, for writing into a result manifest."""
    return _thaw(cfg)
