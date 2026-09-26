#!/usr/bin/env python3
"""S4.4 — the CatBoost hyperparameter grid, under nested cross-validation.

The grid is iterations x depth x learning rate. Nested, for the same reason as S4.3: the point of
the exercise is partly to measure how much a flat search would have flattered itself.

**The iterations axis is collapsed, exactly.** A CatBoost model fitted for 1,500 iterations
*contains* the 500- and 1,000-iteration models as prefixes, so `predict_proba(X, ntree_end=n)` is
bit-identical to a separate n-iteration fit — verified on the real matrix at three grid corners,
maximum absolute difference 0.0. Fitting once per (depth, learning rate) and reading the shorter
models off the prefix therefore cuts the grid from 36 fits to 12 with no approximation. This holds
only because there is no early stopping, the learning rate is explicit, `model_shrink_rate` is 0
and boosting is Plain; the script asserts those conditions rather than assuming them.

    python3 scripts/05_hyperparameters.py
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import io, metrics, models, splits              # noqa: E402


def grid(config):
    settings = config['hyperparameter_grid']
    axes = [settings['iterations'], settings['depth'], settings['learning_rate'],
            settings['subsample'], settings['colsample_bylevel']]
    for iterations, depth, lr, subsample, colsample in itertools.product(*axes):
        yield {'iterations': iterations, 'depth': depth, 'learning_rate': lr,
               'subsample': subsample, 'colsample_bylevel': colsample}


def _base(config, spec):
    """The configuration with the largest iteration count for this (depth, lr, ...) cell."""
    hyperparameters = dict(config['model']['hyperparameters'])
    hyperparameters.update(spec)
    hyperparameters['iterations'] = max(config['hyperparameter_grid']['iterations'])
    return hyperparameters


def _assert_prefix_safe(hyperparameters):
    assert 'early_stopping_rounds' not in hyperparameters, \
        'early stopping breaks the prefix identity — the shorter model is not a prefix'
    assert hyperparameters.get('model_shrink_rate', 0) == 0
    assert hyperparameters.get('boosting_type', 'Plain') == 'Plain'


def evaluate_cell(X, y, consequence, config, categories, train_index, test_index, cell):
    """Every iteration count for one (depth, learning rate) cell, from a single fit."""
    hyperparameters = _base(config, cell)
    _assert_prefix_safe(hyperparameters)
    model = models.fit(X[train_index], y[train_index], config, consequence[train_index],
                       'catboost', hyperparameters)
    out = {}
    for iterations in config['hyperparameter_grid']['iterations']:
        probability = model.predict_proba(X[test_index], ntree_end=iterations)[:, 1]
        out[iterations] = metrics.macro_auroc(y[test_index], probability,
                                              consequence[test_index], categories,
                                              config['metrics']['min_cell'])
    return out


def label(spec):
    return (f'it{spec["iterations"]}|d{spec["depth"]}|lr{spec["learning_rate"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='hyperparameters')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    started = time.time()

    X, feature_names, df = io.load_matrix('clinvar_train', cfg)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()

    configurations = list(grid(cfg))
    cells = sorted({(c['depth'], c['learning_rate'], c['subsample'], c['colsample_bylevel'])
                    for c in configurations})
    print(f'{X.shape[0]:,} training variants x {X.shape[1]} features', flush=True)
    print(f'{len(configurations)} configurations, collapsed to {len(cells)} fits per fold '
          f'via the iteration prefix\n', flush=True)

    outer_k, inner_k = cfg['cv']['outer_k'], cfg['cv']['inner_k']
    folds = splits.nested_folds(y, consequence, outer_k, inner_k, cfg['seed'])
    fold_reports, outer_scores = [], []

    for i, fold in enumerate(folds):
        inner = {label(c): [] for c in configurations}
        for inner_train, inner_test in fold['inner']:
            for depth, lr, subsample, colsample in cells:
                cell = {'depth': depth, 'learning_rate': lr, 'subsample': subsample,
                        'colsample_bylevel': colsample}
                by_iterations = evaluate_cell(X, y, consequence, cfg, categories,
                                              inner_train, inner_test, cell)
                for iterations, macro in by_iterations.items():
                    inner[label({**cell, 'iterations': iterations})].append(macro)
        mean_inner = {k: float(np.mean(v)) for k, v in inner.items()}
        winner = max(mean_inner, key=mean_inner.get)
        chosen = next(c for c in configurations if label(c) == winner)

        hyperparameters = dict(cfg['model']['hyperparameters'])
        hyperparameters.update(chosen)
        model = models.fit(X[fold['train']], y[fold['train']], cfg, consequence[fold['train']],
                           'catboost', hyperparameters)
        outer = metrics.macro_auroc(y[fold['test']], models.predict(model, X[fold['test']]),
                                    consequence[fold['test']], categories,
                                    cfg['metrics']['min_cell'])
        outer_scores.append(outer)
        fold_reports.append({'fold': i, 'winner': winner, 'inner_macro': mean_inner[winner],
                             'outer_macro': outer, 'all_inner': mean_inner})
        print(f'fold {i}: {winner}  inner {mean_inner[winner]:.4f}  outer {outer:.4f}  '
              f'({(time.time() - started) / 60:.0f} min)', flush=True)

    winners = [f['winner'] for f in fold_reports]
    mean_inner = {label(c): float(np.mean([f['all_inner'][label(c)] for f in fold_reports]))
                  for c in configurations}

    # The documented selection rule: outer-fold majority per axis, broken by the inner-loop mean.
    selected = {}
    for axis in ('iterations', 'depth', 'learning_rate'):
        votes = [next(c for c in configurations if label(c) == w)[axis] for w in winners]
        counts = {v: votes.count(v) for v in set(votes)}
        top = max(counts.values())
        tied = [v for v, n in counts.items() if n == top]
        if len(tied) == 1:
            selected[axis] = tied[0]
            note = f'{top}/{outer_k} folds'
        else:
            best = max(mean_inner, key=mean_inner.get)
            selected[axis] = next(c for c in configurations if label(c) == best)[axis]
            note = f'tied {tied}, broken on the inner-loop mean'
        print(f'  {axis:15s} -> {selected[axis]}   ({note})', flush=True)

    nested_macro = float(np.mean(outer_scores))
    flat_best = max(mean_inner.values())
    print(f'\nnested macro {nested_macro:.4f}   best inner-loop mean {flat_best:.4f}   '
          f'selection bias {flat_best - nested_macro:+.4f}', flush=True)
    print(f'folds agreeing on the complete configuration: '
          f'{winners.count(max(set(winners), key=winners.count))} of {outer_k}', flush=True)

    locked = config_module.as_dict(cfg['model']['hyperparameters'])
    matches = all(locked[a] == selected[a] for a in selected)
    print(f'selected {selected}; config carries '
          f'{ {a: locked[a] for a in selected} }  '
          + ('MATCH' if matches else '<-- MISMATCH'), flush=True)

    io.write_result('05_hyperparameters', args.name, {
        'protocol': f'nested {outer_k}x{inner_k} cross-validation, {len(configurations)} '
                    f'configurations, iterations axis collapsed exactly via the model prefix. '
                    f'Selection rule: outer-fold majority per axis, ties broken on the '
                    f'inner-loop mean.',
        'n_train': int(len(y)), 'n_configurations': len(configurations),
        'selected': selected, 'config_hyperparameters': locked, 'matches_config': matches,
        'nested_macro': nested_macro, 'best_inner_mean': flat_best,
        'selection_bias': flat_best - nested_macro,
        'mean_inner_by_configuration': mean_inner, 'folds': fold_reports,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
