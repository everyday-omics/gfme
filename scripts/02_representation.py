#!/usr/bin/env python3
"""S4.1 — how should AlphaGenome's 2,536 columns be represented?

AlphaGenome emits a matrix per variant, not a score, and publishes no recommended aggregation into
a pathogenicity estimate, so the representation is a modelling choice that has to be made and
justified. Ten arms are compared, each isolating one axis: raw against the quantile layer, which
scorers contribute to the per-biosample block, max against mean, the pattern across biosamples
against the overall level, and the two reference points of no aggregation at all and each block
on its own.

Selection is on the AlphaGenome block alone. Carrying GPN-MSA and Evo2 through the comparison
would let a strong side feature mask a weak representation, and the question here is what the
AlphaGenome output is worth. The combined figure is reported alongside but never selected on.

Nested cross-validation: the arm is chosen in the inner loop and scored on the outer fold. A flat
best-of-ten over the same data would report the maximum of ten noisy estimates as if it were an
estimate.

    python3 scripts/02_representation.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                      # noqa: E402
from gfme import consequences, features, io, metrics, models, splits   # noqa: E402

RF = {'n_estimators': 200, 'min_samples_leaf': 5, 'random_state': 42}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='representation')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    started = time.time()

    df = io.load_dataset('clinvar_train')
    keys = df['key'].tolist()
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    raw, feature_names, _ = io.load_alphagenome('clinvar_train', keys)
    quantile, _, _ = io.load_alphagenome('clinvar_train', keys, layer='quantile')
    print(f'{len(df):,} training variants x {raw.shape[1]:,} AlphaGenome columns', flush=True)

    scores = io.load_scores('clinvar_train', keys)
    side = np.column_stack([
        -scores['gpnmsa'].to_numpy(float), -scores['evo2'].to_numpy(float),
        np.array([consequences.encode(c, 'snv') for c in consequence], dtype=float)])
    side = np.nan_to_num(side, nan=0.0, posinf=0.0, neginf=0.0)

    arms = cfg['alphagenome']['arms']
    activity = cfg['alphagenome']['activity_scorers']
    built = {}
    for name, arm in arms.items():
        matrix, _ = features.build(raw, feature_names, arm, quantile, activity)
        built[name] = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        print(f'  {arm["label"]:62s} {built[name].shape[1]:5d} columns', flush=True)

    outer_k, inner_k = cfg['cv']['outer_k'], cfg['cv']['inner_k']
    folds = splits.nested_folds(y, consequence, outer_k, inner_k, cfg['seed'])
    fold_reports = []
    print(f'\nnested {outer_k}x{inner_k} cross-validation over {len(arms)} arms', flush=True)

    for i, fold in enumerate(folds):
        inner = {}
        for name, matrix in built.items():
            values = []
            for inner_train, inner_test in fold['inner']:
                model = models.fit(matrix[inner_train], y[inner_train], cfg,
                                   consequence[inner_train], 'random_forest', RF)
                values.append(metrics.macro_auroc(
                    y[inner_test], models.predict(model, matrix[inner_test]),
                    consequence[inner_test], categories, cfg['metrics']['min_cell']))
            inner[name] = float(np.mean(values))
        winner = max(inner, key=inner.get)
        matrix = built[winner]
        model = models.fit(matrix[fold['train']], y[fold['train']], cfg,
                           consequence[fold['train']], 'random_forest', RF)
        outer = metrics.macro_auroc(y[fold['test']], models.predict(model, matrix[fold['test']]),
                                    consequence[fold['test']], categories,
                                    cfg['metrics']['min_cell'])
        with_side = np.hstack([matrix, side])
        model = models.fit(with_side[fold['train']], y[fold['train']], cfg,
                           consequence[fold['train']], 'random_forest', RF)
        outer_side = metrics.macro_auroc(
            y[fold['test']], models.predict(model, with_side[fold['test']]),
            consequence[fold['test']], categories, cfg['metrics']['min_cell'])
        fold_reports.append({'fold': i, 'winner': winner, 'inner_macro': inner[winner],
                             'outer_macro': outer, 'outer_macro_with_side': outer_side,
                             'all_inner': inner})
        print(f'  fold {i}: {arms[winner]["label"]}  inner {inner[winner]:.4f}  '
              f'outer {outer:.4f}  (+side {outer_side:.4f})', flush=True)

    mean_inner = {name: float(np.mean([f['all_inner'][name] for f in fold_reports]))
                  for name in arms}
    selected = max(mean_inner, key=mean_inner.get)
    winners = [f['winner'] for f in fold_reports]

    print('\nmean inner macro-AUROC by arm:', flush=True)
    for name in sorted(mean_inner, key=mean_inner.get, reverse=True):
        mark = '  <- SELECTED' if name == selected else ''
        print(f'  {mean_inner[name]:.4f}  {arms[name]["label"]}{mark}', flush=True)

    nested_macro = float(np.mean([f['outer_macro'] for f in fold_reports]))
    nested_side = float(np.mean([f['outer_macro_with_side'] for f in fold_reports]))
    print(f'\nnested macro, AlphaGenome only {nested_macro:.4f}   with side features '
          f'{nested_side:.4f}', flush=True)
    print(f'folds agreeing on the selected arm: {winners.count(selected)} of {outer_k}', flush=True)
    print(f'config selects {cfg["alphagenome"]["selected"]!r}; the search selects '
          f'{selected!r}  ' + ('MATCH' if selected == cfg['alphagenome']['selected']
                               else '<-- MISMATCH, the pipeline is not running the winner'),
          flush=True)

    io.write_result('02_representation', args.name, {
        'protocol': f'RandomForest {RF["n_estimators"]} trees min_samples_leaf='
                    f'{RF["min_samples_leaf"]}, nested {outer_k}x{inner_k} CV on the training '
                    f'partition, macro-AUROC over {len(categories)} categories. Selection on the '
                    f'AlphaGenome block alone; side features carried through, never selected on.',
        'n_train': int(len(df)), 'n_arms': len(arms),
        'selected': selected, 'config_selected': cfg['alphagenome']['selected'],
        'folds_agreeing': winners.count(selected),
        'nested_macro_ag_only': nested_macro, 'nested_macro_with_side': nested_side,
        'mean_inner_by_arm': {arms[k]['label']: v for k, v in mean_inner.items()},
        'n_columns_by_arm': {arms[k]['label']: int(built[k].shape[1]) for k in arms},
        'folds': fold_reports,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
