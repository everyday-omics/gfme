#!/usr/bin/env python3
"""S4.3 — which classifier family?

Seven families on the selected feature set, each given several hyperparameter configurations so
that none of them loses on a default it was never meant to run with. The choice is made in the
inner loop and scored on the outer fold, so the reported figure is a nested estimate and not the
maximum of the values the choice was made on.

Reporting a flat best-of-N under a nested heading is a specific and easy mistake: the maximum of
several noisy estimates is biased upward by roughly the spread between them, which for a
close-run comparison is the whole margin. The flat figure is therefore reported too, as the
selection bias, rather than being quietly substituted for the nested one.

    python3 scripts/04_classifier_comparison.py
    python3 scripts/04_classifier_comparison.py --families catboost,xgboost
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import io, metrics, models, splits              # noqa: E402

# Several configurations per family, so a family is judged at its best rather than at its default.
FAMILIES = {
    'catboost': [
        {'iterations': 500, 'depth': 6, 'learning_rate': 0.05},
        {'iterations': 1000, 'depth': 6, 'learning_rate': 0.03},
        {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05},
        {'iterations': 1500, 'depth': 10, 'learning_rate': 0.03},
    ],
    'xgboost': [
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05},
        {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03},
        {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05},
    ],
    'lightgbm': [
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05},
        {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03},
        {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05},
    ],
    'random_forest': [
        {'n_estimators': 200, 'min_samples_leaf': 5, 'random_state': 42},
        {'n_estimators': 500, 'min_samples_leaf': 2, 'random_state': 42},
        {'n_estimators': 1000, 'min_samples_leaf': 1, 'random_state': 42},
    ],
    'logistic_regression': [
        {'C': 0.1, 'max_iter': 2000}, {'C': 1.0, 'max_iter': 2000},
        {'C': 10.0, 'max_iter': 2000},
    ],
    'knn': [{'n_neighbors': 5}, {'n_neighbors': 15}, {'n_neighbors': 50}],
    'mlp': [
        {'hidden_layer_sizes': (256, 64), 'max_iter': 400, 'random_state': 42},
        {'hidden_layer_sizes': (512, 128, 32), 'max_iter': 400, 'random_state': 42},
    ],
    # The torch networks and the two tabular foundation models. These need a GPU to be worth
    # running, so the step is submitted to a compute node rather than run on the head node.
    'flat_mlp': [{'lr': 1e-3}, {'lr': 3e-4}],
    'cnn': [{'lr': 1e-3}],
    'tabicl': [{'n_components': 50}, {'n_components': 100}, {'n_components': 200}],
    'tabpfn': [{'n_components': 50}, {'n_components': 100}, {'n_components': 200}],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='classifier_comparison')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    wanted = [f.strip() for f in args.families.split(',') if f.strip()]
    started = time.time()

    X, feature_names, df = io.load_matrix('clinvar_train', cfg)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    print(f'{X.shape[0]:,} training variants x {X.shape[1]} features\n', flush=True)

    outer_k, inner_k = cfg['cv']['outer_k'], cfg['cv']['inner_k']
    folds = splits.nested_folds(y, consequence, outer_k, inner_k, cfg['seed'])
    results = {}

    for family in wanted:
        candidates = {f'{family}[{i}]': {'family': family, 'hyperparameters': hp}
                      for i, hp in enumerate(FAMILIES[family])}
        print(f'=== {family} — {len(candidates)} configurations ===', flush=True)
        nested = models.nested(X, y, consequence, cfg, candidates, categories,
                               outer_k, inner_k, family)

        # The flat best-of-N over the same folds, for the selection-bias comparison.
        flat = {}
        for name, spec in candidates.items():
            predictions = np.full(len(y), np.nan)
            for train_index, test_index in splits.folds(y, consequence, outer_k, cfg['seed']):
                model = models.fit(X[train_index], y[train_index], cfg,
                                   consequence[train_index], spec['family'],
                                   spec['hyperparameters'])
                predictions[test_index] = models.predict(model, X[test_index])
            flat[name] = metrics.macro_auroc(y, predictions, consequence, categories,
                                             cfg['metrics']['min_cell'])
        flat_best = max(flat.values())
        nested['flat_best'] = flat_best
        nested['flat_by_candidate'] = flat
        nested['selection_bias'] = flat_best - nested['nested_macro']
        results[family] = nested
        print(f'  nested {nested["nested_macro"]:.4f}   flat best-of-N {flat_best:.4f}   '
              f'selection bias {nested["selection_bias"]:+.4f}   '
              f'{nested["folds_agreeing"]}/{outer_k} folds agree\n', flush=True)

    print('summary, by nested macro-AUROC:', flush=True)
    print(f'  {"family":22s}{"nested":>9s}{"flat":>9s}{"bias":>9s}{"agree":>8s}', flush=True)
    for family in sorted(results, key=lambda f: results[f]['nested_macro'], reverse=True):
        row = results[family]
        print(f'  {family:22s}{row["nested_macro"]:9.4f}{row["flat_best"]:9.4f}'
              f'{row["selection_bias"]:+9.4f}{row["folds_agreeing"]:>5d}/{outer_k}', flush=True)

    winner = max(results, key=lambda f: results[f]['nested_macro'])
    runner_up = sorted(results, key=lambda f: results[f]['nested_macro'], reverse=True)[1] \
        if len(results) > 1 else None
    if runner_up:
        margin = results[winner]['nested_macro'] - results[runner_up]['nested_macro']
        print(f'\n{winner} wins by {margin:.4f} over {runner_up}', flush=True)
        if margin < 0.005:
            print('  the margin is thin enough that the text should say so', flush=True)

    io.write_result('04_classifier_comparison', args.name, {
        'protocol': f'nested {outer_k}x{inner_k} cross-validation on the training partition, '
                    f'macro-AUROC over {len(categories)} categories. Every row is nested; the '
                    f'flat best-of-N is reported alongside as the selection bias, not in place '
                    f'of it.',
        'n_train': int(len(y)), 'n_features': int(X.shape[1]),
        'winner': winner, 'families': results,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
