#!/usr/bin/env python3
"""S4.2 — which foundation models belong in the ensemble?

All 2^9 - 1 = 511 non-empty subsets of the nine models, each scored by the same Random Forest
under the same folds, so the comparison is between feature sets and nothing else. Exhaustive
rather than greedy: forward selection would commit to whichever model looks best alone, and the
question is whether a model earns its place *given the others*, which a greedy path never asks.

The consequence encoding is deliberately absent. It is not a foundation model, and including it
would lift every subset by roughly the same amount while compressing the differences the search
exists to measure.

Models contribute what they actually emit: AlphaGenome its 747-column aggregate, Borzoi its
7,611-track log2 fold change, the five embedding models their allele-difference vectors, and
GPN-MSA and Evo2 a single score each.

    python3 scripts/03_fm_combination.py
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import features, io, metrics, models, splits    # noqa: E402

RF = {'n_estimators': 200, 'min_samples_leaf': 5, 'random_state': 42}
SCALAR = {'gpnmsa', 'evo2'}


def load_blocks(config, dataset='clinvar_train'):
    """One feature block per foundation model, all keyed to the same variant order."""
    df = io.load_dataset(dataset)
    keys = df['key'].tolist()
    scores = io.load_scores(dataset, keys)
    blocks = {}

    raw, feature_names, _ = io.load_alphagenome(dataset, keys)
    quantile, _, _ = io.load_alphagenome(dataset, keys, layer='quantile')
    ag, _ = features.build_selected(raw, feature_names, config, quantile)
    blocks['ag'] = ag

    for name in config['foundation_models']:
        if name == 'ag':
            continue
        if name in SCALAR:
            blocks[name] = (-scores[name].to_numpy(dtype=float)).reshape(-1, 1)
        else:
            blocks[name] = io.load_block(dataset, name, keys)[0]

    for name, block in blocks.items():
        blocks[name] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return blocks, df


def merge(config, name):
    """Combine per-fold files into the result the single-job run would have written."""
    import json

    directory = io.results_dir('03_fm_combination')
    paths = sorted(directory.glob(f'{name}_fold*.json'))
    if not paths:
        raise SystemExit(f'no {name}_fold*.json in {directory}')
    reports = []
    for path in paths:
        with open(path) as handle:
            reports.append(json.load(handle)['fold'])
    reports.sort(key=lambda r: r['fold'])
    print(f'merging {len(reports)} folds: {[r["fold"] for r in reports]}', flush=True)

    mean_inner = {subset: float(np.mean([r['all_inner'][subset] for r in reports]))
                  for subset in reports[0]['all_inner']}
    selected = max(mean_inner, key=mean_inner.get)
    winners = [r['winner'] for r in reports]
    outer_scores = [r['outer_macro'] for r in reports]
    nested_macro = float(np.mean(outer_scores))
    print(f'nested macro {nested_macro:.4f}; selected by mean inner {selected}; '
          f'{winners.count(selected)}/{len(reports)} folds agree', flush=True)
    path = io.write_result('03_fm_combination', name, {
        'protocol': f'RandomForest {RF["n_estimators"]} trees min_samples_leaf='
                    f'{RF["min_samples_leaf"]}, nested '
                    f'{config["cv"]["outer_k"]}x{config["cv"]["inner_k"]} over the subset search '
                    f'itself on the training partition, macro-AUROC over '
                    f'{len(config["categories"]["snv"])} categories. No consequence encoding. '
                    f'Folds were run as separate jobs and merged.',
        'n_subsets': len(reports[0]['all_inner']),
        'nested_macro': nested_macro, 'outer_scores': outer_scores,
        'selected_by_mean_inner': selected, 'folds_agreeing': winners.count(selected),
        'mean_inner_by_subset': mean_inner, 'folds': reports,
    }, config)
    print(f'-> {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='fm_combination')
    # Score a named handful of subsets instead of the full 511, e.g. after changing one model's
    # input. Same code path, same folds, same metric, so the numbers drop straight into the full
    # table; the marginal contributions are skipped, since they are defined over all of it.
    parser.add_argument('--only', default='',
                        help="comma-separated combinations, e.g. 'borzoi,borzoi+gpnmsa+evo2'")
    # The search omits the consequence encoding so that a subset is scored on what its models
    # contribute and nothing else. The earlier standalone scripts included it in every arm, so
    # this flag exists to compare against those numbers on equal terms rather than by assertion.
    parser.add_argument('--with-consequence', action='store_true')
    # Nest the subset search itself: an inner loop picks the subset, an untouched outer fold
    # scores it. Costs one search per outer fold, so about five times the flat run.
    parser.add_argument('--nested', action='store_true')
    # The outer folds are independent, so each can run as its own job and the five are merged
    # afterwards. One fold at a time also lets every core go to the forest rather than sitting
    # idle behind a serial loop: `--fold i` writes <name>_fold<i>.json, `--merge` combines them.
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    started = time.time()

    if args.merge:
        merge(cfg, args.name)
        return

    blocks, df = load_blocks(cfg)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    names = list(cfg['foundation_models'])
    print(f'{len(df):,} training variants', flush=True)
    for name in names:
        print(f'  {name:10s} {blocks[name].shape[1]:6,} columns', flush=True)

    pairs = splits.folds(y, consequence, cfg['cv']['oof_k'], cfg['seed'])
    subsets = [c for size in range(1, len(names) + 1)
               for c in itertools.combinations(names, size)]
    if args.only:
        wanted = {frozenset(c.split('+')) for c in args.only.split(',') if c.strip()}
        unknown = {m for c in wanted for m in c} - set(names)
        if unknown:
            raise SystemExit(f'unknown model(s) in --only: {sorted(unknown)}')
        subsets = [s for s in subsets if frozenset(s) in wanted]
        if len(subsets) != len(wanted):
            raise SystemExit(f'--only matched {len(subsets)} of {len(wanted)} combinations')
    print(f'\n{len(subsets)} subsets x {len(pairs)} folds\n', flush=True)

    extra = []
    if args.with_consequence:
        from gfme import consequences
        vtype = 'snv'
        extra = [np.array([consequences.encode(c, vtype) for c in consequence],
                          dtype=np.float32).reshape(-1, 1)]
        print('  + consequence encoding (1 column)', flush=True)

    if args.nested:
        outer_k, inner_k = cfg['cv']['outer_k'], cfg['cv']['inner_k']
        folds = splits.nested_folds(y, consequence, outer_k, inner_k, cfg['seed'])
        wanted = range(outer_k) if args.fold is None else [args.fold]
        print(f'nested {outer_k}x{inner_k} over {len(subsets)} subsets, '
              f'folds {list(wanted)}\n', flush=True)
        reports, outer_scores = [], []
        for i in wanted:
            fold = folds[i]
            inner = {}
            for index, subset in enumerate(subsets, 1):
                X = np.hstack([blocks[n] for n in subset] + extra)
                values = []
                for inner_train, inner_test in fold['inner']:
                    model = models.fit(X[inner_train], y[inner_train], cfg,
                                       consequence[inner_train], 'random_forest', RF)
                    values.append(metrics.macro_auroc(
                        y[inner_test], models.predict(model, X[inner_test]),
                        consequence[inner_test], categories, cfg['metrics']['min_cell']))
                inner['+'.join(subset)] = float(np.mean([v for v in values if v is not None]))
                if index % 100 == 0:
                    print(f'  fold {i}: {index}/{len(subsets)} '
                          f'({(time.time() - started) / 60:.0f} min)', flush=True)
            winner = max(inner, key=inner.get)
            X = np.hstack([blocks[n] for n in winner.split('+')] + extra)
            model = models.fit(X[fold['train']], y[fold['train']], cfg,
                               consequence[fold['train']], 'random_forest', RF)
            outer = metrics.macro_auroc(y[fold['test']], models.predict(model, X[fold['test']]),
                                        consequence[fold['test']], categories,
                                        cfg['metrics']['min_cell'])
            outer_scores.append(outer)
            reports.append({'fold': i, 'winner': winner, 'inner_macro': inner[winner],
                            'outer_macro': outer, 'all_inner': inner})
            print(f'  fold {i}: selected {winner} inner {inner[winner]:.4f} '
                  f'outer {outer:.4f}', flush=True)

        if args.fold is not None:
            path = io.write_result('03_fm_combination', f'{args.name}_fold{args.fold}',
                                   {'fold': reports[0]}, cfg)
            print(f'\n-> {path}   ({(time.time() - started) / 60:.1f} min)', flush=True)
            return

        mean_inner = {name: float(np.mean([r['all_inner'][name] for r in reports]))
                      for name in reports[0]['all_inner']}
        selected = max(mean_inner, key=mean_inner.get)
        winners = [r['winner'] for r in reports]
        nested_macro = float(np.mean(outer_scores))
        print(f'\nnested macro {nested_macro:.4f}; selected by mean inner {selected}; '
              f'{winners.count(max(set(winners), key=winners.count))}/{outer_k} folds agree',
              flush=True)
        io.write_result('03_fm_combination', args.name, {
            'protocol': f'RandomForest {RF["n_estimators"]} trees min_samples_leaf='
                        f'{RF["min_samples_leaf"]}, nested {outer_k}x{inner_k} over the subset '
                        f'search itself on the training partition, macro-AUROC over '
                        f'{len(categories)} categories.'
                        + (' Consequence encoding included.' if args.with_consequence else
                           ' No consequence encoding.'),
            'n_train': int(len(df)), 'n_subsets': len(subsets),
            'nested_macro': nested_macro, 'outer_scores': outer_scores,
            'selected_by_mean_inner': selected, 'folds_agreeing': winners.count(selected),
            'mean_inner_by_subset': mean_inner, 'folds': reports,
        }, cfg)
        print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)
        return

    rows = []
    for index, subset in enumerate(subsets, 1):
        X = np.hstack([blocks[n] for n in subset] + extra)
        predictions = np.full(len(y), np.nan)
        for train_index, test_index in pairs:
            model = models.fit(X[train_index], y[train_index], cfg, consequence[train_index],
                               'random_forest', RF)
            predictions[test_index] = models.predict(model, X[test_index])
        macro = metrics.macro_auroc(y, predictions, consequence, categories,
                                    cfg['metrics']['min_cell'])
        rows.append({'combination': '+'.join(subset), 'n_models': len(subset),
                     'n_features': int(X.shape[1]), 'macro': macro})
        if index % 25 == 0 or index == len(subsets):
            best = max(rows, key=lambda r: r['macro'])
            print(f'  {index:3d}/{len(subsets)}  best so far {best["combination"]} '
                  f'{best["macro"]:.4f}  ({(time.time() - started) / 60:.0f} min)', flush=True)

    rows.sort(key=lambda r: r['macro'], reverse=True)
    print('\ntop 10 combinations:', flush=True)
    for row in rows[:10]:
        print(f'  {row["macro"]:.4f}  {row["combination"]:52s} '
              f'{row["n_features"]:6,} features', flush=True)

    print('\neach model on its own:', flush=True)
    alone = {r['combination']: r['macro'] for r in rows if r['n_models'] == 1}
    for name in sorted(alone, key=alone.get, reverse=True):
        print(f'  {alone[name]:.4f}  {name}', flush=True)

    # Two definitions, because they can disagree in sign and the supplement reports one of them.
    # `marginal_contribution` is the mean change in macro-AUROC when the model is added, taken over
    # every non-empty subset of the other eight — the statistic the supplement table gives.
    # `marginal_best_minus_best` is how much the best subset containing the model beats the best
    # subset without it, which answers redundancy instead. Writing only the second is what put a
    # figure in the results file that the text contradicted.
    # Both are defined over the whole search: skipped when only a handful of subsets were scored.
    marginal, best_minus_best, improved = {}, {}, {}
    if args.only:
        print('\nmarginal contribution: skipped (--only)', flush=True)
        names = []
    else:
        print('\nmarginal contribution:', flush=True)
    macro_by_subset = {frozenset(r['combination'].split('+')): r['macro'] for r in rows}
    for name in names:
        others = [n for n in names if n != name]
        deltas = [macro_by_subset[frozenset(subset) | {name}] - macro_by_subset[frozenset(subset)]
                  for size in range(1, len(others) + 1)
                  for subset in itertools.combinations(others, size)]
        marginal[name] = sum(deltas) / len(deltas)
        improved[name] = sum(1 for d in deltas if d > 0)
        with_model = max(r['macro'] for r in rows if name in r['combination'].split('+'))
        without = max(r['macro'] for r in rows if name not in r['combination'].split('+'))
        best_minus_best[name] = with_model - without
        print(f'  {marginal[name]:+.4f}  {name:10s} mean over {len(deltas)} subsets, improves '
              f'{improved[name]}; best-minus-best {best_minus_best[name]:+.4f}', flush=True)

    winner = rows[0]
    expected = '+'.join(cfg['selected_combination'])
    print(f'\nwinner {winner["combination"]} {winner["macro"]:.4f}; config expects {expected}  '
          + ('MATCH' if winner['combination'] == expected else '<-- MISMATCH'), flush=True)

    io.write_result('03_fm_combination', args.name, {
        'protocol': f'RandomForest {RF["n_estimators"]} trees min_samples_leaf='
                    f'{RF["min_samples_leaf"]}, {cfg["cv"]["oof_k"]}-fold out-of-fold on the '
                    f'training partition, macro-AUROC over {len(categories)} categories. No '
                    f'consequence encoding.',
        'n_train': int(len(df)), 'n_subsets': len(subsets),
        'winner': winner, 'config_expects': expected,
        'single_model': alone, 'marginal_contribution': marginal,
        'marginal_best_minus_best': best_minus_best, 'marginal_improved': improved,
        'marginal_definition': 'mean change in macro-AUROC when the model is added, over all '
                               'non-empty subsets of the remaining eight',
        'n_columns_by_model': {n: int(blocks[n].shape[1]) for n in names},
        'all_combinations': rows,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
