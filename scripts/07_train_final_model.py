#!/usr/bin/env python3
"""Fit and freeze the ensemble, and produce the out-of-fold predictions it is calibrated on.

One model, fitted on the whole training partition, is what every later analysis applies: the
internal test table, the external cohort, the reclassification cohorts, the evidence tiers.
Refitting per analysis is how a paper ends up with several slightly different "the ensemble"
numbers that cannot be reconciled afterwards.

The out-of-fold predictions come from the same script because they must describe the same
configuration. They are the only honest in-training estimate — the resubstitution score on the
training matrix is not one and is never reported.

    python3 scripts/07_train_final_model.py
    python3 scripts/07_train_final_model.py --set alphagenome.selected=quantile_activity_max
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module    # noqa: E402
from gfme import io, metrics, models        # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clinvar_train',
                        help='training partition: clinvar_train or indel_train')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        help='dotted.key=value override, repeatable')
    parser.add_argument('--name', default=None, help='output name; defaults to the settings')
    parser.add_argument('--skip-oof', action='store_true')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    label = config_module.describe(cfg)
    name = args.name or label.replace('/', '_')
    print(f'configuration: {label}  ({config_module.fingerprint(cfg)})', flush=True)

    started = time.time()
    X, feature_names, df = io.load_matrix(args.dataset, cfg)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    categories = list(cfg['categories'][io.variant_type(args.dataset)])
    print(f'training partition: {X.shape[0]:,} variants x {X.shape[1]} features '
          f'({int((y == 1).sum()):,} P / {int((y == 0).sum()):,} B)\n', flush=True)

    result = {'configuration': label, 'dataset': args.dataset, 'n_train': int(len(y)), 'n_features': X.shape[1],
              'representation': cfg['alphagenome']['selected'],
              'sample_weights': cfg['model']['sample_weights'],
              'hyperparameters': config_module.as_dict(cfg['model']['hyperparameters'])}

    if not args.skip_oof:
        print(f'{cfg["cv"]["oof_k"]}-fold out-of-fold predictions...', flush=True)
        oof = models.oof(X, y, consequence, cfg)
        per_category, macro, counts = metrics.per_category(
            y, oof, consequence, categories, cfg['metrics']['min_cell'])
        low, high = metrics.bootstrap_ci(y, oof, consequence, categories,
                                         cfg['metrics']['min_cell'],
                                         cfg['metrics']['bootstrap_iterations'],
                                         cfg['metrics']['confidence_level'], cfg['seed'])
        print(f'  OOF macro-AUROC {macro:.4f}  [{low:.4f}, {high:.4f}]', flush=True)
        for category in categories:
            if category in per_category:
                print(f'    {category:12s} {per_category[category]:.4f}  '
                      f'n={counts[category]:,}', flush=True)
        result['oof'] = {'macro': macro, 'ci': [low, high], 'per_category': per_category,
                         'n_by_category': counts, 'k': cfg['cv']['oof_k']}
        out = io.results_dir('07_final_model') / f'oof_{name}.npz'
        np.savez_compressed(out, keys=df['key'].to_numpy(dtype=object), oof=oof, label=y,
                            consequence=consequence)
        print(f'  -> {out.name}', flush=True)

    print('\nfitting the frozen model on the whole partition...', flush=True)
    model = models.fit(X, y, cfg, consequence)
    path = models.save(model, Path(__file__).resolve().parent.parent / 'results' /
                       '07_final_model' / f'ensemble_{name}.cbm', cfg, feature_names,
                       f'{args.dataset} (n={len(y)})')
    print(f'  -> {path.name}', flush=True)

    result['model'] = path.name
    io.write_result('07_final_model', f'summary_{name}', result, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
