#!/usr/bin/env python3
"""S5 — Evo2 model size, and the ablation.

Two questions the ensemble invites.

**Does the smaller Evo2 cost anything?** The pre-scored database uses Evo2-1B rather than the 7B
model used in development, so that inference fits on commodity hardware. Uncalibrated the two
differ; the question is whether that difference survives the ensemble.

**Is any component dispensable?** If the classifier can compensate for a missing model, that model
is redundant and the ensemble is more complicated than it needs to be. Every meaningful subset is
fitted and evaluated identically, including the best combination that contains no AlphaGenome, so
the comparison is against the strongest alternative rather than against nothing.

Both are evaluated on the temporal test set with models fitted on the training partition only.

    python3 scripts/10_evo2_size.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                       # noqa: E402
from gfme import consequences, features, io, metrics, models   # noqa: E402

BLOCKS = ('ag', 'gpnmsa', 'evo2', 'borzoi')


def build(dataset, config, include, evo2_column='evo2'):
    """A model matrix from a chosen set of blocks, plus the consequence encoding."""
    df = io.load_dataset(dataset)
    keys = df['key'].tolist()
    scores = io.load_scores(dataset, keys)
    parts, names = [], []

    if 'ag' in include:
        raw, feature_names, _ = io.load_alphagenome(dataset, keys)
        quantile = io.load_alphagenome(dataset, keys, layer='quantile')[0] \
            if config['alphagenome']['arms'][config['alphagenome']['selected']].get(
                'biosample') == 'quantile' else None
        ag, ag_names = features.build_selected(raw, feature_names, config, quantile)
        parts.append(ag)
        names.extend(ag_names)
    if 'borzoi' in include:
        block, _ = io.load_block(dataset, 'borzoi', keys)
        parts.append(block)
        names.extend([f'borzoi_{i}' for i in range(block.shape[1])])
    if 'gpnmsa' in include:
        parts.append((-scores['gpnmsa'].to_numpy(float)).reshape(-1, 1))
        names.append('gpnmsa_neg')
    if 'evo2' in include:
        parts.append((-scores[evo2_column].to_numpy(float)).reshape(-1, 1))
        names.append('evo2_neg')

    cons = np.array([consequences.encode(c, 'snv') for c in df['consequence']], dtype=float)
    parts.append(cons.reshape(-1, 1))
    names.append('cons_num')
    X = np.nan_to_num(np.hstack(parts), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, names, df


def fit_and_score(include, config, categories, evo2_column='evo2', label='', mask=None):
    """Fit on the training partition, score the test set.

    `mask` restricts the evaluation to a subset. It matters for the Evo2 size comparison: the 1B
    model was scored on the benchmark variants only, so evaluating the two sizes over different
    variant sets would confound model size with cohort, and letting the unscored 85% fall through
    as zeros would measure the missing column rather than the smaller model.
    """
    X_train, _, train = build('clinvar_train', config, include, evo2_column)
    X_test, _, test = build('clinvar_test', config, include, evo2_column)
    model = models.fit(X_train, train['label'].to_numpy(int), config,
                       train['consequence'].to_numpy())
    scores = models.predict(model, X_test)
    if mask is not None:
        scores = np.where(mask, scores, np.nan)
    macro = metrics.macro_auroc(test['label'].to_numpy(int), scores,
                                test['consequence'].to_numpy(), categories,
                                config['metrics']['min_cell'])
    print(f'  {label:44s} {X_train.shape[1]:6,} features   {macro:.4f}', flush=True)
    return macro, int(X_train.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='evo2_size_and_ablation')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    min_cell = cfg['metrics']['min_cell']
    started = time.time()

    test = io.load_dataset('clinvar_test')
    test_scores = io.load_scores('clinvar_test', test['key'].tolist())
    y = test['label'].to_numpy(int)
    consequence = test['consequence'].to_numpy()

    print(f'=== Evo2 model size, on the {len(test):,}-variant temporal test set ===', flush=True)
    # Evo2-1B was scored on the benchmark variants only, so the two sizes are compared on the
    # subset carrying both. Comparing them on different variant sets would confound the model
    # size with the cohort.
    columns = [c for c in ('evo2', 'evo2_1b') if c in test_scores.columns]
    both = np.ones(len(test), bool)
    for column in columns:
        both &= np.isfinite(test_scores[column].to_numpy(float))
    print(f'  {int(both.sum()):,} of {len(test):,} variants carry both model sizes', flush=True)

    zero_shot = {}
    for size, column in (('Evo2-7B', 'evo2'), ('Evo2-1B', 'evo2_1b')):
        if column not in test_scores.columns:
            print(f'  {size}: no {column} column for this cohort — skipped', flush=True)
            continue
        values = np.where(both, -test_scores[column].to_numpy(float), np.nan)
        zero_shot[size] = metrics.macro_auroc(y, values, consequence, categories, min_cell)
        print(f'  {size} zero-shot macro {zero_shot[size]:.4f}', flush=True)

    ensemble = {}
    train_columns = io.load_scores('clinvar_train').columns
    for size, column in (('Evo2-7B', 'evo2'), ('Evo2-1B', 'evo2_1b')):
        if column not in train_columns or column not in test_scores.columns:
            continue
        ensemble[size], _ = fit_and_score(('ag', 'gpnmsa', 'evo2'), cfg, categories, column,
                                          f'ensemble with {size}', mask=both)
    if len(zero_shot) == 2 and len(ensemble) == 2:
        print(f'\n  delta (1B - 7B): zero-shot '
              f'{zero_shot["Evo2-1B"] - zero_shot["Evo2-7B"]:+.4f}   ensemble '
              f'{ensemble["Evo2-1B"] - ensemble["Evo2-7B"]:+.4f}', flush=True)

    print(f'\n=== ablation, same test set ===', flush=True)
    configurations = {
        'AG + Evo2 + GPN-MSA (full ensemble)': ('ag', 'gpnmsa', 'evo2'),
        'AG only': ('ag',),
        'Evo2 + GPN-MSA (no AG)': ('gpnmsa', 'evo2'),
        'Borzoi + Evo2 + GPN-MSA (best non-AG)': ('borzoi', 'gpnmsa', 'evo2'),
        'Evo2 only': ('evo2',),
        'AG + GPN-MSA (no Evo2)': ('ag', 'gpnmsa'),
        'GPN-MSA only': ('gpnmsa',),
        'AG + Evo2 (no GPN-MSA)': ('ag', 'evo2'),
    }
    ablation = {}
    for label, include in configurations.items():
        try:
            macro, n_features = fit_and_score(include, cfg, categories, 'evo2', label)
        except io.MissingData as missing:
            # Never swallow this. A configuration that cannot be built is a missing table row, and
            # a printed `skipped` inside a long log is indistinguishable from success: the Borzoi
            # row of this table disappeared that way and stayed gone through a full re-run.
            raise SystemExit(
                f'{label}: cannot be built — {missing}\n'
                f'Score the missing block, or drop the configuration from CONFIGURATIONS '
                f'deliberately. Do not leave it to fail at run time.') from missing
        ablation[label] = {'macro': macro, 'n_features': n_features,
                           'blocks': list(include)}

    full = ablation['AG + Evo2 + GPN-MSA (full ensemble)']['macro']
    print('\n  cost of removing each component:', flush=True)
    for label, without in (('AlphaGenome', 'Evo2 + GPN-MSA (no AG)'),
                           ('Evo2', 'AG + GPN-MSA (no Evo2)'),
                           ('GPN-MSA', 'AG + Evo2 (no GPN-MSA)')):
        if without in ablation:
            print(f'    {label:12s} {ablation[without]["macro"] - full:+.4f}', flush=True)

    io.write_result('10_evo2_size', args.name, {
        'evaluation_set': f'temporal test set, n={len(test)}',
        'training_set': 'clinvar_train',
        'representation': cfg['alphagenome']['selected'],
        'hyperparameters': config_module.as_dict(cfg['model']['hyperparameters']),
        'evo2_zero_shot': zero_shot, 'evo2_ensemble': ensemble,
        'ablation': ablation,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
