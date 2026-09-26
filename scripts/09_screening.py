#!/usr/bin/env python3
"""Tables 1 and S1 — the nine foundation models against the established tools.

The comparison that motivates everything after it. Every model is scored the way it is meant to
be used: as a single number, zero-shot, except AlphaGenome and Borzoi, which emit multi-track
molecular predictions rather than a variant score and publish no recommended aggregation, so they
get a supervised out-of-fold random forest. That follows the AlphaGenome authors, who likewise
evaluate on ClinVar by fitting a random forest to their predicted molecular effects.

Table 1 reports per-category and macro AUROC on the whole pre-2023 partition. AUROC does not
depend on class prevalence, so the partition is used as it stands rather than balanced within
category. Table S1 reports pooled alongside macro on a random 10,000 of the same snapshot at its
natural composition, where the two columns rank the models differently; that divergence is the
reason the article reports macro throughout.

    python3 scripts/09_screening.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import io, metrics, models                     # noqa: E402
from gfme import features                                 # noqa: E402

RF = {'n_estimators': 200, 'min_samples_leaf': 5, 'random_state': 42}
LABELS = {'gpnmsa': 'GPN-MSA', 'evo2': 'Evo2', 'dnabert2': 'DNABERT-2', 'ntv2': 'NT-v2',
          'hyenadna': 'HyenaDNA', 'grover': 'GROVER', 'caduceus': 'Caduceus-Ph',
          'cadd': 'CADD', 'phylop': 'phyloP', 'gerp': 'GERP++', 'dann': 'DANN',
          'phastcons': 'phastCons', 'borzoi': 'Borzoi-L2'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='screening')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    min_cell = cfg['metrics']['min_cell']
    started = time.time()

    full = io.load_dataset('clinvar_pre2023')
    # AUROC does not depend on class prevalence, so the per-category comparison uses every
    # variant in the partition. Balancing within category would discard several hundred for no
    # gain in the metric; it belongs to model training, not to this table.
    df = full
    keys = df['key'].tolist()
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    print(f'Table 1 — pre-2023 partition, {len(df):,} variants '
          f'({int((y == 1).sum()):,} P / {int((y == 0).sum()):,} B)\n', flush=True)

    scores = io.load_scores('clinvar_pre2023', keys)
    rows = {}
    for column, label in LABELS.items():
        if column not in scores.columns:
            continue
        values = scores[column].to_numpy(dtype=float)
        if column in ('gpnmsa', 'evo2'):
            values = -values                       # distributed higher-is-more-benign
        rows[label] = values

    # The two supervised rows. Out-of-fold, so no variant is scored by a model that saw it.
    raw, feature_names, _ = io.load_alphagenome('clinvar_pre2023', keys)
    quantile, _, _ = io.load_alphagenome('clinvar_train', keys, layer='quantile')
    ag, _ = features.build_selected(raw, feature_names, cfg, quantile)
    ag = np.nan_to_num(ag, nan=0.0, posinf=0.0, neginf=0.0)
    rows['AG-RF (OOF)'] = models.oof(ag, y, consequence, cfg, family='random_forest',
                                     hyperparameters=RF)
    borzoi, _ = io.load_block('clinvar_pre2023', 'borzoi', keys)
    borzoi = np.nan_to_num(borzoi, nan=0.0, posinf=0.0, neginf=0.0)
    rows['Borzoi-RF (OOF)'] = models.oof(borzoi, y, consequence, cfg, family='random_forest',
                                         hyperparameters=RF)

    evaluated = metrics.evaluate(y, rows, consequence, categories, min_cell)
    order = sorted(evaluated, key=lambda k: evaluated[k]['macro'] or 0, reverse=True)
    short = {'intronic': 'Intr.', 'synonymous': 'Syn.', 'non-coding': 'Non-cod.',
             'splice': 'Splice', 'LoF': 'LoF', 'other': 'Other', 'missense': 'Miss.'}
    print(f'{"":18s}{"Macro":>8s}' + ''.join(f'{short[c]:>10s}' for c in categories), flush=True)
    for label in order:
        row = evaluated[label]
        cells = ''.join(f'{row["per_category"][c]:10.3f}' if c in row['per_category']
                        else f'{"---":>10s}' for c in categories)
        print(f'{label:18s}{row["macro"]:8.3f}{cells}', flush=True)

    # S1 — the same models on a random 10,000 of the pre-2023 snapshot at its natural class and
    # category composition, where pooled and macro rank the models differently.
    screen = io.load_dataset('clinvar_screen10k')
    screen_keys = screen['key'].tolist()
    screen_y = screen['label'].to_numpy(int)
    screen_consequence = screen['consequence'].to_numpy()
    screen_scores = io.load_scores('clinvar_screen10k', screen_keys)

    screen_rows = {}
    for column, label in LABELS.items():
        if column not in screen_scores.columns:
            continue
        values = screen_scores[column].to_numpy(dtype=float)
        if column in ('gpnmsa', 'evo2'):
            values = -values
        screen_rows[label] = values
    # AG-RF is carried as a score column, fitted on this sample; Borzoi-RF is refitted here.
    if 'ag_rf' in screen_scores.columns:
        screen_rows['AG-RF (OOF)'] = screen_scores['ag_rf'].to_numpy(dtype=float)
    screen_borzoi, _ = io.load_block('clinvar_screen10k', 'borzoi', screen_keys)
    screen_borzoi = np.nan_to_num(screen_borzoi, nan=0.0, posinf=0.0, neginf=0.0)
    screen_rows['Borzoi-RF (OOF)'] = models.oof(screen_borzoi, screen_y, screen_consequence,
                                                cfg, family='random_forest', hyperparameters=RF)

    screen_min_cell = 2
    screened = {}
    for label, values in screen_rows.items():
        per, macro, _ = metrics.per_category(screen_y, values, screen_consequence,
                                             categories, screen_min_cell)
        screened[label] = {'pooled': metrics.auroc(screen_y, values),
                           'macro': macro, 'per_category': per}
    minority = {c: int(min((screen_y[screen_consequence == c] == 1).sum(),
                           (screen_y[screen_consequence == c] == 0).sum()))
                for c in categories}

    print(f'\nS1 — {len(screen):,} variants at natural composition '
          f'({int(screen_y.sum()):,} P / {int((1 - screen_y).sum()):,} B)', flush=True)
    print('minority class per category: '
          + ', '.join(f'{c} {n}' for c, n in minority.items()), flush=True)
    print(f'{"":18s}{"Pooled":>9s}{"Macro":>9s}', flush=True)
    for label in sorted(screened, key=lambda k: screened[k]['pooled'] or 0, reverse=True):
        row = screened[label]
        macro = row['macro'] if row['macro'] is not None else float('nan')
        print(f'{label:18s}{row["pooled"]:9.3f}{macro:9.3f}', flush=True)

    io.write_result('09_screening', args.name, {
        'protocol': 'Table 1: the pre-2023 partition, every variant, per-category AUROC. '
                    'Table S1: a random 10,000 of the same snapshot at its natural composition, '
                    'pooled and macro. Zero-shot except AG-RF and Borzoi-RF, which are '
                    'out-of-fold random forests',
        'n_full': int(len(full)), 'n_evaluated': int(len(df)),
        'categories': categories, 'rows': evaluated,
        'screening_10k': {'n': int(len(screen)), 'n_pathogenic': int(screen_y.sum()),
                          'minority_per_category': minority, 'rows': screened},
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
