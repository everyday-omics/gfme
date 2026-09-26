#!/usr/bin/env python3
"""The reclassification cohorts.

Variants that were uncertain (VUS) or had conflicting interpretations in January 2023, and that
subsequent curation resolved to pathogenic or benign by February 2026. Scoring them with a model
trained only on pre-2023 data asks the question the clinic actually asks: given what was known
then, would this tool have pointed curation the right way?

Neither cohort contributed to training, and both are scored by the frozen model without refitting.

SNVs and indels are scored by their own locked model over their own category set, so the two are
run as separate invocations rather than pooled: an AUROC over a mixed cohort would average two
different models.

    python3 scripts/12_reclassification.py --model results/07_final_model/ensemble_<name>.cbm
    python3 scripts/12_reclassification.py --model results/07_final_model/ensemble_indel.cbm \
        --variant-type indel --name reclassification_indel
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                 # noqa: E402
from gfme import io, metrics, models, splits, tables     # noqa: E402

COHORTS = {
    'snv': {'reclass_vus': 'resolved from VUS',
            'reclass_conflicting': 'resolved from conflicting interpretations'},
    'indel': {'reclass_vus_indel': 'resolved from VUS',
              'reclass_conflicting_indel': 'resolved from conflicting interpretations'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='reclassification')
    parser.add_argument('--variant-type', default='snv', choices=('snv', 'indel'))
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories'][args.variant_type])
    cohorts = COHORTS[args.variant_type]
    training = 'clinvar_train' if args.variant_type == 'snv' else 'indel_train'
    min_cell = cfg['metrics']['min_cell']
    print(f'configuration: {config_module.describe(cfg)}\n', flush=True)

    training_keys = io.load_dataset(training)['key'].tolist()
    result = {'model': args.model, 'variant_type': args.variant_type, 'cohorts': {}}
    pooled = {'scores': [], 'labels': [], 'consequence': []}

    for cohort, description in cohorts.items():
        X, feature_names, df = io.load_matrix(cohort, cfg)
        splits.assert_no_leakage(training_keys, df['key'], cohort)
        model, _ = models.load(args.model, cfg, feature_names)
        scores = models.predict(model, X)
        y = df['label'].to_numpy(int)
        consequence = df['consequence'].to_numpy()

        rows = {'FM Ensemble': scores}
        rows.update(tables.tool_scores(cohort, cfg)[0])
        evaluated = metrics.evaluate(y, rows, consequence, categories, min_cell)
        low, high = metrics.bootstrap_ci(y, scores, consequence, categories, min_cell,
                                         cfg['metrics']['bootstrap_iterations'],
                                         cfg['metrics']['confidence_level'], cfg['seed'])

        print(f'=== {cohort} — {description} ===', flush=True)
        print(f'{len(df):,} variants ({int((y == 1).sum()):,} resolved pathogenic / '
              f'{int((y == 0).sum()):,} resolved benign)', flush=True)
        print(f'  ensemble macro-AUROC {evaluated["FM Ensemble"]["macro"]:.4f} '
              f'[{low:.4f}, {high:.4f}]', flush=True)
        for category in categories:
            value = evaluated['FM Ensemble']['per_category'].get(category)
            if value is not None:
                print(f'    {category:12s} {value:.4f}  '
                      f'n={evaluated["FM Ensemble"]["n_scored"][category]:,}', flush=True)
        for label in ('CADD', 'GPN-MSA', 'Evo2', 'phyloP'):
            if label in evaluated and evaluated[label]['macro'] is not None:
                print(f'  {label:12s} {evaluated[label]["macro"]:.4f}', flush=True)
        print(flush=True)

        result['cohorts'][cohort] = {
            'description': description, 'n': int(len(df)),
            'n_pathogenic': int((y == 1).sum()), 'n_benign': int((y == 0).sum()),
            'ensemble_ci': [low, high], 'rows': evaluated,
        }
        pooled.setdefault('rows', []).append(rows)
        pooled['scores'].append(scores)
        pooled['labels'].append(y)
        pooled['consequence'].append(consequence)

    scores = np.concatenate(pooled['scores'])
    y = np.concatenate(pooled['labels'])
    consequence = np.concatenate(pooled['consequence'])
    shared = set.intersection(*[set(r) for r in pooled['rows']])
    pooled_rows = {label: np.concatenate([r[label] for r in pooled['rows']]) for label in shared}
    per_category, macro, counts = metrics.per_category(y, scores, consequence, categories,
                                                       min_cell)
    low, high = metrics.bootstrap_ci(y, scores, consequence, categories, min_cell,
                                     cfg['metrics']['bootstrap_iterations'],
                                     cfg['metrics']['confidence_level'], cfg['seed'])
    print(f'=== both cohorts, {len(y):,} variants ===', flush=True)
    print(f'  ensemble macro-AUROC {macro:.4f} [{low:.4f}, {high:.4f}]', flush=True)
    for category in categories:
        if category in per_category:
            print(f'    {category:12s} {per_category[category]:.4f}  '
                  f'n={counts[category]:,}', flush=True)

    evaluated = metrics.evaluate(y, pooled_rows, consequence, categories, min_cell)
    order = [r for r in cfg['table_row_order'][args.variant_type] if r in evaluated]
    order += [r for r in evaluated if r not in order]
    result['pooled'] = {'n': int(len(y)), 'macro': macro, 'ci': [low, high],
                        'per_category': per_category, 'n_by_category': counts,
                        'categories': categories, 'row_order': order, 'rows': evaluated}
    path = io.write_result('12_reclassification', args.name, result, cfg)
    print(f'\n-> {path}', flush=True)


if __name__ == '__main__':
    main()
