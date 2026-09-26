#!/usr/bin/env python3
"""Every figure, from the result files the earlier steps wrote.

No figure recomputes an AUROC. Each reads the JSON that the analysis step produced, so a number
in a figure and the same number in a table cannot disagree — which is exactly what happened when
figure scripts held their values as literals in their own source and could not be re-run after a
model changed.

The one exception is the indel size-and-type stratification, which is a figure and nothing else:
it is computed here and written to a result file alongside the plot.

    python3 scripts/13_figures.py --model results/07_final_model/ensemble_indel.cbm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import figures, io, metrics, models             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'figures'


def read(step, name):
    path = ROOT / 'results' / step / f'{name}.json'
    if not path.exists():
        print(f'  skipped — {path.relative_to(ROOT)} not found', flush=True)
        return None
    return json.loads(path.read_text())


def size_and_type(config, model_path):
    """Indel AUROC by length and by insertion against deletion.

    Length is binned rather than plotted per base: single-base bins beyond about ten hold too few
    variants to carry an AUROC, and a curve drawn through them would show noise as structure.
    """
    X, feature_names, df = io.load_matrix('indel_test', config)
    model, _ = models.load(model_path, config, feature_names)
    scores = models.predict(model, X)
    y = df['label'].to_numpy(int)

    length = (df['ref'].str.len() - df['alt'].str.len()).abs().to_numpy()
    is_insertion = (df['alt'].str.len() > df['ref'].str.len()).to_numpy()
    edges = [(1, 1), (2, 2), (3, 3), (4, 10), (11, 50)]
    strata = ['1', '2', '3', '4-10', '11-50']

    series, aggregate = {}, {}
    for name, mask in (('Deletion', ~is_insertion), ('Insertion', is_insertion)):
        values = {}
        for (low, high), stratum in zip(edges, strata):
            inside = mask & (length >= low) & (length <= high)
            auroc = metrics.auroc(y[inside], scores[inside]) if inside.sum() >= 20 else None
            values[stratum] = {'auroc': auroc, 'n': int(inside.sum())}
        series[name] = values
        aggregate[name] = metrics.auroc(y[mask], scores[mask])
        print(f'  {name}: aggregate {aggregate[name]:.4f}  n={int(mask.sum()):,}', flush=True)
    return {'strata': strata, 'series': series, 'aggregate': aggregate,
            'n': int(len(df)), 'model': str(model_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='indel model, for the size/type figure')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    OUT.mkdir(parents=True, exist_ok=True)

    print('per_category_figure — internal temporal test set', flush=True)
    snv = read('08_evaluate', 'internal_raw') or read('08_evaluate', 'internal_raw_activity_max_unweighted')
    if snv:
        figures.per_category(snv, cfg, OUT / 'per_category_figure.pdf')

    print('reclass_figure — the reclassification cohorts', flush=True)
    result = read('12_reclassification', 'reclassification')
    if result and 'rows' in result.get('pooled', {}):
        figures.per_category(result['pooled'], cfg, OUT / 'reclass_figure.pdf')

    print('fm_combination_matrix — S4.2', flush=True)
    result = read('03_fm_combination', 'fm_combination_maxbin')
    if result:
        figures.combination_matrix(result, cfg, OUT / 'fm_combination_matrix.pdf', labels={
            'ag': 'AlphaGenome', 'gpnmsa': 'GPN-MSA', 'evo2': 'Evo2', 'borzoi': 'Borzoi',
            'dnabert2': 'DNABERT-2', 'ntv2': 'NT-v2', 'hyenadna': 'HyenaDNA',
            'grover': 'GROVER', 'caduceus': 'Caduceus-Ph'})

    print('indel_tool_comparison — internal', flush=True)
    indel = read('08_evaluate', 'indel_raw')
    if indel:
        figures.per_category(indel, cfg, OUT / 'indel_tool_comparison.pdf')

    print('per_category_combined — SNVs and indels as one figure', flush=True)
    if snv and indel:
        # The indel panel is drawn in the SNV category order, with in-frame in the slot
        # synonymous occupies for SNVs, so the five shared categories line up between panels.
        figures.per_category_panels(
            [snv, indel], cfg, OUT / 'per_category_combined.pdf', titles=['SNVs', 'Indels'],
            orders=[None, ['intronic', 'in-frame', 'non-coding', 'splice', 'LoF', 'other']])

    print('indel_tool_comparison_external', flush=True)
    indel_external = read('08_evaluate', 'indel_external_raw')
    if indel_external:
        figures.per_category(indel_external, cfg, OUT / 'indel_tool_comparison_external.pdf')

    print('per_category_external_combined — the external cohort as one figure', flush=True)
    snv_external = read('08_evaluate', 'external_raw')
    if snv_external and indel_external:
        figures.per_category_panels(
            [snv_external, indel_external], cfg, OUT / 'per_category_external_combined.pdf',
            titles=['SNVs', 'Indels'],
            orders=[None, ['intronic', 'in-frame', 'non-coding', 'splice', 'LoF', 'other']])

    print('predictive_values — PPV against assumed prevalence', flush=True)
    result = read('15_predictive_values', 'predictive_values')
    if result:
        figures.predictive_value_curve(result, cfg, OUT / 'predictive_values.pdf')

    print('indel_size_type', flush=True)
    if args.model:
        result = size_and_type(cfg, args.model)
        io.write_result('13_figures', 'indel_size_type', result, cfg)
        figures.stratified(result, cfg, OUT / 'indel_size_type.pdf',
                           xlabel='Indel length (bp)')
    else:
        print('  skipped — pass --model for the indel model', flush=True)

    print(f'\nfigures in {OUT}', flush=True)


if __name__ == '__main__':
    main()
