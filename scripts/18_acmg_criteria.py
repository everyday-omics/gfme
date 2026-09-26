#!/usr/bin/env python3
"""Do the PP3/BP4 calls agree with classifications reached without computational evidence?

The natural comparison, our tier against the tier a submitter recorded, measures agreement with
CADD, REVEL and SpliceAI rather than with the truth, since those are what made the submitter
invoke PP3 or BP4 in the first place. The comparison that does mean something is our tier against
the classification itself, and that is only clean where no predictor contributed to the
classification.

ClinVar has no structured field for the criteria, so the only variants where this is knowable are
the ones whose submitters wrote the codes into free text. Those split into two groups:

    independent  criteria recorded, none of them PP3 or BP4, so the P/LP or B/LB call rests on
                 evidence such as PVS1, PS3, PM2 or segregation. Our tier is an independent
                 prediction here.
    documented   criteria recorded including PP3 or BP4, so a predictor contributed to the label.

Both are reported. The first is the validation; the second is the contrast, and the gap between
them is what a computational criterion is worth to the label on the variants that document it.

Thresholds come from step 11, which fitted them on training out-of-fold predictions, and the
model is the frozen one from step 07: nothing is refitted here.

    python3 scripts/18_acmg_criteria.py \
        --model results/07_final_model/ensemble_raw_unweighted.cbm \
        --thresholds results/11_acmg_calibration/acmg_thresholds.json
"""
import argparse
import csv
import json
import os
import gzip
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module    # noqa: E402
from gfme import io, models                 # noqa: E402

# The ClinVar tabular releases, variant_summary.txt.gz and submission_summary.txt.gz. Set
# GFME_CLINVAR_TABULAR to point at your own download.
CLINVAR = Path(os.environ.get('GFME_CLINVAR_TABULAR', 'data/external'))

CRITERION = re.compile(r'\b(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])\b')
COMPUTATIONAL = re.compile(r'\b(PP3|BP4)\b')

TIERS = ['Very_Strong', 'Strong', 'Moderate', 'Supporting']

csv.field_size_limit(10 ** 8)


def criteria_groups(keys):
    """Split the given variant keys by what their submitters recorded.

    Returns (any_criteria, cites_pp3_bp4) as sets of keys. A variant belongs to `cites_pp3_bp4`
    if any one of its submissions mentions PP3 or BP4, since a single such submission is enough
    for the label to have been touched by a predictor.
    """
    wanted = set(keys)
    variation_to_key = {}
    with gzip.open(CLINVAR / 'variant_summary.txt.gz', 'rt',
                   encoding='utf8', errors='replace') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row.get('Assembly') != 'GRCh38':
                continue
            key = (f"chr{row['Chromosome']}:{row['PositionVCF']}:"
                   f"{row['ReferenceAlleleVCF']}:{row['AlternateAlleleVCF']}")
            if key in wanted:
                variation_to_key[row['VariationID']] = key
    print(f'mapped to VariationID: {len(set(variation_to_key.values())):,}', flush=True)

    any_criteria, computational = set(), set()
    with gzip.open(CLINVAR / 'submission_summary.txt.gz', 'rt',
                   encoding='utf8', errors='replace') as handle:
        for line in handle:
            if line[0] == '#':
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 13:
                continue
            key = variation_to_key.get(fields[0])
            if not key:
                continue
            # Column 3 is the submitter description of the classification, column 12 the comment.
            text = (fields[3] or '') + ' ' + (fields[12] or '')
            if CRITERION.search(text):
                any_criteria.add(key)
                if COMPUTATIONAL.search(text):
                    computational.add(key)
    return any_criteria, computational


def call(probability, category, thresholds):
    """PP3, BP4 or indeterminate under the step 11 thresholds."""
    entry = thresholds.get(category, {})
    for tier in TIERS:
        cut = entry.get('PP3', {}).get(tier)
        if cut is not None and probability >= cut:
            return 'PP3'
    for tier in TIERS:
        cut = entry.get('BP4', {}).get(tier)
        if cut is not None and probability < cut:
            return 'BP4'
    return 'indeterminate'


def summarise(frame, label):
    """Crosstab, concordance among the called, and the same per category."""
    called = frame[frame['call'] != 'indeterminate']
    agree = int(((called['call'] == 'PP3') & (called['truth'] == 'P/LP')).sum()
                + ((called['call'] == 'BP4') & (called['truth'] == 'B/LB')).sum())
    indeterminate = int((frame['call'] == 'indeterminate').sum())

    print(f'\n=== {label}: {len(frame):,} variants '
          f'({int((frame.truth == "P/LP").sum()):,} P/LP, '
          f'{int((frame.truth == "B/LB").sum()):,} B/LB) ===', flush=True)
    print(pd.crosstab(frame['call'], frame['truth']).to_string(), flush=True)
    print(f'\nindeterminate {indeterminate} ({100 * indeterminate / max(len(frame), 1):.1f}%)',
          flush=True)
    print(f'of the {len(called)} called: {agree} agree -> '
          f'{100 * agree / max(len(called), 1):.1f}% concordance', flush=True)

    per_category = {}
    print(f'\n{"category":12s} {"n":>5s} {"IND":>5s} {"called":>6s} {"agree":>6s} {"conc":>7s}',
          flush=True)
    for category, group in frame.groupby('cat'):
        group_called = group[group['call'] != 'indeterminate']
        group_agree = int(((group_called['call'] == 'PP3')
                           & (group_called['truth'] == 'P/LP')).sum()
                          + ((group_called['call'] == 'BP4')
                             & (group_called['truth'] == 'B/LB')).sum())
        concordance = (round(100 * group_agree / len(group_called), 1)
                       if len(group_called) else None)
        per_category[category] = {
            'n': len(group), 'indeterminate': int((group['call'] == 'indeterminate').sum()),
            'called': len(group_called), 'agree': group_agree, 'concordance': concordance}
        shown = f'{concordance:6.1f}%' if concordance is not None else '     --'
        print(f'{category:12s} {len(group):5d} {per_category[category]["indeterminate"]:5d} '
              f'{len(group_called):6d} {group_agree:6d} {shown}', flush=True)

    return {'n': len(frame), 'n_pathogenic': int((frame.truth == 'P/LP').sum()),
            'n_benign': int((frame.truth == 'B/LB').sum()),
            'n_indeterminate': indeterminate, 'n_called': len(called), 'agree': agree,
            'concordance': round(100 * agree / len(called), 1) if len(called) else None,
            'crosstab': pd.crosstab(frame['call'], frame['truth']).to_dict(),
            'per_category': per_category}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='the frozen model from step 07')
    parser.add_argument('--thresholds', required=True, help='the JSON written by step 11')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='acmg_criteria')
    args = parser.parse_args()

    started = time.time()
    cfg = config_module.override(config_module.load(), args.overrides)
    print(f'configuration: {config_module.describe(cfg)}\n', flush=True)

    with open(args.thresholds) as handle:
        thresholds = json.load(handle)['categories']
    X, feature_names, test = io.load_matrix('clinvar_test', cfg)
    model, _ = models.load(args.model, cfg, feature_names)
    probability = models.predict(model, X)
    keys = test['key'].astype(str).to_numpy()
    print(f'test variants: {len(keys):,}', flush=True)

    any_criteria, computational = criteria_groups(keys)
    independent = any_criteria - computational
    print(f'criteria recorded: {len(any_criteria):,}; citing PP3/BP4: {len(computational):,}; '
          f'citing criteria but not PP3/BP4: {len(independent):,}', flush=True)

    frame = pd.DataFrame({
        'key': keys,
        'cat': test['consequence'].astype(str).to_numpy(),
        'truth': np.where(test['label'].to_numpy(int) == 1, 'P/LP', 'B/LB'),
        'score': probability})
    frame['call'] = [call(p, c, thresholds) for p, c in zip(frame['score'], frame['cat'])]
    frame['group'] = np.where(frame['key'].isin(independent), 'independent',
                              np.where(frame['key'].isin(computational), 'documented', 'none'))
    frame[frame['group'] != 'none'].to_csv(
        io.results_dir('18_acmg_criteria') / f'{args.name}_calls.tsv', sep='\t', index=False)

    payload = {
        'thresholds': args.thresholds, 'model': args.model, 'cohort': 'clinvar_test',
        'n_test': int(len(keys)), 'n_any_criteria': len(any_criteria),
        'n_citing_pp3_bp4': len(computational), 'n_independent': len(independent),
        'groups': {
            # The whole test set is the comparison that matters: every variant carries a ClinVar
            # classification, so nothing has to be discarded to make the check.
            'all': summarise(frame, 'all test variants'),
            'with_criteria': summarise(frame[frame['group'] != 'none'],
                                       'variants whose submissions record criteria'),
            'independent': summarise(frame[frame['group'] == 'independent'],
                                     'independent (criteria recorded, no PP3/BP4)'),
            'documented': summarise(frame[frame['group'] == 'documented'],
                                    'documented (criteria recorded including PP3/BP4)'),
        },
    }
    path = io.write_result('18_acmg_criteria', args.name, payload, cfg)
    print(f'\n-> {path}   ({(time.time() - started) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
