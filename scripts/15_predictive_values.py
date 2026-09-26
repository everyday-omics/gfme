#!/usr/bin/env python3
"""Predictive values as a function of assumed prevalence.

AUROC is prevalence-independent, which is what makes it the right metric for comparing tools —
and also what makes it silent about the question a clinician actually asks: given a positive
call, how likely is this variant to be pathogenic? That depends on the pre-test probability, and
for most consequence categories no derived prior exists (Methods). So rather than assert one, the
predictive values are reported as a function of an assumed prior and the reader supplies it.

The point of doing this for every tool, not only the ensemble, is that positive predictive value
collapses at low prevalence for all of them. That is a property of rare events, not a weakness of
any one model, and reporting the ensemble's decline alone would misrepresent it as the latter.

Thresholds are chosen on the training partition — out-of-fold for the ensemble, which is
supervised — and applied unchanged to the test set. Choosing them on the test set would report
the operating point that flatters each tool most.

    python3 scripts/15_predictive_values.py --oof results/07_final_model/oof_raw_unweighted.npz \
        --model results/07_final_model/ensemble_raw_unweighted.cbm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                      # noqa: E402
from gfme import io, models, tables                           # noqa: E402

PRIORS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
REPORT_AT = 0.10


def youden_threshold(scores, labels):
    """The threshold maximising sensitivity + specificity - 1, chosen on training data."""
    finite = np.isfinite(scores)
    scores, labels = scores[finite], np.asarray(labels)[finite]
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    positives, negatives = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    j = tp / positives - fp / negatives
    return float(s[int(np.argmax(j))])


def operating_point(scores, labels, threshold):
    finite = np.isfinite(scores)
    scores, labels = scores[finite], np.asarray(labels)[finite]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return None, None, int(len(scores))
    called = scores >= threshold
    sensitivity = float((called & (labels == 1)).sum() / max((labels == 1).sum(), 1))
    specificity = float((~called & (labels == 0)).sum() / max((labels == 0).sum(), 1))
    return sensitivity, specificity, int(len(scores))


def predictive_values(sensitivity, specificity, prior):
    """PPV and NPV implied by an operating point at an assumed pre-test probability."""
    if sensitivity is None:
        return None, None
    tp, fn = sensitivity * prior, (1 - sensitivity) * prior
    fp, tn = (1 - specificity) * (1 - prior), specificity * (1 - prior)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    return ppv, npv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oof', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='predictive_values')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])

    training = np.load(args.oof, allow_pickle=True)
    X, feature_names, test = io.load_matrix('clinvar_test', cfg)
    model, _ = models.load(args.model, cfg, feature_names)
    test_scores = models.predict(model, X)
    y = test['label'].to_numpy(int)
    consequence = test['consequence'].to_numpy()

    # --- the ensemble, per category ---------------------------------------------------------
    print(f'ensemble, thresholds from the training out-of-fold predictions, '
          f'reported at a pre-test probability of {REPORT_AT:.0%}\n', flush=True)
    print(f'  {"category":12s}{"threshold":>10s}{"sens":>7s}{"spec":>7s}{"PPV":>7s}{"NPV":>7s}',
          flush=True)
    per_category = {}
    for category in categories:
        train_mask = training['consequence'] == category
        threshold = youden_threshold(training['oof'][train_mask], training['label'][train_mask])
        mask = consequence == category
        sensitivity, specificity, n = operating_point(test_scores[mask], y[mask], threshold)
        curve = {f'{p:g}': predictive_values(sensitivity, specificity, p) for p in PRIORS}
        ppv, npv = predictive_values(sensitivity, specificity, REPORT_AT)
        per_category[category] = {'threshold': threshold, 'sensitivity': sensitivity,
                                  'specificity': specificity, 'n': n,
                                  'ppv_at_report': ppv, 'npv_at_report': npv,
                                  'curve': {k: {'ppv': v[0], 'npv': v[1]}
                                            for k, v in curve.items()}}
        print(f'  {category:12s}{threshold:10.3f}{sensitivity:7.3f}{specificity:7.3f}'
              f'{ppv:7.3f}{npv:7.3f}', flush=True)

    # --- every tool, pooled, across the prevalence range -------------------------------------
    train_frame = io.load_dataset('clinvar_train')
    train_tools, _ = tables.tool_scores('clinvar_train', cfg)
    test_tools, _ = tables.tool_scores('clinvar_test', cfg)
    train_labels = train_frame['label'].to_numpy(int)

    by_tool = {'FM Ensemble': {}}
    threshold = youden_threshold(training['oof'], training['label'])
    sensitivity, specificity, _ = operating_point(test_scores, y, threshold)
    by_tool['FM Ensemble'] = {'sensitivity': sensitivity, 'specificity': specificity,
                              'ppv': {f'{p:g}': predictive_values(sensitivity, specificity, p)[0]
                                      for p in PRIORS}}
    for label, values in test_tools.items():
        if label not in train_tools:
            continue
        threshold = youden_threshold(train_tools[label], train_labels)
        sensitivity, specificity, n = operating_point(values, y, threshold)
        if sensitivity is None:
            continue
        by_tool[label] = {'sensitivity': sensitivity, 'specificity': specificity,
                          'ppv': {f'{p:g}': predictive_values(sensitivity, specificity, p)[0]
                                  for p in PRIORS}}

    print(f'\n  positive predictive value across the prevalence range, all tools:', flush=True)
    header = ''.join(f'{p:>8g}' for p in PRIORS)
    print(f'  {"tool":16s}{header}', flush=True)
    for label in sorted(by_tool, key=lambda k: -(by_tool[k]['ppv'][f'{REPORT_AT:g}'] or 0)):
        cells = ''.join(f'{by_tool[label]["ppv"][f"{p:g}"]:8.2f}' for p in PRIORS)
        print(f'  {label:16s}{cells}', flush=True)

    io.write_result('15_predictive_values', args.name, {
        'note': 'thresholds chosen on the training partition (out-of-fold for the ensemble) and '
                'applied to the test set; predictive values are reported against an assumed '
                'pre-test probability because no derived prior exists for most categories',
        'reported_at': REPORT_AT, 'priors': PRIORS,
        'ensemble_per_category': per_category, 'all_tools_pooled': by_tool,
    }, cfg)


if __name__ == '__main__':
    main()
