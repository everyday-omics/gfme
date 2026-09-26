#!/usr/bin/env python3
"""S7 — PP3/BP4 evidence tiers.

Thresholds are estimated on the training out-of-fold predictions from step 07, then checked
against the internal test set. That order matters: reporting the likelihood ratios a threshold was
chosen to achieve is not a validation of it.

    python3 scripts/11_acmg_calibration.py --oof results/07_final_model/oof_<name>.npz \
        --model results/07_final_model/ensemble_<name>.cbm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import calibration                             # noqa: E402
from gfme import config as config_module                 # noqa: E402
from gfme import io, models                              # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oof', required=True, help='out-of-fold npz from step 07')
    parser.add_argument('--model', required=True, help='the frozen model, for the test check')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='acmg_thresholds')
    # The deployed tool scores indels as well, and an indel is not an SNV of the same category: the
    # categories differ and so does the score distribution, so its tiers have to be fitted on the
    # indel arm rather than borrowed.
    parser.add_argument('--arm', default='snv', choices=['snv', 'indel'])
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories'][args.arm])
    test_set = 'clinvar_test' if args.arm == 'snv' else 'indel_test'
    print(f'configuration: {config_module.describe(cfg)}\n', flush=True)

    training = np.load(args.oof, allow_pickle=True)
    print(f'fitting thresholds on {len(training["oof"]):,} out-of-fold training predictions',
          flush=True)
    calibrated = calibration.calibrate(training['oof'], training['label'],
                                       training['consequence'], cfg, categories)

    X, feature_names, test = io.load_matrix(test_set, cfg)
    model, _ = models.load(args.model, cfg, feature_names)
    test_scores = models.predict(model, X)
    print(f'checking them on the {len(test):,}-variant {args.arm} test set\n', flush=True)
    validated = calibration.validate(calibrated, test_scores, test['label'].to_numpy(int),
                                     test['consequence'].to_numpy(), categories)

    for category in categories:
        entry = calibrated.get(category, {})
        if 'PP3' not in entry:
            print(f'{category:12s} {entry.get("note", "")} (n={entry.get("n", 0)})', flush=True)
            continue
        reached_pp3 = [t for t, v in entry['PP3'].items() if v is not None]
        reached_bp4 = [t for t, v in entry['BP4'].items() if v is not None]
        print(f'{category:12s} n={entry["n"]:5,}  prior {entry["prior"]}', flush=True)
        print(f'{"":12s}   PP3 reaches {" / ".join(reached_pp3) or "none"}', flush=True)
        print(f'{"":12s}   BP4 reaches {" / ".join(reached_bp4) or "none"}', flush=True)
        for tier, row in validated.get(category, {}).items():
            ratio = 'None' if row['test_LR'] is None else f'{row["test_LR"]:.2f}'
            print(f'{"":12s}     {tier:18s} test LR {ratio:>6s}  n={row["n"]:4d}  '
                  f'target {row["target"]}  {row["status"]}', flush=True)

    unvalidated = [(c, t) for c, rows in validated.items() for t, r in rows.items()
                   if r['test_LR'] is None]
    if unvalidated:
        print('\nUnvalidated tiers — no benign variant fell in the band, so the likelihood ratio '
              'is a division by zero rather than a measured value:', flush=True)
        for category, tier in unvalidated:
            print(f'  {category} {tier}', flush=True)

    path = io.write_result('11_acmg_calibration', args.name, {
        'method': 'Pejaver local lr+ with a one-sided bootstrap margin; thresholds from the '
                  'training out-of-fold predictions, checked on the internal test set',
        'arm': args.arm, 'fitted_on': args.oof, 'checked_on': test_set,
        'model': args.model,
        'categories': calibrated, 'validation': validated,
    }, cfg)
    print(f'\n-> {path}', flush=True)


if __name__ == '__main__':
    main()
