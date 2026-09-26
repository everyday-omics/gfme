#!/usr/bin/env python3
"""S4.5 — is the ensemble's margin an artefact of supervision?

The ensemble is a supervised model and the baselines are zero-shot scores, so a sceptical reader
should ask whether the gap is just the benefit of having seen labelled data. This tests it
directly: give each baseline the same treatment the ensemble gets — a CatBoost wrapped around it,
at the same configuration, cross-validated on the same folds — and see whether supervision lifts
it.

A single score carries no interaction structure for a tree to exploit, so a wrapper around it can
only re-monotonise the score, which cross-validation then charges for. If every tool's calibrated
figure comes out at or below its zero-shot figure, the ensemble's margin is not supervision.

    python3 scripts/06_per_tool_calibration.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                  # noqa: E402
from gfme import features, io, metrics, models, tables    # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='per_tool_calibration')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    categories = list(cfg['categories']['snv'])
    min_cell = cfg['metrics']['min_cell']
    started = time.time()

    X, feature_names, df = io.load_matrix('clinvar_train', cfg)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()
    print(f'{len(df):,} training variants — the test set is never loaded here\n', flush=True)

    baselines, _ = tables.tool_scores('clinvar_train', cfg)
    rows = {}

    # Only tools that address every category. A missense-only predictor has no score outside
    # missense, so a wrapper fitted across all seven is being asked to learn from a column that
    # is absent for four fifths of the rows — the resulting collapse measures the missing data,
    # not the effect of supervision, and would misrepresent the tool.
    restricted = {spec['label'] for spec in cfg['tools']['snv'] if spec.get('categories')}
    for label in list(baselines):
        if label in restricted:
            print(f'  {label:16s} scores only '
                  f'{", ".join(next(s["categories"] for s in cfg["tools"]["snv"] if s["label"] == label))}'
                  f' — excluded from the wrapper comparison', flush=True)
            del baselines[label]

    for label, values in baselines.items():
        zero_shot = metrics.macro_auroc(y, values, consequence, categories, min_cell)
        column = np.nan_to_num(np.asarray(values, float), nan=0.0,
                               posinf=0.0, neginf=0.0).reshape(-1, 1)
        calibrated_scores = models.oof(column, y, consequence, cfg)
        calibrated = metrics.macro_auroc(y, calibrated_scores, consequence, categories, min_cell)
        rows[label] = {'zero_shot': zero_shot, 'calibrated': calibrated,
                       'delta': None if zero_shot is None else calibrated - zero_shot}
        print(f'  {label:16s} zero-shot {zero_shot if zero_shot is None else f"{zero_shot:.3f}"}'
              f'   calibrated {calibrated:.3f}   '
              f'{rows[label]["delta"]:+.3f}' if zero_shot is not None else
              f'  {label:16s} calibrated {calibrated:.3f}', flush=True)

    # The two supervised models, for the comparison the table is really about.
    ag_X, ag_names, _ = tables.ag_only_matrix('clinvar_train', cfg)
    ag_scores = models.oof(ag_X, y, consequence, cfg)
    rows['AG-CatBoost'] = {'zero_shot': None,
                           'calibrated': metrics.macro_auroc(y, ag_scores, consequence,
                                                             categories, min_cell),
                           'delta': None, 'n_features': int(ag_X.shape[1])}
    ensemble_scores = models.oof(X, y, consequence, cfg)
    rows['FM Ensemble'] = {'zero_shot': None,
                           'calibrated': metrics.macro_auroc(y, ensemble_scores, consequence,
                                                             categories, min_cell),
                           'delta': None, 'n_features': int(X.shape[1])}
    print(f'\n  {"AG-CatBoost":16s} calibrated {rows["AG-CatBoost"]["calibrated"]:.3f}  '
          f'({ag_X.shape[1]} features)', flush=True)
    print(f'  {"FM Ensemble":16s} calibrated {rows["FM Ensemble"]["calibrated"]:.3f}  '
          f'({X.shape[1]} features)', flush=True)

    deltas = [r['delta'] for r in rows.values() if r['delta'] is not None]
    lifted = [k for k, r in rows.items() if r['delta'] is not None and r['delta'] > 0]
    print(f'\nsingle-tool wrappers: {len(deltas)} tested, {len(lifted)} improved '
          f'({", ".join(lifted) if lifted else "none"}), mean change '
          f'{np.mean(deltas):+.4f}', flush=True)
    if not lifted:
        print('  no single tool gains from supervision, so the ensemble\'s margin is not a '
              'supervision artefact', flush=True)

    io.write_result('06_per_tool_calibration', args.name, {
        'protocol': f'{cfg["cv"]["oof_k"]}-fold out-of-fold CatBoost on each single score, at '
                    f'the ensemble configuration, training partition only.',
        'n_train': int(len(y)), 'rows': rows,
        'n_tools_improved_by_supervision': len(lifted),
        'mean_delta': float(np.mean(deltas)) if deltas else None,
    }, cfg)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
