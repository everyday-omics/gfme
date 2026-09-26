#!/usr/bin/env python3
"""Where the locked model's importance sits, by block.

Two numbers are reported for each block because they answer different questions and can disagree.
Gain importance says how much the model used a feature while fitting; the ablation in step 10 says
how much accuracy is lost without it. A block can dominate the first and still be partly
replaceable in the second, which is exactly the case here and is why the limitation in the paper
quotes both.

Blocks are the four inputs the ensemble is built from, and every feature belongs to exactly one.

    python3 scripts/17_importance.py --model results/07_final_model/ensemble_raw_unweighted.cbm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                      # noqa: E402
from gfme import io, models                                   # noqa: E402

BLOCKS = {'gpnmsa_neg': 'GPN-MSA', 'evo2_neg': 'Evo2', 'msa_gap': 'MSA gap',
          'cons_num': 'Consequence category', 'indel_size': 'Indel size'}


def block_of(name):
    if name.startswith(('scorer__', 'activity__')):
        return 'AlphaGenome'
    return BLOCKS.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='clinvar_test')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='importance')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    X, feature_names, _ = io.load_matrix(args.dataset, cfg)
    model, _ = models.load(args.model, cfg, feature_names)

    gain = np.asarray(model.get_feature_importance(type='PredictionValuesChange'), dtype=float)
    assert len(gain) == len(feature_names)
    totals = {}
    for name, value in zip(feature_names, gain):
        totals[block_of(name)] = totals.get(block_of(name), 0.0) + float(value)
    total = sum(totals.values())
    shares = {k: v / total for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}

    print(f'gain importance by block, {args.model}', flush=True)
    for name, share in shares.items():
        print(f'  {name:22s} {100 * share:5.1f}%', flush=True)

    io.write_result('17_importance', args.name, {
        'model': args.model, 'dataset': args.dataset,
        'method': 'CatBoost PredictionValuesChange, summed within each block',
        'n_features': len(feature_names), 'share_by_block': shares,
    }, cfg)


if __name__ == '__main__':
    main()
