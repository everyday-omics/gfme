#!/usr/bin/env python3
"""The per-category comparison tables.

Tables 2, 3, 4 and 5 differ only in which cohort they are computed on, so they are one script.
The ensemble is applied, never refitted: the frozen model from step 07 is loaded and asked for
probabilities, and the loader refuses a model whose configuration does not match the one building
the matrix.

    python3 scripts/08_evaluate.py --model results/07_final_model/ensemble_raw_weighted.cbm
    python3 scripts/08_evaluate.py --cohort external --model ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module    # noqa: E402
from gfme import io, tables                 # noqa: E402

COHORTS = {
    'internal': ('clinvar_test', 'Table 2 — internal temporal test set'),
    'external': ('external', 'Table 5 — external cohort, VKGL + MGeND'),
    'indel': ('indel_test', 'Table 3 — indel temporal test set'),
    'indel_external': ('external_indel', 'Table 4 — external indel cohort'),
    'train': ('clinvar_train', 'training partition (resubstitution — not an estimate)'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', default='internal', choices=sorted(COHORTS))
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    dataset, title = COHORTS[args.cohort]
    name = args.name or f'{args.cohort}_{config_module.describe(cfg).replace("/", "_")}'
    print(f'{title}\nconfiguration: {config_module.describe(cfg)}\n', flush=True)

    result = tables.comparison(dataset, cfg, args.model)
    text, latex = tables.render(result, cfg)
    print(f'{result["n"]:,} variants '
          f'({result["n_pathogenic"]:,} P / {result["n_benign"]:,} B)\n', flush=True)
    print(text, flush=True)
    print('\nLaTeX rows:\n', flush=True)
    print(latex, flush=True)
    print('\ncoverage: ' + '  '.join(f'{k} {100 * v:.1f}%'
                                     for k, v in result['coverage'].items()), flush=True)

    result['title'] = title
    path = io.write_result('08_evaluate', name, result, cfg)
    print(f'\n-> {path}', flush=True)


if __name__ == '__main__':
    main()
