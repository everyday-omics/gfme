"""The comparison tables.

Tables 2, 3, 4 and 5 are the same table computed on four cohorts: per-category and macro AUROC
for the ensemble, for an AlphaGenome-only model of the same architecture, and for every baseline.
They are produced by one function, so a difference between two of them is a difference in the
data and never a difference in how the row was computed.

The ensemble is applied, never refitted: the frozen model is loaded and asked for probabilities.
The AlphaGenome-only row is a genuinely different model — AlphaGenome features plus the
consequence encoding, nothing else — so it has to be fitted, and it is fitted on the training
partition alone at the same configuration.
"""
from __future__ import annotations

import numpy as np

from . import features, io, metrics, models


def ag_only_matrix(name, config):
    """AlphaGenome features and the consequence encoding, without the side scores."""
    from . import consequences

    df = io.load_dataset(name)
    vtype = io.variant_type(name)
    raw, feature_names, found = io.load_alphagenome(name, df['key'].tolist())
    arm = config['alphagenome']['arms'][config['alphagenome']['selected']]
    quantile = io.load_alphagenome(name, df['key'].tolist(), layer='quantile')[0] \
        if arm.get('biosample') == 'quantile' else None
    ag, ag_names = features.build_selected(raw, feature_names, config, quantile)
    cons_num = np.array([consequences.encode(c, vtype) for c in df['consequence']], dtype=float)
    matrix, names = features.assemble(ag, ag_names, np.zeros((len(df), 0)), [], cons_num, config)
    return matrix, names, df


def train_ag_only(config, train='clinvar_train', path=None):
    """Fit the AlphaGenome-only comparison model on the training partition."""
    X, names, df = ag_only_matrix(train, config)
    model = models.fit(X, df['label'].to_numpy(int), config, df['consequence'].to_numpy())
    if path is not None:
        models.save(model, path, config, names, f'{train} (n={len(df)})',
                    extra={'description': 'AlphaGenome features + consequence encoding only'})
    return model, names


def tool_scores(name, config, keys=None):
    """Every baseline column for a cohort, already pointing the pathogenic way."""
    vtype = io.variant_type(name)
    df = io.load_dataset(name)
    scores = io.load_scores(name, df['key'].tolist())
    out = {}
    for spec in config['tools'][vtype]:
        column = spec['name']
        if column not in scores.columns:
            continue
        values = scores[column].to_numpy(dtype=float)
        if spec['negate']:
            values = -values
        if spec.get('categories'):
            mask = df['consequence'].isin(list(spec['categories'])).to_numpy()
            values = np.where(mask, values, np.nan)
        out[spec['label']] = values
    return out, df


def comparison(name, config, ensemble_path, ag_only_model=None, ag_only_names=None):
    """The per-category AUROC table for one cohort."""
    vtype = io.variant_type(name)
    categories = list(config['categories'][vtype])
    min_cell = config['metrics']['min_cell']

    X, names, df = io.load_matrix(name, config)
    y = df['label'].to_numpy(int)
    consequence = df['consequence'].to_numpy()

    ensemble, manifest = models.load(ensemble_path, config, names)
    rows = {'FM Ensemble': models.predict(ensemble, X)}

    if ag_only_model is None:
        ag_only_model, ag_only_names = train_ag_only(config)
    ag_X, ag_names, _ = ag_only_matrix(name, config)
    assert list(ag_names) == list(ag_only_names), 'AG-only feature names differ from the fit'
    rows['AG-CatBoost'] = models.predict(ag_only_model, ag_X)

    baselines, _ = tool_scores(name, config)
    rows.update(baselines)

    evaluated = metrics.evaluate(y, rows, consequence, categories, min_cell)
    coverage = {label: float(np.isfinite(values).mean()) for label, values in rows.items()}
    order = [r for r in config['table_row_order'][vtype] if r in evaluated]
    order += [r for r in evaluated if r not in order]

    return {
        'cohort': name,
        'n': int(len(df)),
        'n_pathogenic': int((y == 1).sum()),
        'n_benign': int((y == 0).sum()),
        'categories': categories,
        'representation': config['alphagenome']['selected'],
        'ensemble_model': str(ensemble_path),
        'ensemble_manifest': manifest.get('config_fingerprint'),
        'min_cell': min_cell,
        'row_order': order,
        'coverage': coverage,
        'rows': {label: evaluated[label] for label in order},
    }


def render(result, config):
    """The table as text and as LaTeX rows, in the published column order."""
    vtype = 'indel' if 'indel' in result['cohort'] else 'snv'
    categories = result['categories']
    short = {'intronic': 'Intr.', 'synonymous': 'Syn.', 'non-coding': 'Non-cod.',
             'splice': 'Splice', 'LoF': 'LoF', 'other': 'Other', 'missense': 'Miss.',
             'in-frame': 'In-fr.'}
    lines = [' ' * 16 + ''.join(f'{short.get(c, c):>10s}' for c in categories) + f'{"Macro":>10s}']
    latex = []
    for label in result['row_order']:
        row = result['rows'][label]
        cells = ''.join(f'{row["per_category"][c]:10.3f}' if c in row['per_category']
                        else f'{"---":>10s}' for c in categories)
        macro = f'{row["macro"]:10.3f}' if row['macro'] is not None else f'{"---":>10s}'
        lines.append(f'{label:16s}{cells}{macro}')
        tex_cells = ' & '.join(f'{row["per_category"][c]:.3f}' if c in row['per_category']
                               else '---' for c in categories)
        tex_macro = f'{row["macro"]:.3f}' if row['macro'] is not None else '---'
        latex.append(f'{label} & {tex_cells} & {tex_macro} \\\\')
    return '\n'.join(lines), '\n'.join(latex)
