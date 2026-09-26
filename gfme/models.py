"""Classifiers, cross-validation and the frozen model.

The ensemble is a single CatBoost classifier. `fit` is the only place its hyperparameters are
applied, `oof` the only place out-of-fold predictions are made, and `nested` the only place a
selection is scored — so a configuration used in one experiment cannot differ from the same
configuration used in another.

A saved model travels with a manifest recording the config fingerprint, the feature names in
order, and the partition it was fitted on. Loading checks the fingerprint: a model fitted under
one representation cannot be applied under another without the mismatch being raised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as config_module
from . import metrics, splits


#: The configuration sections that determine a fitted model. A change to any of them makes a
#: saved model inapplicable; a change to the metric or calibration settings does not, so those are
#: deliberately excluded — otherwise editing a bootstrap count would invalidate every model on
#: disk and the check would be trained away as noise.
MODEL_SECTIONS = ('seed', 'categories', 'alphagenome', 'side_features', 'block_order', 'model')


def _catboost(hyperparameters, thread_count):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(thread_count=thread_count, **dict(hyperparameters))


def _other_family(family, hyperparameters, thread_count):
    """The comparison families in S4.3. Each is given several configurations, so none of them
    loses on a default it was never intended to run with."""
    params = dict(hyperparameters)
    if family == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_jobs=thread_count, **params)
    if family == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_jobs=thread_count, verbose=-1, **params)
    if family == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=thread_count, **params)
    if family == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    if family == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=thread_count, **params))
    if family == 'mlp':
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(**params))
    if family in ('flat_mlp', 'cnn', 'tabpfn', 'tabicl'):
        from . import deep
        if family == 'flat_mlp':
            return deep.FlatMLP(**params)
        if family == 'cnn':
            return deep.SimpleCNN(**params)
        return deep.TabularFM(backend=family, **params)
    raise ValueError(f'unknown family {family!r}')


def build(config, family=None, hyperparameters=None):
    family = family or config['model']['family']
    hyperparameters = hyperparameters or config['model']['hyperparameters']
    thread_count = config['model']['thread_count']
    if family == 'catboost':
        return _catboost(hyperparameters, thread_count)
    return _other_family(family, hyperparameters, thread_count)


def fit(X, y, config, consequence=None, family=None, hyperparameters=None, weights=None):
    """Fit one model. Sample weights balance the categories unless told otherwise."""
    model = build(config, family, hyperparameters)
    if weights is None and config['model']['sample_weights'] == 'by_category':
        if consequence is None:
            raise ValueError('by_category weights need the consequence column')
        vtype = 'indel' if len(set(consequence) & set(config['categories']['indel'])) and \
            'missense' not in set(consequence) else 'snv'
        weights = splits.sample_weights(y, consequence, config['categories'][vtype])
    if weights is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=weights)
    except (TypeError, ValueError):
        # scikit-learn pipelines reject `sample_weight` by name and want it routed to a step;
        # some families take no weights at all. Either way, fall back rather than lose the
        # candidate from the comparison — and say so, so a silent change of protocol is visible.
        print(f'    [{type(model).__name__}] does not accept sample weights; fitted unweighted',
              flush=True)
        model.fit(X, y)
    return model


def predict(model, X):
    return model.predict_proba(X)[:, 1]


def shap_values(model, X):
    """Per-feature attributions in log-odds, one row per variant.

    CatBoost computes exact TreeSHAP itself, so no separate package is involved. The returned
    array has one column per feature plus a final column holding the base value, and the row
    sums to the model's log-odds output — which the caller is expected to check.
    """
    from catboost import Pool
    return np.asarray(model.get_feature_importance(Pool(np.asarray(X)), type='ShapValues'))


def oof(X, y, consequence, config, k=None, family=None, hyperparameters=None):
    """Out-of-fold predictions over the training partition.

    Every fold's transform and fit see training rows only; the held-out fold contributes nothing
    to the model that scores it. These are the predictions S7 calibrates its thresholds on, which
    is why they must be out-of-fold and not in-sample.
    """
    k = k or config['cv']['oof_k']
    predictions = np.full(len(y), np.nan)
    pairs = splits.folds(y, consequence, k, config['seed'])
    splits.assert_fits_in_train(pairs, len(y), 'oof')
    for train_index, test_index in pairs:
        model = fit(X[train_index], np.asarray(y)[train_index], config,
                    np.asarray(consequence)[train_index], family, hyperparameters)
        predictions[test_index] = predict(model, X[test_index])
    assert np.isfinite(predictions).all(), 'a fold produced no prediction'
    return predictions


def nested(X, y, consequence, config, candidates, categories, outer_k=None, inner_k=None,
           label=''):
    """Nested cross-validation over `candidates`.

    Selection happens in the inner loop and is scored on the outer fold, so the reported figure
    is not the maximum of the values the selection was made on. The gap between the two — the
    selection bias — is reported alongside, because a flat best-of-N is systematically optimistic
    and the difference is the size of that effect.
    """
    outer_k = outer_k or config['cv']['outer_k']
    inner_k = inner_k or config['cv']['inner_k']
    y = np.asarray(y)
    consequence = np.asarray(consequence)

    fold_reports, outer_scores = [], []
    for i, fold in enumerate(splits.nested_folds(y, consequence, outer_k, inner_k, config['seed'])):
        inner_scores = {}
        for name, spec in candidates.items():
            values = []
            for inner_train, inner_test in fold['inner']:
                model = fit(X[inner_train], y[inner_train], config, consequence[inner_train],
                            spec.get('family'), spec.get('hyperparameters'))
                values.append(metrics.macro_auroc(
                    y[inner_test], predict(model, X[inner_test]), consequence[inner_test],
                    categories, config['metrics']['min_cell']))
            inner_scores[name] = float(np.mean([v for v in values if v is not None]))

        winner = max(inner_scores, key=inner_scores.get)
        spec = candidates[winner]
        model = fit(X[fold['train']], y[fold['train']], config, consequence[fold['train']],
                    spec.get('family'), spec.get('hyperparameters'))
        outer = metrics.macro_auroc(y[fold['test']], predict(model, X[fold['test']]),
                                    consequence[fold['test']], categories,
                                    config['metrics']['min_cell'])
        outer_scores.append(outer)
        fold_reports.append({'fold': i, 'winner': winner, 'inner_macro': inner_scores[winner],
                             'outer_macro': outer, 'all_inner': inner_scores})
        print(f'[{label}] fold {i}: {winner} inner {inner_scores[winner]:.4f} '
              f'outer {outer:.4f}', flush=True)

    winners = [f['winner'] for f in fold_reports]
    mean_inner = {name: float(np.mean([f['all_inner'][name] for f in fold_reports]))
                  for name in candidates}
    selected = max(mean_inner, key=mean_inner.get)
    majority = max(set(winners), key=winners.count)
    return {
        'nested_macro': float(np.mean(outer_scores)),
        'selected_by_mean_inner': selected,
        'selected_by_fold_majority': majority,
        'folds_agreeing': winners.count(majority),
        'outer_k': outer_k, 'inner_k': inner_k,
        'mean_inner_by_candidate': mean_inner,
        'folds': fold_reports,
    }


# ----------------------------------------------------------------------------------------------
# saving and loading
# ----------------------------------------------------------------------------------------------
def save(model, path, config, feature_names, partition, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    manifest = {
        'model': path.name,
        'config_fingerprint': config_module.fingerprint(config, *MODEL_SECTIONS),
        'representation': config['alphagenome']['selected'],
        'hyperparameters': config_module.as_dict(config['model']['hyperparameters']),
        'n_features': len(feature_names),
        'feature_names': list(feature_names),
        'partition': partition,
    }
    manifest.update(extra or {})
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, default=float))
    return path


def load(path, config=None, feature_names=None):
    """Load a saved model, refusing one built under a different configuration."""
    from catboost import CatBoostClassifier

    path = Path(path)
    model = CatBoostClassifier()
    model.load_model(str(path))
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if config is not None and manifest:
        expected = config_module.fingerprint(config, *MODEL_SECTIONS)
        if manifest.get('config_fingerprint') != expected:
            raise AssertionError(
                f'{path.name} was fitted under config {manifest.get("config_fingerprint")} '
                f'({manifest.get("representation")}), not {expected} '
                f'({config["alphagenome"]["selected"]}). Refusing to apply it.')
    if feature_names is not None and manifest.get('feature_names'):
        if list(feature_names) != manifest['feature_names']:
            raise AssertionError(f'{path.name}: feature names or their order differ from the '
                                 f'fitted model — CatBoost is not permutation invariant')
    return model, manifest
