"""Metrics, against hand computation and against scikit-learn directly."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from gfme import metrics

CATS = ('intronic', 'synonymous', 'non-coding', 'splice', 'LoF', 'other', 'missense')


@pytest.fixture
def cohort():
    rng = np.random.default_rng(7)
    n_per = 100
    consequence = np.repeat(np.array(CATS), n_per)
    labels = np.tile([0, 1], len(CATS) * n_per // 2)
    scores = labels + rng.normal(scale=0.6, size=len(labels))
    return labels, scores, consequence


def test_auroc_matches_sklearn(cohort):
    labels, scores, _ = cohort
    assert metrics.auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_perfect_and_random_separation():
    assert metrics.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_single_class_cell_is_absent_not_zero():
    assert metrics.auroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_macro_is_the_unweighted_mean_of_the_cells(cohort):
    labels, scores, consequence = cohort
    per, macro, _ = metrics.per_category(labels, scores, consequence, CATS, min_cell=20)
    assert len(per) == 7
    assert macro == pytest.approx(float(np.mean(list(per.values()))))


def test_macro_ignores_category_size():
    """A category with ten times the variants must not carry ten times the weight."""
    rng = np.random.default_rng(1)
    big_n, small_n = 2000, 40
    consequence = np.array(['intronic'] * big_n + ['non-coding'] * small_n)
    labels = np.array([0, 1] * (big_n // 2) + [0, 1] * (small_n // 2))
    scores = np.concatenate([labels[:big_n] + rng.normal(scale=0.1, size=big_n),
                             labels[big_n:] + rng.normal(scale=3.0, size=small_n)])
    per, macro, _ = metrics.per_category(labels, scores, consequence,
                                         ('intronic', 'non-coding'), min_cell=20)
    assert macro == pytest.approx((per['intronic'] + per['non-coding']) / 2)


def test_small_cells_are_dropped_by_min_cell(cohort):
    labels, scores, consequence = cohort
    consequence = consequence.copy()
    consequence[:95] = 'other'                       # leaves intronic with 5 variants
    per, _, counts = metrics.per_category(labels, scores, consequence, CATS, min_cell=20)
    assert 'intronic' not in per
    assert all(n >= 20 for n in counts.values())


def test_a_missense_only_tool_gets_a_macro_over_missense_alone(cohort):
    """REVEL and AlphaMissense score one category; their macro must not average in six blanks."""
    labels, scores, consequence = cohort
    scores = np.where(consequence == 'missense', scores, np.nan)
    per, macro, _ = metrics.per_category(labels, scores, consequence, CATS, min_cell=20)
    assert set(per) == {'missense'}
    assert macro == pytest.approx(per['missense'])


def test_non_finite_scores_are_excluded_not_treated_as_low(cohort):
    labels, scores, consequence = cohort
    partial = scores.copy()
    partial[::3] = np.nan
    per_full, _, _ = metrics.per_category(labels, scores, consequence, CATS)
    per_part, _, counts = metrics.per_category(labels, partial, consequence, CATS)
    assert all(counts[c] < 100 for c in counts)
    for c in per_part:
        assert per_part[c] == pytest.approx(
            roc_auc_score(labels[(consequence == c) & np.isfinite(partial)],
                          partial[(consequence == c) & np.isfinite(partial)]))
    assert per_full != per_part


def test_bootstrap_interval_brackets_the_point_estimate(cohort):
    labels, scores, consequence = cohort
    _, macro, _ = metrics.per_category(labels, scores, consequence, CATS)
    lo, hi = metrics.bootstrap_ci(labels, scores, consequence, CATS, iterations=200, seed=3)
    assert lo < macro < hi


def test_bootstrap_is_deterministic_for_a_seed(cohort):
    labels, scores, consequence = cohort
    a = metrics.bootstrap_ci(labels, scores, consequence, CATS, iterations=100, seed=11)
    b = metrics.bootstrap_ci(labels, scores, consequence, CATS, iterations=100, seed=11)
    assert a == b


def test_evaluate_returns_one_entry_per_tool(cohort):
    labels, scores, consequence = cohort
    out = metrics.evaluate(labels, {'a': scores, 'b': -scores}, consequence, CATS)
    assert set(out) == {'a', 'b'}
    assert out['a']['macro'] == pytest.approx(1 - out['b']['macro'])
