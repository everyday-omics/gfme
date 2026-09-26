"""Partitions, folds and the leakage assertions."""
import numpy as np
import pytest

from gfme import splits

CATS = ('intronic', 'synonymous', 'non-coding', 'splice', 'LoF', 'other', 'missense')


@pytest.fixture
def cohort():
    rng = np.random.default_rng(5)
    sizes = {'intronic': 3265, 'synonymous': 734, 'non-coding': 168, 'splice': 527,
             'LoF': 851, 'other': 467, 'missense': 1565}
    consequence = np.concatenate([np.repeat(c, n) for c, n in sizes.items()])
    labels = rng.integers(0, 2, size=len(consequence))
    return labels, consequence


def test_folds_are_deterministic(cohort):
    labels, consequence = cohort
    a = splits.folds(labels, consequence, 5, seed=42)
    b = splits.folds(labels, consequence, 5, seed=42)
    for (a_tr, a_te), (b_tr, b_te) in zip(a, b):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)


def test_folds_partition_every_row_exactly_once(cohort):
    labels, consequence = cohort
    test_indices = np.concatenate([te for _, te in splits.folds(labels, consequence, 5)])
    assert np.array_equal(np.sort(test_indices), np.arange(len(labels)))


def test_train_and_test_folds_are_disjoint(cohort):
    labels, consequence = cohort
    for train, test in splits.folds(labels, consequence, 5):
        assert not set(train) & set(test)


def test_folds_preserve_the_category_mix(cohort):
    """Categories run from 168 to 3,265 variants; label-only stratification lets the small ones
    drift between folds by enough to move macro-AUROC on its own."""
    labels, consequence = cohort
    overall = {c: (consequence == c).mean() for c in CATS}
    for _, test in splits.folds(labels, consequence, 5):
        for c in CATS:
            assert (consequence[test] == c).mean() == pytest.approx(overall[c], abs=0.01)


def test_folds_preserve_the_label_balance(cohort):
    labels, consequence = cohort
    overall = labels.mean()
    for _, test in splits.folds(labels, consequence, 5):
        assert labels[test].mean() == pytest.approx(overall, abs=0.02)


def test_nested_inner_folds_stay_inside_the_outer_training_rows(cohort):
    labels, consequence = cohort
    for outer in splits.nested_folds(labels, consequence, 5, 5):
        outer_train = set(outer['train'])
        assert not outer_train & set(outer['test'])
        for inner_train, inner_test in outer['inner']:
            assert set(inner_train) <= outer_train
            assert set(inner_test) <= outer_train
            assert not set(inner_train) & set(inner_test)


def test_sample_weights_equalise_category_influence(cohort):
    labels, consequence = cohort
    weights = splits.sample_weights(labels, consequence, CATS)
    totals = [weights[consequence == c].sum() for c in CATS]
    assert np.allclose(totals, totals[0])


def test_leakage_assertion_fires(cohort):
    splits.assert_no_leakage(['chr1:1:A:T'], ['chr1:2:A:T'])
    with pytest.raises(AssertionError, match='both partitions'):
        splits.assert_no_leakage(['chr1:1:A:T'], ['chr1:1:A:T'])


def test_exclusion_assertion_fires():
    splits.assert_excluded(['chr1:1:A:T'], ['chr2:9:G:C'])
    with pytest.raises(AssertionError, match='still in training'):
        splits.assert_excluded(['chr1:1:A:T'], ['chr1:1:A:T'])


def test_fold_index_assertion_catches_folds_built_over_the_wrong_table(cohort):
    labels, consequence = cohort
    pairs = splits.folds(labels, consequence, 5)
    splits.assert_fits_in_train(pairs, len(labels))
    with pytest.raises(AssertionError, match='indexes row'):
        splits.assert_fits_in_train(pairs, 10)
