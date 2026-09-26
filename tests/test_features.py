"""AlphaGenome feature construction."""
import numpy as np
import pytest

from gfme import config, features

ACTIVITY = ('atac_center_active_sum', 'cage_center_active_sum', 'dnase_center_active_sum',
            'histone_center_active_sum', 'procap_center_active_sum', 'tf_center_active_sum',
            'rnaseq_gene_active')


@pytest.fixture(scope='module')
def cfg():
    return config.load()


@pytest.fixture(scope='module')
def synthetic():
    """A small matrix with the real column-naming scheme: 19 scorers x 40 biosamples."""
    scorers = ['atac_center_active_sum', 'atac_center_diff_log2_sum', 'cage_center_active_sum',
               'cage_center_diff_log2_sum', 'contact_contact', 'dnase_center_active_sum',
               'dnase_center_diff_log2_sum', 'histone_center_active_sum',
               'histone_center_diff_log2_sum', 'procap_center_active_sum',
               'procap_center_diff_log2_sum', 'rnaseq_gene_active', 'rnaseq_gene_lfc',
               'rnaseq_polyA', 'splice_junc_splice_junc', 'splice_sites_gene_splice',
               'splice_usage_gene_splice', 'tf_center_active_sum', 'tf_center_diff_log2_sum']
    biosamples = [f'B{i:03d}' for i in range(40)]
    names = [f'{s}__{b}' for s in scorers for b in biosamples]
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, len(names))).astype(np.float32)
    raw[raw > 2.5] = np.nan                                  # AlphaGenome leaves real gaps
    return raw, names, scorers, biosamples


def test_selected_arm_is_one_scorer_column_plus_one_biosample_column(cfg, synthetic):
    raw, names, scorers, biosamples = synthetic
    matrix, columns = features.build_selected(raw, names, cfg)
    assert matrix.shape == (50, len(scorers) + len(biosamples))
    assert columns[:len(scorers)] == [f'scorer__{s}' for s in sorted(scorers)]
    assert columns[len(scorers):] == [f'activity__{b}' for b in sorted(biosamples)]


def test_real_alphagenome_width_reduces_to_747(cfg):
    """19 scorers x 728 biosamples is the real vocabulary, and 19 + 728 = 747."""
    assert features.N_AG747 == 19 + 728
    assert cfg['alphagenome']['n_raw_columns'] == 2536


def test_selected_arm_reads_raw_not_quantile(cfg, synthetic):
    """The selected representation is raw throughout. A quantile layer must not change it."""
    raw, names, _, _ = synthetic
    quantile = np.zeros_like(raw)
    a, _ = features.build_selected(raw, names, cfg)
    b, _ = features.build_selected(raw, names, cfg, quantile=quantile)
    assert np.array_equal(a, b)


def test_quantile_arm_does_read_the_quantile_layer(cfg, synthetic):
    raw, names, _, _ = synthetic
    quantile = np.zeros_like(raw)
    arm = cfg['alphagenome']['arms']['quantile_activity_max']
    matrix, _ = features.build(raw, names, arm, quantile, cfg['alphagenome']['activity_scorers'])
    # Scorer block still comes from raw; the biosample block is all zeros.
    assert not np.allclose(matrix[:, :19], 0.0)
    assert np.allclose(matrix[:, 19:], 0.0)


def test_deterministic(cfg, synthetic):
    raw, names, _, _ = synthetic
    a, _ = features.build_selected(raw, names, cfg)
    b, _ = features.build_selected(raw, names, cfg)
    assert np.array_equal(a, b)


def test_the_two_blocks_share_their_maximum_over_the_activity_scorers(cfg, synthetic):
    """The blocks overlap by construction, and the overlap is exact.

    The biosample block maxes the activity scorers within each biosample; the scorer block maxes
    each scorer across biosamples. Both therefore reach the same set of cells for the seven
    activity scorers, so the largest value in the biosample block equals the largest of those
    seven scorer columns. The redundancy is real and is kept because the frozen model was fitted
    with both blocks; pinning it here keeps it a known property rather than a later discovery.
    """
    raw, names, scorers, biosamples = synthetic
    raw = np.nan_to_num(raw, nan=0.0)               # exactness, without the all-missing fallback
    matrix, columns = features.build_selected(raw, names, cfg)
    order = sorted(scorers)
    scorer_block = matrix[:, :len(scorers)]
    biosample_block = matrix[:, len(scorers):]
    activity_max = np.max([scorer_block[:, order.index(s)] for s in ACTIVITY], axis=0)
    assert np.allclose(activity_max, biosample_block.max(axis=1), atol=1e-6)


def test_all_missing_group_becomes_zero(cfg, synthetic):
    raw, names, scorers, biosamples = synthetic
    raw = raw.copy()
    target = [i for i, n in enumerate(names) if n.endswith('__B000')]
    raw[:, target] = np.nan
    matrix, columns = features.build_selected(raw, names, cfg)
    assert np.all(matrix[:, columns.index('activity__B000')] == 0.0)


def test_signed_max_keeps_direction(cfg):
    """Aggregation must not take magnitudes: a strongly negative effect stays negative."""
    names = [f'{s}__{b}' for s in ('atac_center_active_sum',) for b in ('B0',)]
    raw = np.array([[-5.0]], dtype=np.float32)
    arm = dict(scorer='raw', biosample='raw', scorers='activity', agg='max')
    matrix, _ = features.build(raw, names, arm, activity_scorers=ACTIVITY)
    assert matrix.min() == pytest.approx(-5.0)


def test_mean_arm_differs_from_max_arm(cfg, synthetic):
    raw, names, _, _ = synthetic
    arms = cfg['alphagenome']['arms']
    a, _ = features.build(raw, names, arms['raw_activity_max'],
                          activity_scorers=cfg['alphagenome']['activity_scorers'])
    b, _ = features.build(raw, names, arms['raw_activity_mean'],
                          activity_scorers=cfg['alphagenome']['activity_scorers'])
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_no_aggregation_arm_returns_the_raw_matrix(cfg, synthetic):
    raw, names, _, _ = synthetic
    matrix, columns = features.build(raw, names, cfg['alphagenome']['arms']['none'],
                                     activity_scorers=cfg['alphagenome']['activity_scorers'])
    assert matrix.shape == raw.shape
    assert columns == names


def test_assemble_enforces_block_order_and_width(cfg, synthetic):
    raw, names, scorers, biosamples = synthetic
    ag, ag_names = features.build_selected(raw, names, cfg)
    side = np.column_stack([np.arange(50.0), np.arange(50.0)])
    matrix, columns = features.assemble(ag, ag_names, side, ['gpnmsa_neg', 'evo2_neg'],
                                        np.zeros(50), cfg)
    assert matrix.shape[1] == ag.shape[1] + 2 + 1
    assert columns[-1] == 'cons_num'
    assert columns[:len(ag_names)] == ag_names
    assert np.isfinite(matrix).all()


def test_assemble_replaces_non_finite_scores_with_zero(cfg, synthetic):
    raw, names, _, _ = synthetic
    ag, ag_names = features.build_selected(raw, names, cfg)
    side = np.full((50, 2), np.nan)
    matrix, _ = features.assemble(ag, ag_names, side, ['a', 'b'], np.zeros(50), cfg)
    assert np.isfinite(matrix).all()


def test_apply_directions_negates_the_declared_columns(cfg):
    scores = {'gpnmsa': np.array([1.0, -2.0]), 'evo2': np.array([3.0, 4.0])}
    out = features.apply_directions(scores, cfg, 'snv')
    assert np.array_equal(out['gpnmsa'], [-1.0, 2.0])
    assert np.array_equal(out['evo2'], [-3.0, -4.0])
