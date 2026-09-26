"""Golden numbers.

Values the manuscript depends on, pinned so a later change cannot move them silently. A failure
here is not necessarily a bug — it may be a deliberate change — but it must be noticed and the
pinned value updated in the same commit that causes it.

The dataset checks run whenever `data/` is present. The result checks run only once the
corresponding step has been run, so the suite is still useful on a fresh checkout.
"""
import json
from pathlib import Path

import pytest

from gfme import config, consequences, features, io

ROOT = Path(__file__).resolve().parent.parent

#: dataset -> (rows, pathogenic, benign)
COUNTS = {
    'clinvar_train': (7577, 3659, 3918),
    'clinvar_train_excluded': (1194, 708, 486),
    'clinvar_test': (40179, 9284, 30895),
    'external': (37745, 14017, 23728),
    'indel_train': (6480, 3240, 3240),
    'indel_test': (4428, 2214, 2214),
    'external_indel': (14573, 10087, 4486),
    'reclass_vus': (20171, 3714, 16457),
    'reclass_conflicting': (7259, 2973, 4286),
}

#: step, file, path into the JSON, expected value, tolerance
RESULTS = [
    ('08_evaluate', 'internal_raw', ('rows', 'FM Ensemble', 'macro'), 0.906, 0.002),
    ('08_evaluate', 'internal_raw', ('rows', 'CADD', 'macro'), 0.8507, 0.002),
    ('08_evaluate', 'internal_raw', ('rows', 'REVEL', 'macro'), 0.9461, 0.002),
    ('08_evaluate', 'external_raw', ('rows', 'FM Ensemble', 'macro'), 0.886, 0.002),
    ('08_evaluate', 'external_raw', ('rows', 'CADD', 'macro'), 0.8566, 0.002),
    ('08_evaluate', 'indel_raw', ('rows', 'FM Ensemble', 'macro'), 0.935, 0.003),
    ('08_evaluate', 'indel_external_raw', ('rows', 'FM Ensemble', 'macro'), 0.881, 0.003),
    ('12_reclassification', 'reclassification', ('pooled', 'macro'), 0.8916, 0.003),
    ('12_reclassification', 'reclassification', ('pooled', 'n'), 27430, 0),
    ('02_representation', 'representation', ('selected',), 'raw_activity_max', None),
]


def _dataset(name):
    try:
        return io.load_dataset(name)
    except io.MissingData:
        pytest.skip(f'{name} not built')


@pytest.mark.parametrize('name,expected', COUNTS.items())
def test_dataset_counts(name, expected):
    df = _dataset(name)
    rows, pathogenic, benign = expected
    assert len(df) == rows
    assert int((df['label'] == 1).sum()) == pathogenic
    assert int((df['label'] == 0).sum()) == benign


def test_training_partition_has_all_seven_categories():
    """Missense is one of the seven, not a separate benchmark."""
    df = _dataset('clinvar_train')
    assert set(df['consequence']) == set(consequences.SNV_CATEGORIES)
    assert int((df['consequence'] == 'missense').sum()) == 1565


def test_indel_partitions_have_no_missense():
    df = _dataset('indel_train')
    assert set(df['consequence']) <= set(consequences.INDEL_CATEGORIES)
    assert 'missense' not in set(df['consequence'])


def test_the_exclusion_is_honoured():
    """The 1,194 variants shared with the external cohort must not be in training."""
    train = _dataset('clinvar_train')
    excluded = _dataset('clinvar_train_excluded')
    assert not set(train['key']) & set(excluded['key'])


def test_no_variant_is_in_both_partitions():
    train, test = _dataset('clinvar_train'), _dataset('clinvar_test')
    assert not set(train['key']) & set(test['key'])


def test_reclassification_cohorts_are_disjoint_from_training():
    train = _dataset('clinvar_train')
    for name in ('reclass_vus', 'reclass_conflicting'):
        assert not set(train['key']) & set(_dataset(name)['key'])


def test_alphagenome_reduces_to_747():
    cfg = config.load()
    try:
        raw, names, _ = io.load_alphagenome('clinvar_train', _dataset('clinvar_train')['key'])
    except io.MissingData:
        pytest.skip('AlphaGenome matrices not built')
    assert raw.shape[1] == cfg['alphagenome']['n_raw_columns']
    matrix, columns = features.build_selected(raw, names, cfg)
    assert matrix.shape[1] == features.N_AG747 == 747
    assert len(columns) == 747


def test_feature_widths():
    cfg = config.load()
    try:
        X, names, _ = io.load_matrix('clinvar_train', cfg)
    except io.MissingData:
        pytest.skip('data not built')
    assert X.shape[1] == 750, 'AG-747 + GPN-MSA + Evo2 + consequence'
    assert names[-3:] == ['gpnmsa_neg', 'evo2_neg', 'cons_num']
    X, names, _ = io.load_matrix('indel_train', cfg)
    assert X.shape[1] == 751, 'AG-747 + Evo2 + MSA gap + indel size + consequence'
    assert names[-1] == 'cons_num'


@pytest.mark.parametrize('step,name,path,expected,tolerance', RESULTS)
def test_golden_numbers(step, name, path, expected, tolerance):
    file = ROOT / 'results' / step / f'{name}.json'
    if not file.exists():
        pytest.skip(f'{step}/{name} not run')
    value = json.loads(file.read_text())
    for key in path:
        value = value[key]
    if tolerance is None:
        assert value == expected
    else:
        assert value == pytest.approx(expected, abs=tolerance)
