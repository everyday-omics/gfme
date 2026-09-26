"""Consequence assignment.

The two annotation vocabularies must agree, because the internal cohort is categorised from
ClinVar's MC field and the external cohort from VEP, and their per-category AUROCs are compared
row for row. A term that means the same thing in both must land in the same category.
"""
import pytest

from gfme import consequences as C


def test_seven_snv_categories_including_missense():
    assert len(C.SNV_CATEGORIES) == 7
    assert 'missense' in C.SNV_CATEGORIES
    assert set(C.PRECEDENCE) == set(C.SNV_CATEGORIES)


def test_six_indel_categories_without_missense():
    assert len(C.INDEL_CATEGORIES) == 6
    assert 'missense' not in C.INDEL_CATEGORIES
    assert 'in-frame' in C.INDEL_CATEGORIES
    assert set(C.PRECEDENCE_INDEL) == set(C.INDEL_CATEGORIES)


def test_encoding_is_the_index_in_the_category_list():
    for i, cat in enumerate(C.SNV_CATEGORIES):
        assert C.encode(cat, 'snv') == i
    for i, cat in enumerate(C.INDEL_CATEGORIES):
        assert C.encode(cat, 'indel') == i


@pytest.mark.parametrize('term,expected', [
    ('missense_variant', 'missense'),
    ('synonymous_variant', 'synonymous'),
    ('intron_variant', 'intronic'),
    ('splice_acceptor_variant', 'splice'),
    ('splice_donor_variant', 'splice'),
    ('frameshift_variant', 'LoF'),
    ('stop_gained', 'LoF'),
    ('5_prime_UTR_variant', 'non-coding'),
    ('3_prime_UTR_variant', 'non-coding'),
    ('stop_lost', 'other'),
])
def test_clinvar_terms(term, expected):
    assert C.assign(term, 'clinvar', 'snv') == expected


def test_rna_gene_variants_are_other_in_both_vocabularies():
    """The spelling differs between the two sources; the category must not."""
    assert C.assign('non-coding_transcript_variant', 'clinvar', 'snv') == 'other'
    assert C.assign('non_coding_transcript_variant', 'vep', 'snv') == 'other'
    assert C.assign('non_coding_transcript_exon_variant', 'vep', 'snv') == 'other'


def test_start_loss_is_other_in_both_vocabularies():
    assert C.assign('initiator_codon_variant', 'clinvar', 'snv') == 'other'
    assert C.assign('start_lost', 'vep', 'snv') == 'other'


def test_extended_splice_terms_are_intronic_not_splice():
    """`splice` is the canonical +/-1,2 sites only, which is all ClinVar's MC field marks."""
    for term in ('splice_region_variant', 'splice_donor_5th_base_variant',
                 'splice_polypyrimidine_tract_variant'):
        assert C.assign(term, 'vep', 'snv') == 'intronic'


def test_precedence_decides_multi_term_records():
    assert C.assign('synonymous_variant,non-coding_transcript_variant', 'clinvar') == 'synonymous'
    assert C.assign('non-coding_transcript_variant,intron_variant', 'clinvar') == 'intronic'
    assert C.assign('missense_variant,non-coding_transcript_variant', 'clinvar') == 'missense'
    assert C.assign('non-coding_transcript_variant,5_prime_UTR_variant', 'clinvar') == 'non-coding'
    assert C.assign('frameshift_variant,missense_variant', 'clinvar') == 'LoF'


def test_absent_annotation_is_none_not_other():
    """An unannotated variant is a distinct state from a variant annotated as something odd."""
    for empty in (None, '', '.', [], ['.']):
        assert C.assign(empty, 'clinvar', 'snv') is None


def test_unknown_term_raises_rather_than_bucketing():
    with pytest.raises(C.UnknownTerm):
        C.assign('a_term_nobody_has_seen', 'clinvar', 'snv')
    assert C.assign('a_term_nobody_has_seen', 'clinvar', 'snv', strict=False) is None


def test_shared_terms_agree_between_the_two_vocabularies():
    for term in set(C.CLINVAR_SNV) & set(C.VEP_SNV):
        assert C.CLINVAR_SNV[term] == C.VEP_SNV[term], term


def test_indel_table_has_no_missense_target():
    assert 'missense' not in set(C.CLINVAR_INDEL.values())
    assert 'missense' not in set(C.VEP_INDEL.values())
    assert C.assign('inframe_deletion', 'clinvar', 'indel') == 'in-frame'
    assert C.assign('inframe_insertion', 'vep', 'indel') == 'in-frame'


def test_every_target_is_a_real_category():
    for table, cats in ((C.CLINVAR_SNV, C.SNV_CATEGORIES), (C.VEP_SNV, C.SNV_CATEGORIES),
                        (C.CLINVAR_INDEL, C.INDEL_CATEGORIES),
                        (C.VEP_INDEL, C.INDEL_CATEGORIES)):
        assert set(table.values()) <= set(cats)
