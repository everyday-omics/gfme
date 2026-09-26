"""Variant consequence categories.

One term table, used by both annotation sources. ClinVar's `MC` field and Ensembl VEP spell the
same biology differently, so each has its own term vocabulary, but both resolve to the same seven
SNV categories (six for indels) through the tables here. Assignment is a lookup, never a substring
test: `non-coding_transcript_variant` and `non_coding_transcript_variant` are different strings and
a substring test on either one silently mis-sorts the other.

A ClinVar record may carry several MC terms — 20.7% of ClinVar does — so a variant that matches
more than one category is resolved by `PRECEDENCE`. The order is a convention, not a biological
claim, and it decides the category for about one variant in five; it is stated in the Methods for
that reason. VEP is asked for `most_severe_consequence`, which applies its own severity ordering;
the two orderings disagree on 0.04% of ClinVar (splice co-occurring with missense or with LoF).

Two assignments are worth naming because they look like mistakes and are not:

* RNA-gene variants (`non-coding_transcript_variant`) are `other`, not `non-coding`. `non-coding`
  in this benchmark means UTR, upstream, downstream and intergenic — variants at a protein-coding
  locus that do not change the protein. A variant in an RNA gene is a different object.
* Start-loss (`initiator_codon_variant`) is `other`, not `LoF`.

Both are what the published datasets contain, and both tables agree on them, so the internal and
external cohorts are categorised identically.
"""
from __future__ import annotations

SNV_CATEGORIES = ('intronic', 'synonymous', 'non-coding', 'splice', 'LoF', 'other', 'missense')
INDEL_CATEGORIES = ('in-frame', 'LoF', 'other', 'splice', 'intronic', 'non-coding')

#: Applied when a record carries terms from more than one category.
PRECEDENCE = ('LoF', 'missense', 'splice', 'synonymous', 'intronic', 'non-coding', 'other')
PRECEDENCE_INDEL = ('LoF', 'in-frame', 'splice', 'intronic', 'non-coding', 'other')

# ------------------------------------------------------------------------------------------
# ClinVar MC vocabulary. Every distinct term observed in the release, categorised.
# ------------------------------------------------------------------------------------------
CLINVAR_SNV = {
    'frameshift_variant': 'LoF',
    'nonsense': 'LoF',
    'stop_gained': 'LoF',
    'missense_variant': 'missense',
    'splice_acceptor_variant': 'splice',
    'splice_donor_variant': 'splice',
    'synonymous_variant': 'synonymous',
    'intron_variant': 'intronic',
    '5_prime_UTR_variant': 'non-coding',
    '3_prime_UTR_variant': 'non-coding',
    'genic_upstream_transcript_variant': 'non-coding',
    'genic_downstream_transcript_variant': 'non-coding',
    'upstream_transcript_variant': 'non-coding',
    'downstream_transcript_variant': 'non-coding',
    'initiator_codon_variant': 'other',
    'stop_lost': 'other',
    'non-coding_transcript_variant': 'other',
    'no_sequence_alteration': 'other',
    'coding_sequence_variant': 'other',
    'inframe_insertion': 'other',
    'inframe_deletion': 'other',
    'inframe_indel': 'other',
}

CLINVAR_INDEL = dict(CLINVAR_SNV)
CLINVAR_INDEL.update({
    'inframe_insertion': 'in-frame',
    'inframe_deletion': 'in-frame',
    'inframe_indel': 'in-frame',
    # An indel ClinVar labels missense or synonymous has no such category in the indel benchmark.
    'missense_variant': 'other',
    'synonymous_variant': 'other',
})

# ------------------------------------------------------------------------------------------
# Ensembl VEP vocabulary, aligned term by term to the ClinVar tables above.
# ------------------------------------------------------------------------------------------
# `splice` is the canonical +/-1,2 acceptor and donor only. Every extended splice term is
# intronic, which is what ClinVar's MC field does and therefore what the internal benchmark did.
VEP_SNV = {
    'transcript_ablation': 'LoF',
    'stop_gained': 'LoF',
    'frameshift_variant': 'LoF',
    'start_lost': 'other',                      # ClinVar: initiator_codon_variant -> other
    'splice_acceptor_variant': 'splice',
    'splice_donor_variant': 'splice',
    'splice_donor_5th_base_variant': 'intronic',
    'splice_donor_region_variant': 'intronic',
    'splice_region_variant': 'intronic',
    'splice_polypyrimidine_tract_variant': 'intronic',
    'splice_polyadenylation_variant': 'intronic',
    'synonymous_variant': 'synonymous',
    'stop_retained_variant': 'synonymous',
    'missense_variant': 'missense',
    'stop_lost': 'other',
    'protein_altering_variant': 'other',
    'coding_sequence_variant': 'other',
    'incomplete_terminal_codon_variant': 'other',
    'intron_variant': 'intronic',
    '5_prime_UTR_variant': 'non-coding',
    '3_prime_UTR_variant': 'non-coding',
    'non_coding_transcript_exon_variant': 'other',   # ClinVar: non-coding_transcript_variant
    'non_coding_transcript_variant': 'other',
    'upstream_gene_variant': 'non-coding',
    'downstream_gene_variant': 'non-coding',
    'intergenic_variant': 'non-coding',
    'regulatory_region_variant': 'non-coding',
    'TF_binding_site_variant': 'non-coding',
    'mature_miRNA_variant': 'non-coding',
}

VEP_INDEL = dict(VEP_SNV)
VEP_INDEL.update({
    'inframe_insertion': 'in-frame',
    'inframe_deletion': 'in-frame',
    'transcript_ablation': 'other',
    'missense_variant': 'other',
    'synonymous_variant': 'other',
    'stop_retained_variant': 'other',
})

_TABLES = {
    ('clinvar', 'snv'): (CLINVAR_SNV, PRECEDENCE),
    ('clinvar', 'indel'): (CLINVAR_INDEL, PRECEDENCE_INDEL),
    ('vep', 'snv'): (VEP_SNV, PRECEDENCE),
    ('vep', 'indel'): (VEP_INDEL, PRECEDENCE_INDEL),
}


class UnknownTerm(KeyError):
    """A term absent from the table. Never silently bucketed — the table is extended instead."""


def categories(variant_type: str = 'snv') -> tuple:
    return SNV_CATEGORIES if variant_type == 'snv' else INDEL_CATEGORIES


def assign(terms, source: str = 'clinvar', variant_type: str = 'snv', strict: bool = True):
    """Category for one variant.

    `terms` is the raw annotation: a comma-delimited MC string, a single VEP term, or an
    iterable of terms. Returns None when there is no annotation at all, which is a real state —
    two thirds of the internal `other` bucket in the previous build was variants with no MC
    field — and the caller decides what to do with it rather than having it absorbed here.

    With `strict`, a term that is not in the table raises rather than defaulting to `other`;
    that default is how a spelling difference stayed invisible through three scripts.
    """
    table, precedence = _TABLES[(source, variant_type)]
    if terms is None:
        return None
    if isinstance(terms, str):
        terms = [t for t in terms.replace('|', ',').split(',') if t]
    terms = [str(t).strip() for t in terms if str(t).strip() and str(t).strip() != '.']
    if not terms:
        return None

    found, unknown = set(), []
    for t in terms:
        cat = table.get(t)
        if cat is None:
            unknown.append(t)
        else:
            found.add(cat)
    if unknown and strict:
        raise UnknownTerm(f'{source}/{variant_type}: {unknown!r} — add to the table in '
                          f'gfme/consequences.py, do not let it fall through to "other"')
    if not found:
        return None
    for cat in precedence:
        if cat in found:
            return cat
    raise AssertionError(f'{found!r} contains no category in the precedence order')


def encode(category, variant_type: str = 'snv') -> int:
    """The `cons_num` model input: the category's index in its category list."""
    cats = categories(variant_type)
    try:
        return cats.index(category)
    except ValueError:
        raise UnknownTerm(f'{category!r} is not one of {cats}')
