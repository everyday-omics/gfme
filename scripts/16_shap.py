#!/usr/bin/env python3
"""Per-variant attribution for a handful of example variants.

This is an illustration of what the deployed tool returns for one variant, not a result: the
attributions are read off the locked model, nothing is fitted here, and no claim is made that a
tissue or an assay is causal. It exists because a per-variant explanation is what a laboratory
sees, and a reader should be able to judge whether that explanation is legible.

Two panels per variant. On the left the model's four inputs at block level --- AlphaGenome, the
two scalar models, and the consequence encoding --- which is the decomposition the score itself
has. On the right the AlphaGenome block broken out, because it is one number covering 747
columns and the left panel says nothing about which of them moved.

The breakdown groups those columns so that every one is counted exactly once: the 19 scorer
columns fold into their nine assay modalities, and each per-biosample activity column stands on
its own, with everything outside the top few netted into a single remainder bar. Because SHAP is
additive the bars sum to the AlphaGenome bar on the left, and that is asserted rather than
assumed.

A biosample column is a maximum over whatever assays AlphaGenome provides for that biosample, so
its name is a label on an activity level rather than evidence about a tissue. The figure says
`activity in X` for that reason.

    python3 scripts/16_shap.py --model results/07_final_model/ensemble_raw_unweighted.cbm
"""
import argparse
import os
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gfme import config as config_module                      # noqa: E402
from gfme import figures, io, models                          # noqa: E402

#: One per mechanism rather than one per gene. The two intronic variants are the point: identical
#: consequence annotation, opposite explanations.
#:
#: A `key` names a specific variant; a `rule` picks one from the test set by the pattern it is
#: there to show. The missense case is a rule because the missense minority is capped at 1,000 in
#: the test set, so no particular missense variant is guaranteed to be in it, and a hard-coded key
#: would break the moment the partition is rebuilt.
EXAMPLES = [
    {'key': 'chr8:117812871:C:G', 'status': 'pathogenic',
     'note': 'canonical splice donor, and the signal is splicing'},
    # Pinned. `pick_missense` selected this variant from the test set; the key is written out so
    # that the figure, and the attribution values quoted alongside it in the supplement, do not
    # move if the partition is rebuilt. Pass --repick-missense to re-derive it.
    {'key': 'chr17:7851749:T:C', 'status': 'pathogenic',
     'note': 'protein-level; GPN-MSA carries it and AlphaGenome is near silent'},
    {'key': 'chr2:47466813:G:A', 'status': 'VUS in 2023, now P/LP',
     'note': 'intronic but splicing-driven'},
    {'key': 'chrX:38352591:C:A', 'status': 'pathogenic',
     'note': 'intronic and not splicing-driven'},
]

#: Searched in this order; the first dataset holding a key supplies its features.
SOURCES = ('clinvar_test', 'reclass_vus', 'reclass_conflicting', 'external', 'clinvar_train')

MODALITY = {'atac': 'ATAC accessibility', 'cage': 'CAGE transcription initiation',
            'contact': 'chromatin contact', 'dnase': 'DNase accessibility',
            'histone': 'histone modification', 'procap': 'PRO-cap',
            'rnaseq': 'RNA expression', 'splice': 'splicing', 'tf': 'TF binding'}

BLOCKS = {'gpnmsa_neg': 'GPN-MSA', 'evo2_neg': 'Evo2', 'cons_num': 'Consequence category'}

#: Display names of the 19 AlphaGenome scorer columns, as the web tool's attribution panel shows
#: them, so the figure and the tool read the same way.
SCORER_LABEL = {
    'splice_junc_splice_junc': 'Splice junctions', 'splice_usage_gene_splice': 'Splice site usage',
    'splice_sites_gene_splice': 'Splice sites', 'contact_contact': '3D contacts',
    'rnaseq_polyA': 'Polyadenylation', 'rnaseq_gene_lfc': 'Gene expression change',
    'rnaseq_gene_active': 'Gene expression level', 'cage_center_diff_log2_sum': 'CAGE change',
    'cage_center_active_sum': 'CAGE level', 'procap_center_diff_log2_sum': 'PRO-cap change',
    'procap_center_active_sum': 'PRO-cap level', 'atac_center_diff_log2_sum': 'ATAC-seq change',
    'atac_center_active_sum': 'ATAC-seq level', 'dnase_center_diff_log2_sum': 'DNase-seq change',
    'dnase_center_active_sum': 'DNase-seq level', 'histone_center_diff_log2_sum': 'Histone ChIP-seq change',
    'histone_center_active_sum': 'Histone ChIP-seq level', 'tf_center_diff_log2_sum': 'TF ChIP-seq change',
    'tf_center_active_sum': 'TF ChIP-seq level',
}

# The ClinVar VCF the examples are annotated from. Set GFME_CLINVAR_VCF to point at your own
# download; the default is where the repository's data README tells you to put it.
CLINVAR_VCF = Path(os.environ.get('GFME_CLINVAR_VCF', 'data/external/clinvar.vcf.gz'))


def locate(key):
    """(dataset, row) for a variant key, over the datasets that carry features."""
    for name in SOURCES:
        try:
            frame = io.load_dataset(name)
        except io.MissingData:
            continue
        hit = np.flatnonzero(frame['key'].to_numpy() == key)
        if len(hit):
            return name, int(hit[0])
    raise KeyError(f'{key} is in none of {SOURCES}')


def pick_missense(config, model_path):
    """A pathogenic missense variant the scalar models carry and AlphaGenome does not.

    Chosen by the attribution itself rather than by name: among correctly scored pathogenic
    missense variants in the test set, the one with the largest GPN-MSA contribution relative to
    the size of the AlphaGenome contribution.
    """
    X, feature_names, frame = io.load_matrix('clinvar_test', config)
    model, _ = models.load(model_path, config, feature_names)
    mask = np.flatnonzero((frame['consequence'] == 'missense').to_numpy()
                          & (frame['label'] == 1).to_numpy())
    scores = models.predict(model, X[mask])
    mask = mask[scores > 0.9]
    values = models.shap_values(model, X[mask])[:, :-1]

    ag = [i for i, n in enumerate(feature_names) if n.startswith(('scorer__', 'activity__'))]
    gpnmsa = np.asarray(values[:, feature_names.index('gpnmsa_neg')])
    alphagenome = np.abs(np.asarray(values[:, ag]).sum(axis=1))
    best = int(np.argmax(gpnmsa - alphagenome))
    return 'clinvar_test', int(mask[best])


def gene_names(keys):
    """GENEINFO for a handful of keys, read once from the ClinVar VCF."""
    wanted, out = set(keys), {}
    if not CLINVAR_VCF.exists():
        return out
    with gzip.open(CLINVAR_VCF, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            f = line.split('\t', 8)
            key = f'chr{f[0]}:{f[1]}:{f[3]}:{f[4]}'
            if key not in wanted or key in out:
                continue
            for field in f[7].split(';'):
                if field.startswith('GENEINFO='):
                    out[key] = field[len('GENEINFO='):].split(':')[0]
                    break
            if len(out) == len(wanted):
                break
    return out


def group_alphagenome(feature_names, values, top=10):
    """Three views of the AlphaGenome block, each a partition of its columns.

    By scorer: the 19 scorer columns on their own, named as the web tool names them, plus the
    728 per-biosample activity columns as one aggregate. By modality: the same 19 folded into
    their nine assay modalities, plus the 728
    per-biosample activity columns as one aggregate. By biosample: those 728 columns on their own,
    the largest few named and the rest netted. Each view sums to the AlphaGenome total, so the two
    are alternative decompositions of the same number rather than different quantities.
    """
    values = np.asarray(values)
    modality, seen = {}, set()
    for key, label in MODALITY.items():
        cols = [i for i, n in enumerate(feature_names) if n.startswith(f'scorer__{key}')]
        if cols:
            modality[f'AG {label}'] = float(values[cols].sum())
            seen.update(cols)

    activity = [i for i, n in enumerate(feature_names) if n.startswith('activity__')]
    ag_columns = {i for i, n in enumerate(feature_names)
                  if n.startswith(('scorer__', 'activity__'))}
    assert seen | set(activity) == ag_columns, 'not a partition of the AlphaGenome columns'

    modality[f'AG activity, all {len(activity)} biosamples'] = float(values[activity].sum())
    modality = dict(sorted(modality.items(), key=lambda kv: -abs(kv[1])))

    scorer = {SCORER_LABEL.get(n[len('scorer__'):], n[len('scorer__'):]): float(values[i])
              for i, n in enumerate(feature_names) if n.startswith('scorer__')}
    scorer[f'activity, all {len(activity)} biosamples'] = float(values[activity].sum())
    scorer = dict(sorted(scorer.items(), key=lambda kv: -abs(kv[1])))

    per_biosample = {feature_names[i][len('activity__'):]: float(values[i]) for i in activity}
    order = sorted(per_biosample, key=lambda k: -abs(per_biosample[k]))
    biosample = {k: per_biosample[k] for k in order[:top]}
    biosample[f'net of the other {len(order) - top} biosamples'] = sum(
        per_biosample[k] for k in order[top:])

    return modality, scorer, biosample, float(values[list(ag_columns)].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--name', default='shap_examples')
    parser.add_argument('--repick-missense', action='store_true',
                        help='re-derive the missense example instead of using the pinned key')
    args = parser.parse_args()

    cfg = config_module.override(config_module.load(), args.overrides)
    training_keys = set(io.load_dataset('clinvar_train')['key'])

    resolved = []
    for spec in EXAMPLES:
        repick = args.repick_missense and spec.get('note', '').startswith('protein-level')
        if repick:
            dataset, row = pick_missense(cfg, args.model)
            print(f'  missense example re-derived: row {row}', flush=True)
        else:
            dataset, row = locate(spec['key'])
        resolved.append({**spec, 'dataset': dataset, 'row': row})

    examples = []
    for dataset in dict.fromkeys(r['dataset'] for r in resolved):
        wanted = [r for r in resolved if r['dataset'] == dataset]
        X, feature_names, frame = io.load_matrix(dataset, cfg)
        model, _ = models.load(args.model, cfg, feature_names)
        index = [r['row'] for r in wanted]
        shap = models.shap_values(model, X[index])
        base = float(shap[0, -1])
        scores = models.predict(model, X[index])

        for j, spec in enumerate(wanted):
            values = shap[j, :-1]
            row = frame.iloc[spec['row']]
            blocks = {label: float(values[feature_names.index(name)])
                      for name, label in BLOCKS.items() if name in feature_names}
            by_modality, by_scorer, by_biosample, ag_total = group_alphagenome(feature_names,
                                                                              values)
            blocks = {'AlphaGenome': ag_total, **blocks}
            logodds = float(np.log(scores[j] / (1 - scores[j])))
            assert abs(sum(blocks.values()) + base - logodds) < 1e-3, 'attributions do not sum'
            examples.append({
                'key': row['key'], 'consequence': row['consequence'], 'status': spec['status'],
                'note': spec['note'], 'dataset': dataset, 'score': float(scores[j]),
                'in_training': row['key'] in training_keys, 'base_value_logodds': base,
                '_row': spec['row'],
                'blocks': blocks, 'alphagenome_by_modality': by_modality,
                'alphagenome_by_scorer': by_scorer,
                'alphagenome_by_biosample': by_biosample,
            })

    genes = gene_names([e['key'] for e in examples])
    declared = {(r['dataset'], r['row']): i for i, r in enumerate(resolved)}
    examples.sort(key=lambda e: declared[(e['dataset'], e['_row'])])
    for e in examples:
        e['gene'] = genes.get(e['key'], '')
        e.pop('_row')
        print(f'  {e["gene"]:8s} {e["key"]:24s} {e["consequence"]:10s} score {e["score"]:.4f}  '
              f'AlphaGenome {e["blocks"]["AlphaGenome"]:+.2f}  '
              f'GPN-MSA {e["blocks"].get("GPN-MSA", 0):+.2f}', flush=True)

    result = {'model': args.model, 'base_value_logodds': examples[0]['base_value_logodds'],
              'note': 'per-variant attribution, an illustration of tool output rather than a '
                      'result; nothing is fitted here',
              'examples': examples}
    io.write_result('16_shap', args.name, result, cfg)

    out = Path(__file__).resolve().parent.parent / 'results' / 'figures' / 'shap_examples.pdf'
    figures.shap_examples(result, cfg, out)


if __name__ == '__main__':
    main()
