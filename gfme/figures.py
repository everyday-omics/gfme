"""Figures.

One style, one palette, one function per figure. A tool is the same colour in every figure it
appears in, which is what makes the figures readable as a set rather than as separate pictures.

Identity is never colour alone. There are more tools than distinguishable hues, so each series
also carries a line style and a marker, and tools that score only some categories are drawn as
isolated points rather than connected lines — which is also the honest depiction, since a line
between two categories a tool does not score would imply a value it never produced.

The palette is the Okabe-Ito colourblind-safe set in an order validated for adjacent-pair
separation under deuteranopia and tritanopia. It lives in the config, not here.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

import numpy as np                       # noqa: E402

SHORT = {'intronic': 'Intronic', 'synonymous': 'Synonymous', 'non-coding': 'Non-coding',
         'splice': 'Splice', 'LoF': 'LoF', 'other': 'Other', 'missense': 'Missense',
         'in-frame': 'In-frame'}

# The two consequence-restricted tools are drawn as points in the palette's yellow, which no line
# uses, with a black edge. They share that yellow with each other, so each also gets its own
# marker — identity is never colour alone.
POINT_MARKER = {'REVEL': 'D', 'AlphaMissense': '^'}

KEY = {'FM Ensemble': 'ensemble', 'AG-CatBoost': 'ag_catboost', 'CADD': 'cadd',
       'GPN-MSA': 'gpnmsa', 'Evo2': 'evo2', 'phyloP': 'phylop', 'GERP++': 'gerp',
       'phastCons': 'phastcons', 'DANN': 'dann', 'SpliceAI': 'spliceai', 'REVEL': 'revel',
       'AlphaMissense': 'alphamissense', 'MSA gap': 'msa_gap'}


def style(config):
    palette = config['palette']
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.fontsize': 7.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': palette['neutral'], 'axes.linewidth': 0.8,
        'grid.color': palette['grid'], 'grid.linewidth': 0.6, 'grid.alpha': 0.9,
        'legend.frameon': False, 'font.family': 'DejaVu Sans',
    })


def colour(config, label):
    return config['palette'].get(KEY.get(label, 'neutral'), config['palette']['neutral'])


def per_category(result, config, path, title=None):
    """Per-category AUROC for every tool on one cohort.

    Genome-wide tools are connected across categories; tools that address only some categories
    are isolated points, so the picture never implies a score that was not produced. Macro-AUROC
    in the legend is over the categories each tool actually scores.
    """
    style(config)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _draw_per_category(ax, result, config)
    _legend(ax, *ax.get_legend_handles_labels())
    if title:
        ax.set_title(title, loc='left')
    _save(fig, path)
    return path


def per_category_panels(results, config, path, titles=None, orders=None):
    """The same picture for two cohorts, as panels of one figure.

    SNVs and indels are scored by overlapping but not identical tool sets over overlapping but
    not identical categories, so each panel keeps its own axes. They share one palette and one
    legend: a tool has the same colour, marker and line style in both panels, and is named once
    for the figure rather than twice. Per-panel macro-AUROC would make the shared legend
    ambiguous, so it is left to the table.
    """
    style(config)
    fig, axes = plt.subplots(len(results), 1, figsize=(8.2, 4.4 * len(results)))
    handles, labels = [], []
    for i, (ax, result) in enumerate(zip(np.atleast_1d(axes), results)):
        _draw_per_category(ax, result, config, order=orders[i] if orders else None,
                           show_macro=False)
        ax.set_title(f'({chr(97 + i)})' + (f' {titles[i]}' if titles else ''),
                     loc='left', fontweight='bold')
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:                  # union, in first-appearance order
                handles.append(handle)
                labels.append(label)
    # Anchor the shared legend to the right edge of the axes, not of the figure: the default
    # subplot margin would otherwise leave a band of white space between the panels and it.
    fig.subplots_adjust(hspace=0.3)
    edge = max(ax.get_position().x1 for ax in np.atleast_1d(axes))
    _legend(fig, handles, labels, title=None, anchor=(edge + 0.01, 0.5))
    _save(fig, path)
    return path


def _draw_per_category(ax, result, config, order=None, show_macro=True):
    categories = list(order) if order else [c for c in result['categories']]
    assert set(categories) == set(result['categories']), 'display order must cover every category'
    x = np.arange(len(categories))
    genome_wide = set(config['genome_wide_tools'])
    dashed = set(config['dashed_tools'])

    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

    for label in result['row_order']:
        row = result['rows'][label]
        values = [row['per_category'].get(c, np.nan) for c in categories]
        if not np.isfinite(values).any():
            continue
        macro = row['macro']
        legend = f'{label} ({macro:.3f})' if macro is not None and show_macro else label
        c = colour(config, label)
        emphasis = label == 'FM Ensemble'
        if label in genome_wide:
            ax.plot(x, values, color=c, linewidth=2.4 if emphasis else 1.6,
                    linestyle='--' if label in dashed else '-',
                    marker='o' if emphasis else 's', markersize=7 if emphasis else 5,
                    markeredgecolor='white', markeredgewidth=1.0,
                    label=legend, zorder=6 if emphasis else 3)
        else:
            ax.plot(x, values, color=c, linestyle='none',
                    marker=POINT_MARKER.get(label, 'D'), markersize=6,
                    markeredgecolor=config['palette'].get('point_edge', 'white'),
                    markeredgewidth=0.8, label=legend, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in categories])
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color=config['palette']['neutral'], linewidth=0.7, linestyle=':', zorder=1)


def _legend(target, handles, labels, title='Macro-AUROC', anchor=(1.01, 0.5)):
    # The handle is long enough to show a full dash period: at the matplotlib default a dashed
    # and a solid line of the same colour render as the same short stub.
    target.legend(handles, labels, loc='center left', bbox_to_anchor=anchor, ncol=1,
                  title=title, title_fontsize=8, handlelength=3.6, handletextpad=0.7)


def combination_matrix(result, config, path, labels=None):
    """The whole combination search: every configuration, sorted by macro-AUROC.

    Two panels sharing an x axis. Above, macro-AUROC across all configurations, with the best
    marked. Below, a binary inclusion matrix — one row per foundation model, ordered by marginal
    contribution — so a model that belongs in the strong configurations shows as a solid band on
    the left and one that does not shows as noise.

    Every configuration is plotted rather than a top-N: the shape of the whole distribution is
    the point, and a truncated view would hide how quickly the field falls away.
    """
    style(config)
    rows = result['all_combinations']
    marginal = result['marginal_contribution']
    order = sorted(marginal, key=marginal.get, reverse=True)
    names = labels or {}

    macros = np.array([r['macro'] for r in rows])
    x = np.arange(len(macros))
    grid = np.array([[1 if m in r['combination'].split('+') else 0 for r in rows]
                     for m in order], dtype=int)

    fig, (top, matrix) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.06})
    accent = config['palette']['ensemble']
    top.fill_between(x, macros, alpha=0.35, color=accent)
    top.plot(x, macros, color=accent, linewidth=0.8)
    top.axhline(macros[0], color=config['palette']['evo2'], linestyle='--', linewidth=1.0,
                alpha=0.8)
    top.set_ylabel('Macro-AUROC')
    top.set_xlim(0, len(macros) - 1)
    top.set_ylim(0.5, None)
    for spine in ('top', 'right', 'bottom'):
        top.spines[spine].set_visible(False)
    top.tick_params(axis='x', length=0)

    matrix.imshow(grid, aspect='auto', interpolation='nearest',
                  cmap=matplotlib.colors.ListedColormap(['#f2f2f2', accent]), vmin=0, vmax=1)
    matrix.set_yticks(range(len(order)))
    matrix.set_yticklabels([names.get(m, m) for m in order])
    matrix.set_xlabel('Configuration (sorted by macro-AUROC)')
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.tick_params(axis='y', length=0)
    _save(fig, path)
    return path


def stratified(result, config, path, xlabel, title=None):
    """AUROC by a stratum (indel size, type), with the sample size beneath it.

    The counts share the x axis rather than a second y axis: a bin holding forty variants and a
    bin holding four thousand should not be read as equally reliable, and two scales on one frame
    invites exactly that.
    """
    style(config)
    strata = result['strata']
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12})
    x = np.arange(len(strata))
    for i, (series, values) in enumerate(result['series'].items()):
        c = list(config['palette'].values())[i % 6]
        ax.plot(x, [values.get(s, {}).get('auroc', np.nan) for s in strata], color=c,
                linewidth=2.0, marker='o', markersize=6, markeredgecolor='white',
                markeredgewidth=1.0, label=f'{series} ({result["aggregate"][series]:.3f})')
        counts.step(x, [values.get(s, {}).get('n', 0) for s in strata], where='mid',
                    color=c, linewidth=1.4)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel('AUROC')
    ax.legend(title='Aggregate AUROC', title_fontsize=8)
    if title:
        ax.set_title(title, loc='left')
    counts.set_ylabel('n')
    counts.set_xlabel(xlabel)
    counts.set_xticks(x)
    counts.set_xticklabels(strata)
    counts.grid(axis='y', zorder=0)
    counts.set_axisbelow(True)
    _save(fig, path)
    return path


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f'  -> {path.name} (+ .png)', flush=True)


def predictive_value_curve(result, config, path):
    """Predictive values against assumed pre-test probability, for every tool.

    Two facets, because the two quantities behave oppositely and the contrast is the finding:
    positive predictive value collapses as prevalence falls, negative predictive value rises
    towards one. Plotting every tool rather than the ensemble alone shows the collapse is a
    property of rare events, not of any one model — the ensemble's own curve in isolation would
    read as a weakness of the ensemble.

    The x axis is logarithmic: the clinically relevant priors span three orders of magnitude and
    sit at the low end.
    """
    style(config)
    priors = [float(p) for p in result['priors']]
    tools = result['all_tools_pooled']
    order = sorted(tools, key=lambda k: -(tools[k]['ppv'][f'{priors[-1]:g}'] or 0))
    reported = result['reported_at']

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharex=True)
    for ax, metric, label in zip(axes, ('ppv', 'npv'),
                                 ('Positive predictive value', 'Negative predictive value')):
        ax.grid(zorder=0)
        ax.set_axisbelow(True)
        for name in order:
            sensitivity = tools[name]['sensitivity']
            specificity = tools[name]['specificity']
            values = []
            for prior in priors:
                tp, fn = sensitivity * prior, (1 - sensitivity) * prior
                fp, tn = (1 - specificity) * (1 - prior), specificity * (1 - prior)
                values.append(tp / (tp + fp) if metric == 'ppv' else tn / (tn + fn))
            emphasis = name == 'FM Ensemble'
            ax.plot(priors, values, color=colour(config, name),
                    linewidth=2.4 if emphasis else 1.3,
                    linestyle='-' if emphasis else '--',
                    marker='o' if emphasis else None, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=name, zorder=6 if emphasis else 3,
                    alpha=1.0 if emphasis else 0.85)
        ax.set_xscale('log')
        # Percent labels on the three decades the priors span: a reader thinks in "1 in 100",
        # not in 0.01, and the unlabelled minor ticks still show where the decades divide.
        ax.set_xticks([0.001, 0.01, 0.1])
        ax.set_xticklabels(['0.1%', '1%', '10%'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('Assumed pre-test probability')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        # The x tick already reads 10%, so the line carries no label of its own; the caption says
        # what it marks.
        ax.axvline(reported, color=config['palette']['neutral'], linewidth=0.8, linestyle=':',
                   zorder=1)
    axes[0].legend(loc='upper left', ncol=2, fontsize=7)
    _save(fig, path)
    return path

def shap_examples(result, config, path, per_page=2):
    """Per-variant attributions for a few example variants, one column per variant.

    Top, the four inputs the score decomposes into; middle, the AlphaGenome block over its 19
    scorer columns, named as the web tool names them, with the 728 biosample columns as one
    aggregate; bottom, the same block over the biosample columns, the ten largest named and the
    rest netted. Bars start at the model base value and are cumulative, so the reading is the
    same as the arithmetic: each contribution shifts the log-odds, and the diamond marks where the
    variant ends up.

    Red is a shift towards pathogenic and blue away from it. This is the one figure in the set
    where colour carries direction rather than identity, so it uses the diverging pair and not
    the categorical palette.

    Drawn for landscape pages, `per_page` variants on each: the first page is written to `path`
    and each further page to `path` with `_2`, `_3`, ... appended to the stem. Two variants per
    page keep the 7.5 pt labels at print size with the full column names.
    """
    style(config)
    examples = result['examples']
    base = result['base_value_logodds']
    up, down = config['palette']['evo2'], config['palette']['ensemble']
    path = Path(path)

    n_scorer = max(len(e['alphagenome_by_scorer']) for e in examples)
    n_bio = max(len(e['alphagenome_by_biosample']) for e in examples)
    written = []
    for page, start in enumerate(range(0, len(examples), per_page)):
        chunk = examples[start:start + per_page]
        fig, axes = plt.subplots(3, len(chunk), figsize=(4.4 * len(chunk), 6.3), squeeze=False,
                                 gridspec_kw={'height_ratios': [4 + 1.5, n_scorer + 1.5,
                                                                n_bio + 1.5]})
        for col, example in zip(axes.T, chunk):
            total = example['blocks']['AlphaGenome']
            blocks = sorted(example['blocks'].items(), key=lambda kv: -abs(kv[1]))
            _waterfall(col[0], base, blocks, up, down,
                       f'{example["gene"]}  {example["key"]}\n{example["consequence"]}, '
                       f'{example["status"]}\n'
                       + '\n'.join(textwrap.wrap(example['note'], 72)), final=example['score'])
            _waterfall(col[1], base, list(example['alphagenome_by_scorer'].items()), up, down,
                       f'AlphaGenome by scorer (19)\n(total {total:+.2f})')
            _waterfall(col[2], base, list(example['alphagenome_by_biosample'].items()), up, down,
                       f'AlphaGenome by biosample (top {n_bio - 1})\n(total {total:+.2f})')
        fig.supxlabel('model output (log-odds)', fontsize=7.5, y=0.005)
        fig.subplots_adjust(hspace=0.62, wspace=1.3)
        out = path if page == 0 else path.with_name(f'{path.stem}_{page + 1}{path.suffix}')
        _save(fig, out)
        written.append(out)
    return written


def _waterfall(ax, start, steps, up, down, title, final=None, note=None):
    """Cumulative bars, largest contribution first, reading top to bottom."""
    cumulative = start
    ends = []
    for i, (label, value) in enumerate(steps):
        ax.barh(i, value, left=cumulative, height=0.62, zorder=3,
                color=up if value > 0 else down, edgecolor='white', linewidth=0.6)
        cumulative += value
        ends.append(cumulative)
        ax.plot([cumulative] * 2, [i - 0.31, i + 1.31], color='0.65', lw=0.7, ls=':', zorder=2)

    # Always to the right of the segment, whichever way it points: a label placed at the far end
    # of a negative bar lands on the category name and becomes unreadable.
    span = (max(ends + [start]) - min(ends + [start])) or 1.0
    starts = [start] + ends[:-1]
    for i, ((label, value), left, end) in enumerate(zip(steps, starts, ends)):
        ax.text(max(left, end) + 0.02 * span, i, f'{value:+.2f}', va='center', ha='left',
                fontsize=7, color=up if value > 0 else down, zorder=4)

    ax.axvline(start, color='0.55', lw=1.0, zorder=1)
    ax.scatter([cumulative], [-0.55], marker='D', s=14, color='0.15', zorder=5)
    if final is not None:
        ax.text(cumulative, -0.95, f'score {final:.4f}', ha='center', fontsize=8,
                fontweight='bold')
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([_short(s[0]) for s in steps], fontsize=7.5)
    ax.set_ylim(len(steps) - 0.3, -1.3)          # inverted: the first step sits at the top
    ax.margins(x=0.28)
    ax.set_title(title, fontsize=8.5, pad=14, loc='left')
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', length=0, labelsize=7.5)
    if note:
        ax.text(0.0, -0.22, '\n'.join(textwrap.wrap(note, 40)), transform=ax.transAxes,
                ha='left', va='top', fontsize=6.5, color='0.45', style='italic')


def _short(label, width=36):
    return label if len(label) <= width else label[:width - 1] + '\u2026'
