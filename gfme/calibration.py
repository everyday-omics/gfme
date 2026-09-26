"""ACMG/AMP PP3 and BP4 evidence tiers, by Pejaver's local likelihood-ratio method.

A pathogenicity score becomes usable evidence only once a threshold has a calibrated strength
attached to it. Pejaver et al. (2022) set the framework: convert a score into a likelihood ratio,
and award Supporting, Moderate, Strong or Very Strong where that ratio clears a target derived
from a prior probability of pathogenicity.

Three properties of this implementation matter, and each corrects a way the estimate can flatter
itself:

* **Local, not cumulative.** lr+(s) is estimated inside an adaptive window around each score, not
  over everything above it. A cumulative statistic reports the average of a region rather than the
  evidence a variant scoring *there* actually carries.
* **A conservative bound, over the whole tail.** A threshold is awarded only if the one-sided
  bootstrap bound clears the target at every score beyond it, not merely at the threshold itself.
  One lucky window therefore cannot open a tier.
* **Fitted out of sample.** Thresholds come from the training out-of-fold predictions. The test
  set is used only to check thresholds that were already fixed. Reporting the likelihood ratios a
  threshold was chosen to achieve is not a validation.

A window containing no benign variants would give an infinite ratio, and a bootstrap percentile
over infinities returns NaN, which then fails the comparison against the target — so perfect
separation would be scored as calibration failure, in exactly the categories that separate best.
A Haldane-Anscombe continuity correction keeps the ratio finite and errs conservative.
"""
from __future__ import annotations

import numpy as np


def _targets_and_prior(config, category):
    priors, targets = config['calibration']['priors'], config['calibration']['targets']
    prior = priors.get(category, priors['default'])
    target = targets.get(category, targets['default'])
    return prior, dict(target)


def local_lr(scores, labels, grid, min_window, bootstrap_iterations, seed=0):
    """lr+(s) in an adaptive window around each grid point, with a bootstrap interval.

    Returns (point estimate, 5th percentile, 95th percentile) per grid point.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    n_pathogenic = max(int((sorted_labels == 1).sum()), 1)
    n_benign = max(int((sorted_labels == 0).sum()), 1)
    rng = np.random.default_rng(seed)

    point = np.full(len(grid), np.nan)
    low = np.full(len(grid), np.nan)
    high = np.full(len(grid), np.nan)

    for i, score in enumerate(grid):
        centre = np.searchsorted(sorted_scores, score)
        left = right = centre
        while (right - left) < min_window and (left > 0 or right < len(sorted_scores)):
            reach_left = left > 0 and (right >= len(sorted_scores) or
                                       (score - sorted_scores[left - 1]) <=
                                       (sorted_scores[right] - score))
            if reach_left:
                left -= 1
            else:
                right += 1
        window = sorted_labels[left:right]
        if not len(window):
            continue

        sensitivity = ((window == 1).sum() + 0.5) / (n_pathogenic + 1)
        false_positive = ((window == 0).sum() + 0.5) / (n_benign + 1)
        point[i] = sensitivity / false_positive

        draw = rng.integers(0, len(window), size=(bootstrap_iterations, len(window)))
        resampled = window[draw]
        boot_sensitivity = ((resampled == 1).sum(axis=1) + 0.5) / (n_pathogenic + 1)
        boot_false_positive = ((resampled == 0).sum(axis=1) + 0.5) / (n_benign + 1)
        ratios = boot_sensitivity / boot_false_positive
        low[i], high[i] = np.percentile(ratios, [5, 95])

    return point, low, high


def thresholds(grid, low, high, targets):
    """The score at which each tier opens.

    PP3: the smallest threshold whose lower bound clears the target for every score above it.
    BP4: the largest threshold whose upper bound stays under the reciprocal for every score below.
    """
    pp3, bp4 = {}, {}
    for tier, target in targets.items():
        clears = np.where(np.isfinite(low), low >= target, False)
        pp3[tier] = next((float(grid[i]) for i in range(len(grid))
                          if clears[i:].size and clears[i:].all()), None)
        under = np.where(np.isfinite(high), high <= 1.0 / target, False)
        bp4[tier] = next((float(grid[i]) for i in range(len(grid) - 1, -1, -1)
                          if under[:i + 1].size and under[:i + 1].all()), None)
    return pp3, bp4


def interval_lr(scores, labels, low_edge, high_edge):
    """The likelihood ratio of the variants falling in [low_edge, high_edge).

    Returned as None when no benign variant falls in the band: the ratio is then a division by
    zero, not a measured value, and a tier resting on it is unvalidated rather than validated.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    inside = (scores >= low_edge) & (scores < high_edge)
    n_pathogenic = max(int((labels == 1).sum()), 1)
    n_benign = max(int((labels == 0).sum()), 1)
    sensitivity = ((labels == 1) & inside).sum() / n_pathogenic
    false_positive = ((labels == 0) & inside).sum() / n_benign
    ratio = None if false_positive == 0 else float(sensitivity / false_positive)
    return ratio, int(inside.sum()), int(((labels == 1) & inside).sum())


def calibrate(scores, labels, consequence, config, categories):
    """Evidence tiers per consequence category, fitted on out-of-fold predictions."""
    settings = config['calibration']
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    consequence = np.asarray(consequence)

    out = {}
    for category in categories:
        mask = (consequence == category) & np.isfinite(scores)
        if mask.sum() < settings['min_window']:
            out[category] = {'n': int(mask.sum()), 'note': 'too few variants to calibrate'}
            continue
        prior, targets = _targets_and_prior(config, category)
        category_scores, category_labels = scores[mask], labels[mask]
        grid = np.unique(np.quantile(category_scores,
                                     np.linspace(0, 1, settings['grid_quantiles'])))
        point, low, high = local_lr(category_scores, category_labels, grid,
                                    settings['min_window'], settings['bootstrap_iterations'],
                                    config['seed'])
        pp3, bp4 = thresholds(grid, low, high, targets)
        out[category] = {
            'n': int(mask.sum()),
            'n_pathogenic': int((category_labels == 1).sum()),
            'prior': prior,
            'targets': targets,
            'PP3': pp3,
            'BP4': bp4,
            # The curve, not only the thresholds: a different prior implies different targets, so
            # the boundaries can be re-read off this rather than being fixed to one prior.
            'curve': {'score': [float(g) for g in grid],
                      'lr': [None if not np.isfinite(v) else float(v) for v in point],
                      'lr_low': [None if not np.isfinite(v) else float(v) for v in low],
                      'lr_high': [None if not np.isfinite(v) else float(v) for v in high]},
        }
    return out


def validate(calibrated, scores, labels, consequence, categories):
    """Check fixed thresholds against a held-out cohort.

    A tier whose band contains no benign variant reports `None`, and is reported as unvalidated.
    Marking it as met would be reading a division by zero as evidence.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    consequence = np.asarray(consequence)
    report = {}

    for category in categories:
        entry = calibrated.get(category, {})
        if 'PP3' not in entry:
            continue
        mask = (consequence == category) & np.isfinite(scores)
        category_scores, category_labels = scores[mask], labels[mask]
        rows = {}
        ordered = [t for t in ('Supporting', 'Moderate', 'Strong', 'Very_Strong')
                   if entry['PP3'].get(t) is not None]
        for i, tier in enumerate(ordered):
            low_edge = entry['PP3'][tier]
            high_edge = entry['PP3'][ordered[i + 1]] if i + 1 < len(ordered) else np.inf
            ratio, n, n_pathogenic = interval_lr(category_scores, category_labels,
                                                 low_edge, high_edge)
            rows[f'PP3_{tier}'] = {
                'threshold': low_edge, 'test_LR': ratio, 'n': n, 'n_pathogenic': n_pathogenic,
                'target': entry['targets'][tier],
                'status': 'unvalidated (no benign variant in band)' if ratio is None
                          else ('met' if ratio >= entry['targets'][tier] else 'below target'),
            }
        # BP4 mirrors PP3 with the score axis reversed: Supporting opens at the highest score and
        # each stronger tier at a lower one, so a tier's band runs from the next stronger threshold
        # up to its own, and it is met when the band's likelihood ratio stays at or under the
        # reciprocal of the target. A band with no pathogenic variant is unvalidated for the same
        # reason as a PP3 band with no benign one: its ratio is a zero count, not a measurement.
        bp4 = entry.get('BP4', {})
        ordered = [t for t in ('Supporting', 'Moderate', 'Strong', 'Very_Strong')
                   if bp4.get(t) is not None]
        for i, tier in enumerate(ordered):
            high_edge = bp4[tier]
            low_edge = bp4[ordered[i + 1]] if i + 1 < len(ordered) else -np.inf
            ratio, n, n_pathogenic = interval_lr(category_scores, category_labels,
                                                 low_edge, high_edge)
            reciprocal = 1.0 / entry['targets'][tier]
            rows[f'BP4_{tier}'] = {
                'threshold': high_edge, 'test_LR': ratio, 'n': n, 'n_pathogenic': n_pathogenic,
                'target': reciprocal,
                'status': 'unvalidated (no pathogenic variant in band)' if n_pathogenic == 0
                          else ('unvalidated (no benign variant in band)' if ratio is None
                                else ('met' if ratio <= reciprocal else 'below target')),
            }
        report[category] = rows
    return report
