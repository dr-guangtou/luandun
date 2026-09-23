"""Manuscript sSFR classification rules and a grouped-cross-validation kNN classifier.

Recent and previous specific star formation rates (sSFR) are passed in per Gyr and
converted to per yr internally (divided by 1e9) to match the manuscript's thresholds,
which are defined in per-yr units.
"""

import numpy as np
from scipy.spatial import cKDTree

CLASS_NAMES = ("star_forming", "rapid_quenching", "transitional", "quiescent")
STAR_FORMING, RAPID_QUENCHING, TRANSITIONAL, QUIESCENT = range(4)

_PREVIOUS_QUIESCENT_THRESHOLD_PER_YR = 1e-10
_RECENT_QUIESCENT_THRESHOLD_PER_YR = 1e-11
_PREVIOUS_RAPID_QUENCHING_THRESHOLD_PER_YR = 1e-10
_RATIO_RAPID_QUENCHING_THRESHOLD = 0.1
_RATIO_STAR_FORMING_THRESHOLD = 1.0


def _ssp_ratio(recent_per_yr, previous_per_yr):
    return np.divide(
        recent_per_yr,
        previous_per_yr,
        out=np.zeros_like(recent_per_yr, dtype=float),
        where=previous_per_yr > 0,
    )


def is_post_starburst(recent_ssfr_per_gyr, previous_ssfr_per_gyr):
    """Rapid-quenching condition, in per-Gyr sSFR inputs.

    True where the previous sSFR was actively star-forming and the recent sSFR
    dropped to below 10% of it.
    """
    recent_per_yr = np.asarray(recent_ssfr_per_gyr, dtype=float) / 1e9
    previous_per_yr = np.asarray(previous_ssfr_per_gyr, dtype=float) / 1e9
    ratio = _ssp_ratio(recent_per_yr, previous_per_yr)
    return (previous_per_yr > _PREVIOUS_RAPID_QUENCHING_THRESHOLD_PER_YR) & (
        ratio < _RATIO_RAPID_QUENCHING_THRESHOLD
    )


def assign_classes(recent_ssfr_per_gyr, previous_ssfr_per_gyr):
    """Assign one of the four manuscript classes to each sample, in per-Gyr sSFR inputs.

    Rules are applied in order (quiescent, rapid-quenching, star-forming, with
    transitional as the default) so each sample receives exactly one code, matching
    `CLASS_NAMES` = ("star_forming", "rapid_quenching", "transitional", "quiescent").
    """
    recent_per_yr = np.asarray(recent_ssfr_per_gyr, dtype=float) / 1e9
    previous_per_yr = np.asarray(previous_ssfr_per_gyr, dtype=float) / 1e9
    ratio = _ssp_ratio(recent_per_yr, previous_per_yr)

    quiescent = (previous_per_yr < _PREVIOUS_QUIESCENT_THRESHOLD_PER_YR) & (
        recent_per_yr < _RECENT_QUIESCENT_THRESHOLD_PER_YR
    )
    rapid_quenching = (previous_per_yr > _PREVIOUS_RAPID_QUENCHING_THRESHOLD_PER_YR) & (
        ratio < _RATIO_RAPID_QUENCHING_THRESHOLD
    )
    star_forming = ratio > _RATIO_STAR_FORMING_THRESHOLD

    codes = np.full(recent_per_yr.shape, TRANSITIONAL, dtype=int)
    codes[star_forming] = STAR_FORMING
    codes[rapid_quenching] = RAPID_QUENCHING
    codes[quiescent] = QUIESCENT
    return codes


def add_measurement_noise(features, sigmas, rng):
    """Add independent Gaussian noise with per-feature standard deviations `sigmas`."""
    features = np.asarray(features, dtype=float)
    return features + rng.normal(0.0, sigmas, features.shape)


def grouped_folds(groups, n_folds, rng):
    """Assign each sample to one of `n_folds` folds without splitting any group.

    Unique group ids are shuffled and distributed round-robin (group i -> fold i mod
    n_folds), then mapped back onto the samples.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    shuffled = rng.permutation(unique_groups)
    group_to_fold = {group: i % n_folds for i, group in enumerate(shuffled)}
    return np.array([group_to_fold[group] for group in groups])


def knn_predict(features, labels, query, k, feature_scales):
    """Predict labels for `query` by majority vote among the k nearest training points.

    Features are divided by `feature_scales` before distances are computed. Ties in
    the vote are broken by the lowest class code (via `np.bincount`).
    """
    features = np.asarray(features, dtype=float) / feature_scales
    query = np.asarray(query, dtype=float) / feature_scales
    labels = np.asarray(labels)

    tree = cKDTree(features)
    _, neighbor_indices = tree.query(query, k=k)
    neighbor_indices = np.atleast_2d(neighbor_indices)
    neighbor_labels = labels[neighbor_indices]

    predictions = np.empty(neighbor_labels.shape[0], dtype=labels.dtype)
    for i, row in enumerate(neighbor_labels):
        predictions[i] = np.argmax(np.bincount(row))
    return predictions


def completeness_purity(true, pred, positive_class):
    """Completeness (TP / (TP + FN)) and purity (TP / (TP + FP)) for `positive_class`.

    Returns `float("nan")` for either quantity when its denominator is zero.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    true_positive = np.sum((true == positive_class) & (pred == positive_class))
    false_negative = np.sum((true == positive_class) & (pred != positive_class))
    false_positive = np.sum((true != positive_class) & (pred == positive_class))

    completeness_denominator = true_positive + false_negative
    purity_denominator = true_positive + false_positive
    completeness = (
        true_positive / completeness_denominator if completeness_denominator > 0 else float("nan")
    )
    purity = true_positive / purity_denominator if purity_denominator > 0 else float("nan")
    return completeness, purity


def cross_validated_metrics(features, labels, groups, feature_scales, k, positive_class=1):
    """Grouped k-fold cross-validation of `knn_predict`.

    `groups` gives each sample's fold membership (e.g. produced by `grouped_folds`).
    For each fold, train on the remaining folds and predict the held-out fold,
    accumulating a 4x4 confusion matrix `confusion[true, pred]` and per-fold
    completeness/purity of `positive_class`. Returns their means and standard
    deviations (`ddof=0`, NaN-aware since a fold without any `positive_class`
    member yields `completeness_purity` = NaN) plus the summed confusion matrix
    as a nested list.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    completeness_per_fold = []
    purity_per_fold = []

    for fold in np.unique(groups):
        test_mask = groups == fold
        train_mask = ~test_mask
        predictions = knn_predict(
            features[train_mask], labels[train_mask], features[test_mask], k, feature_scales
        )
        true_fold = labels[test_mask]
        for true_class, pred_class in zip(true_fold, predictions, strict=True):
            confusion[true_class, pred_class] += 1
        completeness, purity = completeness_purity(true_fold, predictions, positive_class)
        completeness_per_fold.append(completeness)
        purity_per_fold.append(purity)

    return {
        "completeness_mean": float(np.nanmean(completeness_per_fold)),
        "completeness_std": float(np.nanstd(completeness_per_fold, ddof=0)),
        "purity_mean": float(np.nanmean(purity_per_fold)),
        "purity_std": float(np.nanstd(purity_per_fold, ddof=0)),
        "confusion": confusion.tolist(),
    }
