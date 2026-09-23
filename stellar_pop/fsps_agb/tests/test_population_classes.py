import numpy as np

from population_classes import (
    CLASS_NAMES,
    add_measurement_noise,
    assign_classes,
    completeness_purity,
    cross_validated_metrics,
    grouped_folds,
    is_post_starburst,
    knn_predict,
)


def test_assign_classes_follows_manuscript_rules():
    recent = np.array([2e-10, 5e-12, 1e-12, 5e-11, 0.0]) * 1e9  # per Gyr
    previous = np.array([1e-10, 2e-10, 5e-11, 2e-10, 0.0]) * 1e9
    codes = assign_classes(recent, previous)
    assert [CLASS_NAMES[c] for c in codes] == [
        "star_forming",
        "rapid_quenching",
        "quiescent",
        "transitional",
        "quiescent",
    ]
    assert is_post_starburst(recent, previous).tolist() == [False, True, False, False, False]


def test_is_post_starburst_uses_absolute_recent_ssfr_not_ratio():
    # previous = 5e-10 /yr, recent = 2e-11 /yr -> ratio R = 0.04 < 0.1, so assign_classes
    # calls this rapid_quenching, but recent (2e-11) is not below the 1e-11 /yr quiescent
    # threshold, so is_post_starburst must be False.
    recent = np.array([2e-11]) * 1e9
    previous = np.array([5e-10]) * 1e9
    assert is_post_starburst(recent, previous).tolist() == [False]
    assert CLASS_NAMES[assign_classes(recent, previous)[0]] == "rapid_quenching"


def test_noise_has_requested_scale():
    rng = np.random.default_rng(1)
    features = np.zeros((20000, 2))
    noisy = add_measurement_noise(features, np.array([0.05, 0.01]), rng)
    assert np.allclose(noisy.std(axis=0), [0.05, 0.01], rtol=0.05)


def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(40), 5)
    folds = grouped_folds(groups, 5, np.random.default_rng(0))
    assert folds.shape == groups.shape and set(folds) == set(range(5))
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1


def test_knn_predict_recovers_separated_blobs():
    rng = np.random.default_rng(2)
    a = rng.normal([0, 0], 0.1, size=(200, 2))
    b = rng.normal([1, 1], 0.1, size=(200, 2))
    features = np.vstack([a, b])
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    test = np.array([[0.05, -0.05], [0.95, 1.05]])
    assert knn_predict(
        features, labels, test, k=5, feature_scales=np.array([1.0, 1.0])
    ).tolist() == [0, 1]


def test_knn_predict_with_k_one_returns_one_prediction_per_query():
    rng = np.random.default_rng(4)
    a = rng.normal([0, 0], 0.05, size=(50, 2))
    b = rng.normal([3, 3], 0.05, size=(50, 2))
    features = np.vstack([a, b])
    labels = np.r_[np.zeros(50, int), np.ones(50, int)]
    test = np.array([[0.1, 0.1], [2.9, 2.9], [0.0, -0.05]])
    predictions = knn_predict(features, labels, test, k=1, feature_scales=np.array([1.0, 1.0]))
    assert predictions.shape == (3,)
    assert predictions.tolist() == [0, 1, 0]


def test_completeness_purity():
    true = np.array([1, 1, 1, 0, 0])
    pred = np.array([1, 1, 0, 1, 0])
    completeness, purity = completeness_purity(true, pred, positive_class=1)
    assert np.isclose(completeness, 2 / 3) and np.isclose(purity, 2 / 3)


def test_cross_validated_metrics_on_separable_data():
    rng = np.random.default_rng(3)
    n = 400
    features = np.vstack([rng.normal([0, 0], 0.1, (n, 2)), rng.normal([2, 2], 0.1, (n, 2))])
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]
    groups = np.arange(2 * n) // 4
    result = cross_validated_metrics(features, labels, groups, np.array([1.0, 1.0]), k=5)
    assert result["completeness_mean"] > 0.95 and result["purity_mean"] > 0.95
    assert np.asarray(result["confusion"]).shape == (4, 4)
