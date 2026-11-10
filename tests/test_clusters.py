"""Tests for region-aware cluster labeling and filtering."""
import numpy as np

from shivai.postprocessing.clusters import (
    get_clusters_and_filter_image,
    label_clusters,
    label_region_aware,
)


def test_shared_background_is_split_between_regions():
    prediction = np.ones((1, 1, 9), dtype=int)
    regions = np.zeros_like(prediction)
    regions[..., :2] = 10
    regions[..., -2:] = 20

    labels, count = label_region_aware(prediction, regions, [10, 20])

    assert count == 2
    assert np.all(labels > 0)
    assert labels[0, 0, 0] != labels[0, 0, -1]
    assert labels[0, 0, 2] == labels[0, 0, 0]
    assert labels[0, 0, -3] == labels[0, 0, -1]
    reversed_labels, _ = label_region_aware(prediction, regions, [20, 10])
    np.testing.assert_array_equal(labels, reversed_labels)


def test_shared_background_can_touch_three_regions():
    prediction = np.zeros((1, 9, 9), dtype=int)
    prediction[0, 4, :] = 1
    prediction[0, :5, 4] = 1
    regions = np.zeros_like(prediction)
    regions[0, 4, 0] = 1
    regions[0, 4, -1] = 2
    regions[0, 0, 4] = 3

    labels, count = label_region_aware(prediction, regions, [1, 2, 3])

    assert count == 3
    np.testing.assert_array_equal(labels > 0, prediction > 0)
    assert len(set(labels[regions > 0])) == 3
    assert labels[0, 4, 1] == labels[0, 4, 0]
    assert labels[0, 4, -2] == labels[0, 4, -1]
    assert labels[0, 1, 4] == labels[0, 0, 4]


def test_outside_only_component_is_discarded():
    prediction = np.zeros((1, 1, 9), dtype=int)
    prediction[..., :3] = 1
    prediction[..., 6:] = 1
    regions = np.zeros_like(prediction)
    regions[..., 0] = 1

    labels, count = label_region_aware(prediction, regions, [1])

    assert count == 1
    assert np.all(labels[..., :3] == 1)
    assert np.all(labels[..., 3:] == 0)


def test_disconnected_components_in_same_region_remain_separate():
    prediction = np.zeros((1, 1, 9), dtype=int)
    prediction[..., :2] = 1
    prediction[..., 5:] = 1
    regions = prediction * 7

    labels, count = label_region_aware(prediction, regions, [7])

    assert count == 2
    np.testing.assert_array_equal(labels[..., 5:], 1)
    np.testing.assert_array_equal(labels[..., :2], 2)


def test_diagonal_background_connection_is_preserved():
    prediction = np.eye(4, dtype=int)[None, ...]
    regions = np.zeros_like(prediction)
    regions[0, 0, 0] = 1

    labels, count = label_region_aware(prediction, regions, [1])

    assert count == 1
    np.testing.assert_array_equal(labels, prediction)


def test_empty_region_aware_prediction():
    prediction = np.zeros((2, 2, 2), dtype=int)

    labels, count = label_region_aware(prediction, prediction, [])

    assert count == 0
    np.testing.assert_array_equal(labels, prediction)


def test_label_clusters_preserves_region_information():
    prediction = np.ones((2, 2, 8), dtype=float)
    regions = np.ones(prediction.shape, dtype=int)
    regions[..., 4:] = 2

    labels = label_clusters(prediction, 0.5, 0, regions)

    assert set(np.unique(labels)) == {1, 2}
    assert np.all(labels[..., :4] == labels[0, 0, 0])
    assert np.all(labels[..., 4:] == labels[0, 0, -1])
    assert labels[0, 0, 0] != labels[0, 0, -1]


def test_outside_ratio_filters_after_background_assignment():
    prediction = np.ones((1, 1, 9), dtype=int)
    regions = np.zeros_like(prediction)
    regions[..., :4] = 1
    regions[..., -1] = 2

    filtered, _, count, labels, filtered_count = get_clusters_and_filter_image(
        prediction, brain=regions, outside_ratio=0.4
    )

    assert count == 2
    assert filtered_count == 1
    assert labels[0, 0, 0] == 1
    assert labels[0, 0, -1] == 0
    np.testing.assert_array_equal(filtered > 0, labels > 0)


def test_background_assignment_follows_cluster_not_straight_line_distance():
    prediction = np.zeros((1, 9, 7), dtype=int)
    prediction[0, :, 0] = 1
    prediction[0, :, 4] = 1
    prediction[0, -1, :5] = 1
    regions = np.zeros_like(prediction)
    regions[0, 0, 0] = 1
    regions[0, -1, 4] = 2
    regions[0, 0, 3] = 3

    labels, count = label_region_aware(prediction, regions, [1, 2, 3])

    assert count == 2
    np.testing.assert_array_equal(labels > 0, prediction > 0)
    assert labels[0, 0, 4] == labels[0, -1, 4]
    assert labels[0, 0, 4] != labels[0, 0, 0]


def test_filtered_image_excludes_unseeded_components():
    regions = np.zeros((1, 1, 9), dtype=int)
    regions[..., 0] = 1
    for include_seeded_component in (False, True):
        prediction = np.zeros_like(regions)
        prediction[..., 6:] = 1
        if include_seeded_component:
            prediction[..., :3] = 1

        filtered, _, _, labels, count = get_clusters_and_filter_image(
            prediction, brain=regions, outside_ratio=1.0
        )

        assert count == int(include_seeded_component)
        np.testing.assert_array_equal(filtered > 0, labels > 0)
        assert np.all(filtered[..., 6:] == 0)


def test_binary_brain_mask_and_no_mask_preserve_connected_prediction():
    prediction = np.ones((2, 2, 8), dtype=float)
    brain = np.ones(prediction.shape, dtype=bool)
    brain[..., -1] = False

    for regions in (None, brain):
        labels = label_clusters(prediction, 0.5, 0, regions)

        np.testing.assert_array_equal(labels, 1)


def test_size_filter_removes_small_region_piece():
    prediction = np.ones((1, 1, 6), dtype=float)
    regions = np.ones(prediction.shape, dtype=int)
    regions[..., 2:] = 2

    labels = label_clusters(prediction, 0.5, 2, regions)

    np.testing.assert_array_equal(labels[..., :2], 0)
    np.testing.assert_array_equal(labels[..., 2:], 1)
