import numpy as np
import pandas as pd
import nibabel as nib
import pytest

from shivai.interfaces.image import Regionwise_Prediction_metrics
from shivai.utils.stats import prediction_metrics


@pytest.mark.parametrize('voxel_dim', [(1, 1, 1), (0.5, 2, 3)])
def test_raw_load_and_volumes(voxel_dim):
    clusters = np.array([1, 1, 1, 1, 2, 2, 0, 0]).reshape(8, 1, 1)
    segmentation = np.array([1, 1, 1, 2, 2, 0, 3, 0]).reshape(8, 1, 1)
    regions = {'Whole brain': -1, 'A': 1, 'B': 2, 'Empty': 3}

    census, stats = prediction_metrics(
        clusters, segmentation, regions, voxel_dim=voxel_dim)
    stats = stats.set_index('Region')
    scale = np.prod(voxel_dim)

    np.testing.assert_array_equal(census['Cluster size'], np.array([4, 2]) * scale)
    assert census['Biomarker region'].tolist() == ['A', 'B']
    assert stats['Raw Biomarker Load'].tolist() == [5, 3, 2, 0]
    assert stats['Number of clusters'].tolist() == [2, 1, 1, 0]
    assert stats['Total Biomarker volume'].tolist() == [6 * scale, 4 * scale, 2 * scale, 0]
    for column, expected in [
        ('Mean Biomarker volume', 3),
        ('Median Biomarker volume', 3),
        ('StD Biomarker volume', np.sqrt(2)),
        ('Min Biomarker volume', 2),
        ('Max Biomarker volume', 4),
    ]:
        assert stats.loc['Whole brain', column] == pytest.approx(expected * scale)
    assert pd.isna(stats.loc['Empty', 'Mean Biomarker volume'])
    assert regions == {'Whole brain': -1, 'A': 1, 'B': 2, 'Empty': 3}


def test_priority_attribution_does_not_change_raw_load():
    clusters = np.ones((4, 1, 1), dtype=int)
    segmentation = np.array([1, 1, 1, 2]).reshape(4, 1, 1)

    census, stats = prediction_metrics(
        clusters, segmentation, {'A': 1, 'B': 2}, prio_labels=['B'])

    assert census['Biomarker region'].tolist() == ['B']
    assert stats['Number of clusters'].tolist() == [0, 1]
    assert stats['Raw Biomarker Load'].tolist() == [3, 1]


def test_empty_prediction_and_singleton_dimension():
    clusters = np.zeros((2, 2, 2, 1), dtype=int)
    segmentation = np.ones_like(clusters)

    census, stats = prediction_metrics(
        clusters, segmentation, {'Whole brain': -1, 'A': 1}, voxel_dim=(2, 2, 2))

    assert census.empty
    assert stats['Raw Biomarker Load'].tolist() == [0, 0]
    assert stats['Total Biomarker volume'].tolist() == [0, 0]
    assert stats['Mean Biomarker volume'].isna().all()


def test_default_whole_brain_is_repeatable():
    for _ in range(2):
        census, stats = prediction_metrics(
            np.ones((2, 2, 2), dtype=int), np.ones((2, 2, 2), dtype=int))
        assert census['Cluster size'].tolist() == [8]
        assert stats['Region'].tolist() == ['Whole brain']
        assert stats['Raw Biomarker Load'].tolist() == [8]


def test_interface_uses_image_voxel_dimensions(tmp_path):
    clusters = np.ones((2, 2, 2), dtype=np.int16)
    affine = np.diag([0.5, 2, 3, 1])
    cluster_file = tmp_path / 'clusters.nii.gz'
    segmentation_file = tmp_path / 'segmentation.nii.gz'
    nib.save(nib.Nifti1Image(clusters, affine), cluster_file)
    nib.save(nib.Nifti1Image(clusters, affine), segmentation_file)

    result = Regionwise_Prediction_metrics(
        labelled_clusters=str(cluster_file),
        brain_seg=str(segmentation_file),
        brain_seg_type='brain_mask',
        region_list=['Whole brain'],
        biomarker_type='wmh',
    ).run(cwd=str(tmp_path))

    stats = pd.read_csv(result.outputs.biomarker_stats_csv)
    census = pd.read_csv(result.outputs.biomarker_census_csv)
    wide = pd.read_csv(result.outputs.biomarker_stats_wide_csv)
    assert stats['Raw Biomarker Load'].tolist() == [8]
    assert stats['Total Biomarker volume'].tolist() == [24]
    assert census['Cluster size'].tolist() == [24]
    assert wide['Whole brain - Raw Biomarker Load'].tolist() == [8]
    assert wide['Whole brain - Total Biomarker volume'].tolist() == [24]
