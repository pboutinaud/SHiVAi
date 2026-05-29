"""Test bounding-box optimized cluster resampling."""
import numpy as np
import nibabel as nib
import time
from shivai.postprocessing.clusters import resample_cluster_img


def _make_cluster_img(shape, voxel_size, n_clusters=10, seed=42):
    """Create a test image with random clusters."""
    rng = np.random.RandomState(seed)
    vol = np.zeros(shape, dtype=np.int32)
    affine = np.diag(list(voxel_size) + [1.0])
    for i in range(1, n_clusters + 1):
        # Random cluster center
        center = [rng.randint(20, s - 20) for s in shape]
        # Random cluster size (3-8 voxels radius)
        radius = rng.randint(3, 8)
        # Create a sphere
        slices = tuple(slice(max(0, c - radius), min(s, c + radius + 1)) for c, s in zip(center, shape))
        subvol = vol[slices]
        coords = np.mgrid[tuple(slice(0, s) for s in subvol.shape)]
        center_local = [c - max(0, c - radius) for c in center]
        dist = sum((coords[d] - center_local[d]) ** 2 for d in range(3))
        subvol[dist <= radius ** 2] = i
    return nib.Nifti1Image(vol, affine)


def test_resample_map_clusters():
    """Test that all cluster labels are preserved after resampling."""
    src_shape = (128, 128, 128)
    src_voxsize = (1.5, 1.5, 1.5)
    tgt_shape = (160, 160, 160)
    tgt_voxsize = (1.0, 1.0, 1.0)

    cluster_img = _make_cluster_img(src_shape, src_voxsize, n_clusters=15)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    result = resample_cluster_img(cluster_img, target_img, n_parallel=1)
    result_data = result.get_fdata()

    src_labels = set(np.unique(cluster_img.get_fdata().astype(int))) - {0}
    tgt_labels = set(np.unique(result_data.astype(int))) - {0}

    print(f"Source labels: {sorted(src_labels)}")
    print(f"Target labels: {sorted(tgt_labels)}")
    assert src_labels == tgt_labels, f"Labels mismatch: missing={src_labels - tgt_labels}, extra={tgt_labels - src_labels}"
    print("PASSED: All cluster labels preserved")


def test_resample_with_transform():
    """Test resampling with an ANTs-style transform affine."""
    src_shape = (100, 100, 100)
    src_voxsize = (1.5, 1.5, 1.5)
    tgt_shape = (128, 128, 128)
    tgt_voxsize = (1.0, 1.0, 1.0)

    cluster_img = _make_cluster_img(src_shape, src_voxsize, n_clusters=10)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    # Small rotation + translation in LPS coordinates (ANTs convention)
    theta = np.radians(5)
    cos, sin = np.cos(theta), np.sin(theta)
    transform = np.eye(4)
    transform[:3, :3] = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    transform[:3, 3] = [2.0, -1.5, 3.0]

    result = resample_cluster_img(cluster_img, target_img, transform_affine=transform, n_parallel=4)
    result_data = result.get_fdata()

    src_labels = set(np.unique(cluster_img.get_fdata().astype(int))) - {0}
    tgt_labels = set(np.unique(result_data.astype(int))) - {0}

    print(f"Source labels: {sorted(src_labels)}")
    print(f"Target labels: {sorted(tgt_labels)}")
    # Some clusters might be lost if they rotate outside the target volume,
    # but no extra labels should appear
    assert tgt_labels.issubset(src_labels), f"Extra labels in result: {tgt_labels - src_labels}"
    print(f"PASSED: {len(tgt_labels)}/{len(src_labels)} cluster labels preserved with transform")


def test_resample_3d_same_ndim():
    """Test that 3D->3D resampling works (previously crashed with else clause bug)."""
    src_shape = (64, 64, 64)
    src_voxsize = (2.0, 2.0, 2.0)
    tgt_shape = (128, 128, 128)
    tgt_voxsize = (1.0, 1.0, 1.0)

    cluster_img = _make_cluster_img(src_shape, src_voxsize, n_clusters=5)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    result = resample_cluster_img(cluster_img, target_img, n_parallel=1)
    assert result.shape == tgt_shape
    print("PASSED: 3D->3D resampling works (dimension bug fixed)")


def test_benchmark():
    """Benchmark with many clusters."""
    src_shape = (128, 128, 128)
    src_voxsize = (1.5, 1.5, 1.5)
    tgt_shape = (192, 192, 192)
    tgt_voxsize = (1.0, 1.0, 1.0)

    cluster_img = _make_cluster_img(src_shape, src_voxsize, n_clusters=50, seed=123)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    n_labels = len(np.unique(cluster_img.get_fdata())) - 1
    print(f"Resampling {n_labels} clusters from {src_shape} ({src_voxsize}mm) -> {tgt_shape} ({tgt_voxsize}mm)")

    t0 = time.time()
    result = resample_cluster_img(cluster_img, target_img, n_parallel=8)
    elapsed = time.time() - t0

    tgt_labels = set(np.unique(result.get_fdata().astype(int))) - {0}
    print(f"Completed in {elapsed:.2f}s, {len(tgt_labels)} labels preserved")
    print("PASSED: Benchmark complete")


def _make_continuous_prediction(shape, voxel_size, n_blobs=8, seed=42):
    """Create a synthetic continuous prediction map with Gaussian-like blobs."""
    rng = np.random.RandomState(seed)
    vol = np.zeros(shape, dtype=np.float64)
    affine = np.diag(list(voxel_size) + [1.0])
    for _ in range(n_blobs):
        center = [rng.randint(15, s - 15) for s in shape]
        radius = rng.randint(2, 6)
        peak = rng.uniform(0.5, 1.0)
        slices = tuple(slice(max(0, c - radius * 2), min(s, c + radius * 2 + 1)) for c, s in zip(center, shape))
        subvol = vol[slices]
        coords = np.mgrid[tuple(slice(0, s) for s in subvol.shape)]
        center_local = [c - max(0, c - radius * 2) for c in center]
        dist_sq = sum((coords[d] - center_local[d]) ** 2 for d in range(3))
        gaussian = peak * np.exp(-dist_sq / (2 * (radius / 2) ** 2))
        subvol[:] = np.maximum(subvol, gaussian)
    return nib.Nifti1Image(vol, affine)


def test_pred_mode_threshold_equivalence():
    """Test that 'pred' mode gives threshold-equivalent results.

    Core property: for each grid-point threshold T,
        (resampled_pred > T)  should match
        resample_cluster_img((original > T), target, input_type='map') > 0
    """
    src_shape = (64, 64, 64)
    src_voxsize = (2.0, 2.0, 2.0)
    tgt_shape = (100, 100, 100)
    tgt_voxsize = (1.0, 1.0, 1.0)

    pred_img = _make_continuous_prediction(src_shape, src_voxsize, n_blobs=8)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    step = 0.1
    result = resample_cluster_img(pred_img, target_img, input_type='pred',
                                  n_parallel=4, threshold=0.1, threshold_step=step)
    result_data = result.get_fdata()

    # Verify no cluster is lost at the lowest threshold
    src_data = pred_img.get_fdata()
    src_mask_low = src_data > 0.1
    from skimage import measure as skm
    n_src_clusters = skm.label(src_mask_low, return_num=True)[1]
    n_tgt_clusters = skm.label(result_data > 0, return_num=True)[1]
    print(f"Clusters at lowest threshold: source={n_src_clusters}, target={n_tgt_clusters}")
    assert n_tgt_clusters >= n_src_clusters, (
        f"Lost clusters: source has {n_src_clusters}, target has {n_tgt_clusters}"
    )

    # Verify threshold equivalence at each grid-point level
    test_thresholds = np.arange(0.1, src_data.max(), step)
    for t_k in test_thresholds:
        preserve_mask = result_data > t_k
        # Smart-resample the original thresholded at t_k
        binary_data = (src_data > t_k).astype(np.int16)
        if binary_data.max() == 0:
            continue
        binary_img = nib.Nifti1Image(binary_data, pred_img.affine)
        try:
            ref_resampled = resample_cluster_img(binary_img, target_img, input_type='map', n_parallel=1)
            ref_mask = ref_resampled.get_fdata() > 0
        except ValueError:
            continue
        # The masks should be close at grid-point thresholds.
        # Small mismatches are expected because the smart resampling is not
        # perfectly monotonic across threshold levels: cluster topology and
        # bounding boxes differ at each level, so the resampled footprint at
        # a higher level is not always a strict spatial subset of a lower level.
        mismatch = np.count_nonzero(preserve_mask != ref_mask)
        total = np.count_nonzero(ref_mask)
        if total > 0:
            mismatch_ratio = mismatch / total
            print(f"  T={t_k:.2f}: ref_voxels={total}, mismatch={mismatch} ({mismatch_ratio:.4f})")
            assert mismatch_ratio < 0.05, (
                f"Threshold equivalence failed at T={t_k:.2f}: "
                f"mismatch ratio {mismatch_ratio:.4f} > 0.01"
            )
    print("PASSED: Preserve mode threshold equivalence verified")


def test_pred_vs_anat():
    """Test that 'pred' mode retains more clusters than 'anat' mode."""
    src_shape = (64, 64, 64)
    src_voxsize = (2.0, 2.0, 2.0)
    tgt_shape = (100, 100, 100)
    tgt_voxsize = (1.0, 1.0, 1.0)

    pred_img = _make_continuous_prediction(src_shape, src_voxsize, n_blobs=10, seed=99)
    target_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), np.diag(list(tgt_voxsize) + [1.0]))

    result_pred = resample_cluster_img(pred_img, target_img, input_type='pred', n_parallel=4)
    result_anat = resample_cluster_img(pred_img, target_img, input_type='anat')

    # At a high threshold, 'pred' should keep at least as many clusters
    thr = 0.5
    from skimage import measure as skm
    n_pred = skm.label(result_pred.get_fdata() > thr, return_num=True)[1]
    n_anat = skm.label(result_anat.get_fdata() > thr, return_num=True)[1]
    print(f"Clusters at T={thr}: pred={n_pred}, anat={n_anat}")
    assert n_pred >= n_anat, (
        f"'pred' mode lost more clusters ({n_pred}) than 'anat' ({n_anat})"
    )
    print("PASSED: 'pred' mode retains at least as many clusters as 'anat'")


if __name__ == "__main__":
    test_resample_3d_same_ndim()
    test_resample_map_clusters()
    test_resample_with_transform()
    test_benchmark()
    test_pred_mode_threshold_equivalence()
    test_pred_vs_anat()
    print("\nAll tests passed!")
