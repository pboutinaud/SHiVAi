import logging
import numpy as np
import nibabel as nib
from skimage import measure
from concurrent.futures import ThreadPoolExecutor
from shivai.utils.misc import fisin
import nibabel.processing as nip
from scipy import ndimage

logger = logging.getLogger(__name__)


def get_clusters_and_filter_image(image, cluster_filter=0, brain=None, outside_ratio=0.25):
    """
    Compute clusters and filter out those of size "cluster_filter" and smaller.
    Also removes clusters that are mostly outside of the brain segmentation mask
    if a brain segmentation mask is provided (>25% of cluster voxels outside brain).

    """

    clusters, num_clusters = measure.label(image, return_num=True)
    if num_clusters == 0:
        return image, clusters, num_clusters, clusters, num_clusters

    apply_filter = bool(cluster_filter) or brain is not None
    if apply_filter:
        clusnum, counts = np.unique(clusters[clusters > 0], return_counts=True)
        to_remove = set(clusnum[counts <= cluster_filter]) if cluster_filter else set()

        if brain is not None:
            brain_mask = np.asarray(brain).astype(bool)
            if brain_mask.shape != image.shape:
                raise ValueError(
                    f'Brain mask shape ({brain_mask.shape}) does not match image shape ({image.shape}).'
                )
            for clus_i in clusnum:
                clus_mask = (clusters == clus_i)
                clus_size = np.count_nonzero(clus_mask)
                vox_out = np.count_nonzero(clus_mask & ~brain_mask)
                if clus_size > 0 and vox_out > (outside_ratio * clus_size):
                    to_remove.add(clus_i)

        nums_left = [i for i in clusnum if i not in to_remove]

        image_f = image.copy()
        clusters_f = clusters.copy()
        if to_remove:
            to_remove_arr = np.array(sorted(to_remove), dtype=clusters.dtype)
            remove_mask = fisin(clusters, to_remove_arr)
            image_f[remove_mask] = 0
            clusters_f[remove_mask] = 0
        num_clusters_f = len(nums_left)

        for new_i, old_i in enumerate(nums_left, start=1):
            clusters_f[clusters == old_i] = new_i
    else:  # filtered clusters are the same
        image_f, clusters_f, num_clusters_f = image, clusters, num_clusters
    return image_f, clusters, num_clusters, clusters_f, num_clusters_f


def label_clusters(pred_vol, threshold, cluster_filter, brain_seg_vol=None, outside_ratio=0.25):
    """Threshold and labelize the clusters from a prediction map.
    Also removes clusters that are smaller than or equal to the "cluster_filter" size (in voxels) 
    and those that are mostly outside of the brain segmentation mask.

    Args:
        pred_vol (np.ndarray): Prediction map from the AI model
        brain_seg_vol (np.ndarray): Brain seg delimiting the brain
        threshold (float): Value to threshold the prediction map
        cluster_filter (int): size up to which (including) small clusters are removed
        outside_ratio (float): ratio of voxels outside the brain seg mask above which a cluster is removed (default: 0.25)

    Returns:
        labelled_clusters (np.ndarray): Labelled clusters volume
    """
    if len(pred_vol.shape) > 3:
        pred_vol = pred_vol.squeeze()
    if brain_seg_vol is not None and len(brain_seg_vol.shape) > 3:
        brain_seg_vol = brain_seg_vol.squeeze()
    brain_mask = (brain_seg_vol > 0) if brain_seg_vol is not None else None
    thresholded_img = (pred_vol > threshold).astype(int)
    _, _, _, labelled_clusters, _ = get_clusters_and_filter_image(
        thresholded_img,
        cluster_filter=cluster_filter,
        brain=brain_mask,
        outside_ratio=outside_ratio
    )
    return labelled_clusters


def anisotropic_prefilter(vol, voxel_size_ori, voxel_size_target, safety_factor=0.5, verbose=False):
    """
    Anisotropic Gaussian pre-filter before resampling back to another space with strong isotropic differences.
    Smooths only along axes where the target resolution is coarser than the
    original resolution, with sigma scaled to the anisotropy ratio.

    Parameters
    ----------
    vol : np.ndarray
        Image in original space.
    voxel_size_ori : array-like
        Voxel size in original space, one value per axis (mm or any unit).
    voxel_size_target : array-like
        Voxel size in target space, one value per axis.
    safety_factor : float
        Scales the sigma. 0.5 means the cluster needs to be ~1 native (=target) voxel
        wide to survive.

    Returns
    -------
    np.ndarray
        Pre-filtered image, same shape as input.
    """
    voxel_size_target = np.asarray(voxel_size_target, dtype=float)
    voxel_size_ori = np.asarray(voxel_size_ori, dtype=float)

    assert len(voxel_size_target) == vol.ndim, "voxel_size_target must have one value per image axis"
    assert len(voxel_size_ori) == vol.ndim, "voxel_size_ori must have one value per image axis"

    # Ratio > 1 means target is coarser than original along that axis
    ratio = voxel_size_target / voxel_size_ori

    # Sigma in resampled voxel units: how many resampled voxels fit in one
    # native voxel, scaled by safety_factor. Axes where ratio <= 1 get
    # sigma=0 (no smoothing needed).
    sigmas = np.where(ratio > 1.0, ratio * safety_factor, 0.0)

    if verbose:
        print(f"Anisotropy ratios: {ratio}")
        print(f"Sigmas (resampled vox units): {sigmas}")

    return ndimage.gaussian_filter(vol.astype(float), sigma=sigmas)


def _resample_one_cluster(val, cluster_data, ori_vox_zooms, source_affine, target_affine, target_shape, new_vox_zooms, thresh_fractions, interp_order=3, rel_tolerance=1.0):
    """Resample a single cluster label using bounding-box cropping for speed.

    Instead of resampling the full volume, crops both source and target to the
    cluster's bounding box region, dramatically reducing computation since
    scipy.ndimage.affine_transform scales with output volume size.
    """
    _NDIM = 3

    new_vox_vol = np.prod(new_vox_zooms)
    ori_vox_vol = np.prod(ori_vox_zooms)
    mask = cluster_data == val
    mask_vol = np.sum(mask) * ori_vox_vol

    # --- Crop source to cluster bounding box + padding ---
    coords = np.argwhere(mask)
    src_min = coords.min(axis=0)
    src_max = coords.max(axis=0)
    pad = interp_order + 1
    src_min_pad = np.maximum(src_min - pad, 0)
    src_max_pad = np.minimum(src_max + pad, np.array(cluster_data.shape) - 1)
    slices_src = tuple(slice(lo, hi + 1) for lo, hi in zip(src_min_pad, src_max_pad))
    cropped_mask = mask[slices_src].astype(float)
    if interp_order > 1:
        cropped_mask = anisotropic_prefilter(cropped_mask, ori_vox_zooms, new_vox_zooms)

    # Affine for cropped source: shift origin by src_min_pad voxels
    offset_src = np.eye(4)
    offset_src[:_NDIM, 3] = src_min_pad[:_NDIM]
    cropped_src_affine = source_affine @ offset_src

    # --- Compute target bounding box by mapping source bbox corners ---
    src_to_tgt = np.linalg.inv(target_affine) @ source_affine
    bbox_min = src_min[:_NDIM].astype(float)
    bbox_max = src_max[:_NDIM].astype(float)
    # Generate the 8 corners of the 3D source bounding box
    idx = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
    corners_src = bbox_min + idx * (bbox_max - bbox_min)
    corners_h = np.hstack([corners_src, np.ones((8, 1))])
    corners_tgt = (src_to_tgt @ corners_h.T).T[:, :_NDIM]

    tgt_min = np.floor(corners_tgt.min(axis=0)).astype(int) - pad
    tgt_max = np.ceil(corners_tgt.max(axis=0)).astype(int) + pad
    tgt_min = np.maximum(tgt_min, 0)
    tgt_max = np.minimum(tgt_max, np.array(target_shape[:_NDIM]) - 1)

    # Project source centroid into sub-crop target voxel coordinates.
    # Used later to pick the component closest to the true cluster location,
    # which is more robust than picking the largest component when ringing
    # artifacts can be bigger than the true signal (e.g. very coarse target).
    src_centroid = coords.mean(axis=0)  # shape (3,)
    src_centroid_h = np.append(src_centroid[:_NDIM], 1.0)
    tgt_centroid = (src_to_tgt @ src_centroid_h)[:_NDIM] - tgt_min

    if np.any(tgt_min > tgt_max):
        # Cluster projects entirely outside the target image
        return val, None, None, None, tgt_min

    sub_tgt_shape = tuple(tgt_max - tgt_min + 1)
    if len(target_shape) > _NDIM:
        sub_tgt_shape = sub_tgt_shape + target_shape[_NDIM:]

    # Affine for cropped target: shift origin by tgt_min voxels
    offset_tgt = np.eye(4)
    offset_tgt[:_NDIM, 3] = tgt_min.astype(float)
    sub_tgt_affine = target_affine @ offset_tgt

    # --- Resample cropped source to cropped target ---
    cropped_src_img = nib.Nifti1Image(cropped_mask, cropped_src_affine)
    resampled_sub = nip.resample_from_to(cropped_src_img, (sub_tgt_shape, sub_tgt_affine), interp_order)
    raw_data = resampled_sub.get_fdata()
    raw_data[raw_data < 0] = 0  # Remove negative interpolation artifacts

    # Removing ringing artifacts from spline interpolation.
    # Keep the component whose centroid is closest to the projected source
    # centroid, which is more robust than "largest" when resampling to very
    # coarse voxels (ringing blobs can exceed the true cluster in size).
    if interp_order > 1:
        binerized = raw_data > 1e-5
        labeled_sub, n_sub = measure.label(binerized, connectivity=1, return_num=True)
        if n_sub > 1:
            best_label = 1
            best_dist = np.inf
            for lbl in range(1, n_sub + 1):
                comp_centroid = np.argwhere(labeled_sub == lbl).mean(axis=0)
                dist = np.linalg.norm(comp_centroid - tgt_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_label = lbl
            raw_data[labeled_sub != best_label] = 0
    if raw_data.max() == 0:
        return val, None, None, None, tgt_min

    # --- Find best threshold to match original volume ---
    working_data = raw_data.copy()
    thresholds = [frac * raw_data.max() for frac in thresh_fractions]
    prev_vol, prev_thr = None, None
    ok_thr, ok_mask_vol = None, None
    for thr in thresholds:
        working_data[working_data < thr] = 0
        new_mask_vol = np.sum(working_data > 0) * new_vox_vol
        if prev_vol is not None:
            if new_mask_vol < mask_vol:
                if new_mask_vol == 0:
                    ok_thr = prev_thr
                    ok_mask_vol = prev_vol
                    break
                ok_thr, ok_mask_vol = [(thr, new_mask_vol), (prev_thr, prev_vol)][np.argmin([abs(new_mask_vol - mask_vol), abs(prev_vol - mask_vol)])]
                break
        prev_vol = new_mask_vol
        prev_thr = thr
    if ok_thr is None:
        ok_thr = raw_data.max()
        ok_mask_vol = np.sum(raw_data == ok_thr) * new_vox_vol
        if np.abs(mask_vol-ok_mask_vol)/mask_vol > rel_tolerance:
            # If even the best threshold is too far from the original volume, consider that we lost the cluster
            ok_thr = None
            ok_mask_vol = None
    return val, ok_thr, ok_mask_vol, raw_data, tgt_min


def _fsl_scaled_voxel_mat(img: nib.Nifti1Image) -> np.ndarray:
    """Build the voxel-to-FSL-scaled-mm matrix for an image (FLIRT convention).

    FLIRT affines do not operate in NIfTI world (mm) coordinates but in FSL's
    "scaled voxel" coordinates: voxel indices multiplied by the voxel sizes,
    with the first (x) axis flipped when the image's affine has a positive
    determinant (so that FSL always works in a left-handed/radiological frame).

    Args:
        img (nib.Nifti1Image): Image whose scaled-voxel matrix is needed.

    Returns:
        np.ndarray: 4x4 matrix mapping voxel coordinates to FSL scaled-mm coordinates.
    """
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    scale = np.diag(np.append(zooms, 1.0))
    if np.linalg.det(img.affine) > 0:
        nx = img.shape[0]
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (nx - 1) * zooms[0]
        return flip @ scale
    return scale


def resample_cluster_img(cluster_img: nib.Nifti1Image, target_img: nib.Nifti1Image, input_type: str = 'map', transform_affine: np.ndarray = None, affine_type: str = 'ants', n_parallel: int = 8, threshold: float = 0.05, threshold_step: float = 0.05, accept_loss: bool = True) -> nib.Nifti1Image:
    """Resample all the cluster masks from an image to the space of a target image.

    The resampling strategy is determined by the ``input_type`` argument:

    - ``'map'`` (default) — **Labelled cluster maps**. Input must contain integer
      labels (binary masks or one integer per cluster). Each cluster is resampled
      individually with a smart thresholding process that preserves the original
      cluster size in mm³ as much as possible without losing any cluster. When the
      source and target voxel volumes are identical, a simple nearest-neighbour
      resampling is used instead.
    - ``'pred'`` — **Continuous prediction maps** (e.g. posterior probabilities in
      [0, 1]). Applies the smart binary resampling (``input_type='map'``) at
      multiple threshold levels, then reconstructs a continuous map. This
      guarantees that thresholding the resampled map at any grid-point threshold
      *T* gives a result very close to what you would get by thresholding the
      original map at *T* and then applying the smart resampling. The threshold
      grid is controlled by the ``threshold`` and ``threshold_step`` parameters.
    - ``'anat'`` — **Anatomical / continuous images**. Performs a straightforward
      continuous resampling (``nibabel.processing.resample_from_to``, order 3
      spline). Suitable for anatomical images or any continuous data where
      cluster preservation is not needed.

    Args:
        cluster_img (nib.Nifti1Image): Nifti image to be resampled.
        target_img (nib.Nifti1Image): Nifti image defining the target space for resampling.
        input_type (str): Type of input data — one of ``'map'``, ``'pred'``, or
            ``'anat'``. See above for details.
        transform_affine (np.ndarray, optional): 4x4 affine matrix encoding the linear
            transformation between cluster_img and target_img spaces. Its convention is
            given by ``affine_type``. If None (default), the two images are assumed to be
            already aligned (their NIfTI affines alone define the spatial correspondence).
            If provided, the transform is composed into the source affine before resampling.
        affine_type (str): Convention of ``transform_affine`` — one of ``'ants'`` or
            ``'fsl'`` (default: ``'ants'``).
            ``'ants'``: ANTs/ITK-style affine in LPS world coordinates mapping the target
            (fixed) space to the cluster (moving) space.
            ``'fsl'``: FSL FLIRT-style affine in scaled-voxel coordinates mapping the
            cluster (input/moving) space to the target (reference) space. The image
            geometries (voxel sizes and shapes) of ``cluster_img`` and ``target_img`` are
            used to convert it to world coordinates, so they must match the images that
            were given to FLIRT as ``-in`` (cluster_img) and ``-ref`` (target_img).
            Ignored when ``transform_affine`` is None.
        n_parallel (int): Number of parallel threads for resampling individual clusters
            (default: 8). Set to 1 to disable parallelization. Only used for
            ``input_type='map'`` and ``input_type='pred'``.
        threshold (float): Minimum threshold level for the ``'pred'`` mode (default: 0.05).
            Ignored for other input types.
        threshold_step (float): Step between threshold levels for the ``'pred'`` mode
            (default: 0.05). With threshold=0.05 and threshold_step=0.05, levels will be
            0.05, 0.10, ..., up to the maximum value in the data. Ignored for other
            input types.
        accept_loss (bool): Whether to accept some loss of cluster volume during resampling
            (default: True). If False, raises an error when significant volume loss is detected.

    Returns:
        nib.Nifti1Image: Resampled image in the space of the target image.

    Raises:
        ValueError: If ``input_type`` is not one of ``'map'``, ``'pred'``, ``'anat'``,
            or if ``affine_type`` is not one of ``'ants'``, ``'fsl'``.
    """
    _VALID_INPUT_TYPES = ('map', 'pred', 'anat')
    _VALID_AFFINE_TYPES = ('ants', 'fsl')
    _NDIM = 3
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {_VALID_INPUT_TYPES}, got {input_type!r}."
        )
    if affine_type not in _VALID_AFFINE_TYPES:
        raise ValueError(
            f"affine_type must be one of {_VALID_AFFINE_TYPES}, got {affine_type!r}."
        )

    # Harmonizing dimensions by squeezing singleton dimensions
    cluster_img = nib.funcs.squeeze_image(cluster_img)
    target_img = nib.funcs.squeeze_image(target_img)
    # Compute original voxel volume before any affine modification
    ori_vox_zooms = cluster_img.header.get_zooms()

    # Apply the linear transform if given (compose into source affine)
    if transform_affine is not None:
        if affine_type == 'ants':
            # ANTs/ITK affine is in LPS world coords and maps target (fixed) -> cluster
            # (moving). Convert it to RAS, then push the cluster voxels into target world.
            lps2ras = ras2lps = np.diag([-1, -1, 1, 1])
            T_ras = lps2ras @ transform_affine @ ras2lps
            composed_affine = np.linalg.inv(T_ras) @ cluster_img.affine
        else:  # 'fsl'
            # FLIRT affine is in scaled-voxel coords and maps cluster (input) -> target
            # (reference). Compose: cluster voxel -> cluster FSL -> (FLIRT) -> target FSL
            # -> target voxel -> target world.
            src_fsl = _fsl_scaled_voxel_mat(cluster_img)
            ref_fsl = _fsl_scaled_voxel_mat(target_img)
            composed_affine = target_img.affine @ np.linalg.inv(ref_fsl) @ transform_affine @ src_fsl
        cluster_img = nib.Nifti1Image(np.asarray(cluster_img.dataobj), composed_affine)

    new_vox_zooms = target_img.header.get_zooms()

    # Reorder target voxel sizes to match source data axis orientation.
    src_ornt = nib.io_orientation(cluster_img.affine)
    tgt_ornt = nib.io_orientation(target_img.affine)
    aligned_tgt_zooms = np.zeros(_NDIM)
    for src_ax in range(_NDIM):
        spatial_ax = int(src_ornt[src_ax, 0])
        tgt_ax = int(np.where(tgt_ornt[:, 0] == spatial_ax)[0][0])
        aligned_tgt_zooms[src_ax] = new_vox_zooms[tgt_ax]
    aligned_tgt_zooms = tuple(aligned_tgt_zooms)

    if ori_vox_zooms == aligned_tgt_zooms:
        # if voxel sizes are the same, no need for the smart resampling, just do a nearest neighbor resampling to avoid interpolation issues
        return nip.resample_from_to(cluster_img, target_img, order=0)

    if input_type == 'pred':
        # Multi-threshold level-set smart resampling for continuous prediction maps.
        # At each threshold level T_k, threshold the data into a binary mask and apply
        # the smart cluster-preserving resampling (input_type='map'). Then reconstruct
        # a continuous map by assigning voxels the midpoint value of the highest
        # threshold level at which they survive. This ensures that
        # (resampled > T_k) ≈ smart_resample((original > T_k)) for every grid-point T_k.
        source_data = np.asarray(cluster_img.dataobj).astype(float)
        max_val = source_data.max()
        if max_val <= threshold:
            # Nothing above the minimum threshold — return zeros
            return nib.Nifti1Image(np.zeros(target_img.shape, dtype=np.float64), target_img.affine)
        levels = np.arange(threshold, max_val + threshold_step / 2, threshold_step)
        resampled_vol = np.zeros(target_img.shape, dtype=np.float64)
        for t_k in levels:
            # Threshold original data and create a binary NIfTI image
            binary_data = (source_data > t_k).astype(np.int16)
            if binary_data.max() == 0:
                break  # No more voxels above this level — done
            binary_img = nib.Nifti1Image(binary_data, cluster_img.affine)
            try:
                resampled_binary = resample_cluster_img(
                    binary_img, target_img, input_type='map', n_parallel=n_parallel, accept_loss=accept_loss
                )
                mask = resampled_binary.get_fdata() > 0
            except ValueError:
                # Smart resampling could not preserve a cluster at this level.
                # Voxels keep their value from previous (lower) levels.
                logger.warning(
                    f"Could not smart-resample at threshold {t_k:.3f}, skipping level."
                )
                continue
            # Assign midpoint value; use np.maximum to keep the highest value
            # when clusters overlap in target space
            assign_val = t_k + threshold_step / 2
            np.maximum(resampled_vol, mask * assign_val, out=resampled_vol)
        return nib.Nifti1Image(resampled_vol, target_img.affine)

    if input_type == 'anat':
        return nip.resample_from_to(cluster_img, target_img)

    resampled_vol = np.zeros(target_img.shape, dtype=cluster_img.get_fdata().dtype)
    cluster_data = cluster_img.get_fdata()

    # Check and match the number of dim between resampled_vol (i.e. target_img) and cluster_data
    # (may not be necessary since we squeeze the images at the beginning, but just in case)
    if len(cluster_data.shape) < len(target_img.shape) and target_img.shape[-1] == 1:
        cluster_data = np.expand_dims(cluster_data, axis=-1)
    elif len(cluster_data.shape) > len(target_img.shape) and cluster_data.shape[-1] == 1:
        cluster_data = np.squeeze(cluster_data, axis=-1)
    elif len(cluster_data.shape) != len(target_img.shape):
        raise ValueError(f"Cluster image shape {cluster_data.shape} and target image shape {target_img.shape} are not compatible for resampling.")

    # Check if the data is integer
    if not np.all(np.isclose(cluster_data, cluster_data.astype(int), atol=1e-5)):
        raise ValueError("Cluster image must contain integer labels for smart resampling.")
    cluster_data = cluster_data.astype(int)
    cluster_vals = list(np.unique(cluster_data))
    cluster_vals.remove(0)
    ori_val = None
    if len(cluster_vals) == 1:  # if only one value, probably not clustered yet, so need call to label() fist
        ori_val = cluster_vals[0]
        cluster_data = measure.label(cluster_data > 0)
        cluster_vals = list(np.unique(cluster_data))
        cluster_vals.remove(0)  # remove background
    # Order label values by cluster size (largest first) to try to preserve smaller clusters
    vals_to_process = sorted(cluster_vals, key=lambda v: np.sum(cluster_data == v), reverse=True)

    # Thresholds to try for each cluster to find the best match with original size
    # These are fractions of the max value in the resampled cluster mask
    thresh_frac = np.arange(0.2, 1.0, 0.03)
    n_workers = min(n_parallel, len(vals_to_process)) if n_parallel > 1 else 1
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _resample_one_cluster, val, cluster_data, ori_vox_zooms,
                    cluster_img.affine, target_img.affine, target_img.shape, aligned_tgt_zooms, thresh_frac
                ): val for val in vals_to_process
            }
            for future in futures:
                val, ok_thr, ok_mask_vol, sub_data, tgt_origin = future.result()
                if ok_thr is not None and ok_mask_vol > 0:
                    slices = tuple(slice(tgt_origin[i], tgt_origin[i] + sub_data.shape[i]) for i in range(3))
                    if len(resampled_vol.shape) > 3:
                        slices += (slice(None),) * (len(resampled_vol.shape) - 3)
                    resampled_vol[slices][sub_data >= ok_thr] = val
                else:
                    if accept_loss:
                        logger.warning(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                       "Cluster will be lost in the resampled image. Consider using continuous resampling or adjusting the thresholds.")
                    else:
                        raise ValueError(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                         "Consider using continuous resampling or adjusting the thresholds.")
    else:
        for val in vals_to_process:
            _, ok_thr, ok_mask_vol, sub_data, tgt_origin = _resample_one_cluster(
                val, cluster_data, ori_vox_zooms, cluster_img.affine, target_img.affine, target_img.shape, aligned_tgt_zooms, thresh_frac
            )
            if ok_thr is not None and ok_mask_vol > 0:
                slices = tuple(slice(tgt_origin[i], tgt_origin[i] + sub_data.shape[i]) for i in range(3))
                if len(resampled_vol.shape) > 3:
                    slices += (slice(None),) * (len(resampled_vol.shape) - 3)
                resampled_vol[slices][sub_data >= ok_thr] = val
            else:
                if accept_loss:
                    logger.warning(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                   "Cluster will be lost in the resampled image. Consider using continuous resampling or adjusting the thresholds.")
                else:
                    raise ValueError(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                     "Consider using continuous resampling or adjusting the thresholds.")
        if ori_val is not None:
            resampled_vol[resampled_vol > 0] = ori_val
    return nib.Nifti1Image(resampled_vol, target_img.affine)
