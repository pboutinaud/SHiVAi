import numpy as np
import nibabel as nib
from skimage import measure
from concurrent.futures import ThreadPoolExecutor
from shivai.utils.misc import fisin
import nibabel.processing as nip


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


def cluster_registration(input_im: nib.Nifti1Image, ref_im: nib.Nifti1Image, transform_affine: np.ndarray) -> nib.Nifti1Image:
    """Apply a linear registration to labelled clusters in a way that conserve all clusters 
    /!\ Never worked /!\\
        -> Use  resample_cluster_img instead

    Args:
        input_im (nib.Nifti1Image): Image containing labelled clusters (with integers as labels)
        ref_im (nib.Nifti1Image): Image defining the arrival space
        transform_affine (np.ndarray): Affine matrix (4x4) defining the linear transformation

    Returns:
        nib.Nifti1Image: _description_
    """
    input_vol = input_im.get_fdata().astype('int16')
    input_affine = input_im.affine
    ref_affine = ref_im.affine
    pls2ras = np.diag([-1, -1, 1, 1])

    # Combining the different affines
    ref_affine_inv = np.linalg.inv(ref_affine)
    transform_affine_inv = np.linalg.inv(pls2ras @ transform_affine @ pls2ras)  # ANTs affines must be inversed
    full_affine = ref_affine_inv @ transform_affine_inv @ input_affine  # TODO: make this work T.T
    # Getting the new coordinates for each voxel
    ori_coord = np.argwhere(input_vol)
    new_coord = nib.affines.apply_affine(full_affine, ori_coord)
    new_coord = np.round(new_coord).astype(int).T  # rounding and reshaping the coordinate array for indexing
    # Correcting points that got out of the image
    new_coord[(new_coord < 0)] = 0
    new_coord[0, (new_coord[0] >= ref_im.shape[0])] = ref_im.shape[0] - 1
    new_coord[1, (new_coord[1] >= ref_im.shape[1])] = ref_im.shape[1] - 1
    new_coord[2, (new_coord[2] >= ref_im.shape[2])] = ref_im.shape[2] - 1

    clust_reg_vol = np.zeros(ref_im.shape, dtype='int16')
    clust_reg_vol[tuple(new_coord)] = 1

    clust_reg_im = nib.Nifti1Image(clust_reg_vol, affine=ref_affine)
    return clust_reg_im


def _resample_one_cluster(val, cluster_data, ori_vox_vol, source_affine, target_affine, target_shape, new_vox_vol, thresh_fractions, interp_order=3):
    """Resample a single cluster label using bounding-box cropping for speed.

    Instead of resampling the full volume, crops both source and target to the
    cluster's bounding box region, dramatically reducing computation since
    scipy.ndimage.affine_transform scales with output volume size.
    """
    _NDIM = 3
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
    resampled_sub = nip.resample_from_to(cropped_src_img, (sub_tgt_shape, sub_tgt_affine))
    raw_data = resampled_sub.get_fdata()
    raw_data[raw_data < 0] = 0  # Remove negative interpolation artifacts

    if raw_data.max() == 0:
        return val, None, None, None, tgt_min

    # --- Find best threshold to match original volume ---
    working_data = raw_data.copy()
    thresholds = [frac * working_data.max() for frac in thresh_fractions]
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
    return val, ok_thr, ok_mask_vol, raw_data, tgt_min


def resample_cluster_img(cluster_img: nib.Nifti1Image, target_img: nib.Nifti1Image, continuous: bool = False, transform_affine: np.ndarray = None, n_parallel: int = 8) -> nib.Nifti1Image:
    """Resample all the cluster masks from an image to the space of a target image.

    If "continuous" is False, will try to preserve all clusters by resampling
    each cluster separately with a smart thresholding process to find the best
    threshold that preserves the original cluster size in mm^3 as much as possible
    without losing it. If "continuous" is True, will do a simple continuous
    resampling of the whole image, which may lead to some clusters being lost
    if they become too small after resampling.

    Args:
        cluster_img (nib.Nifti1Image): Nifti image containing the clusters to be resampled
        target_img (nib.Nifti1Image): Nifti image defining the target space for resampling
        continuous (bool): Whether to use continuous interpolation (resample_from_to),
            typically because the clusters have continuous values (default: False). If False,
            cluster_img must contain integer labels. Will then use the "smart" resampling by
            individually resampling each cluster (continuously) then trying different thresholds
            to get the best match with the original cluster size (in mm^3) for each cluster. Will
            be careful not to delete clusters during this process.
            One special case is when the original and target voxel volumes are the same, in which 
            case a simple nearest neighbor resampling is done without the smart thresholding process, 
            to avoid interpolation issues.
        transform_affine (np.ndarray, optional): 4x4 ANTs-style affine matrix (in LPS coordinates)
            encoding the linear transformation between cluster_img and target_img spaces.
            If None (default), the two images are assumed to be already aligned (their
            NIfTI affines alone define the spatial correspondence). If provided, the transform
            is composed into the source affine before resampling.
        n_parallel (int): Number of parallel threads for resampling individual clusters
            (default: 8). Set to 1 to disable parallelization.

    Returns:
        nib.Nifti1Image: Resampled cluster mask in the space of the target image
    """
    # Compute original voxel volume before any affine modification
    ori_vox_vol = cluster_img.header.get_zooms()[0] * cluster_img.header.get_zooms()[1] * cluster_img.header.get_zooms()[2]

    # Apply ANTs linear transform if given (compose into source affine)
    if transform_affine is not None:
        lps2ras = np.diag([-1, -1, 1, 1])
        T_ras = lps2ras @ transform_affine @ lps2ras
        composed_affine = np.linalg.inv(T_ras) @ cluster_img.affine
        cluster_img = nib.Nifti1Image(np.asarray(cluster_img.dataobj), composed_affine)

    if continuous:
        return nip.resample_from_to(cluster_img, target_img)
    new_vox_vol = target_img.header.get_zooms()[0] * target_img.header.get_zooms()[1] * target_img.header.get_zooms()[2]
    if ori_vox_vol == new_vox_vol:
        # if voxel volumes are the same, no need for the smart resampling, just do a nearest neighbor resampling to avoid interpolation issues
        return nip.resample_from_to(cluster_img, target_img, order=0)
    resampled_vol = np.zeros(target_img.shape, dtype=cluster_img.get_fdata().dtype)
    cluster_data = cluster_img.get_fdata()

    # Check and match the number of dim between resampled_vol (i.e. target_img) and cluster_data
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
    if len(cluster_vals) == 1:  # if only one value, probably not clustered yet, so need call to label() fist
        cluster_data = measure.label(cluster_data > 0)
        cluster_vals = list(np.unique(cluster_data))
    cluster_vals.remove(0)  # remove background
    # Order label values by cluster size (largest first) to try to preserve smaller clusters
    vals_to_process = sorted(cluster_vals, key=lambda v: np.sum(cluster_data == v), reverse=True)

    # Thresholds to try for each cluster to find the best match with original size
    # These are fractions of the max value in the resampled cluster mask
    thresh_frac = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_workers = min(n_parallel, len(vals_to_process)) if n_parallel > 1 else 1
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _resample_one_cluster, val, cluster_data, ori_vox_vol,
                    cluster_img.affine, target_img.affine, target_img.shape, new_vox_vol, thresh_frac
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
                    raise ValueError(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                     "Consider using continuous resampling or adjusting the thresholds.")
    else:
        for val in vals_to_process:
            _, ok_thr, ok_mask_vol, sub_data, tgt_origin = _resample_one_cluster(
                val, cluster_data, ori_vox_vol, cluster_img.affine, target_img.affine, target_img.shape, new_vox_vol, thresh_frac
            )
            if ok_thr is not None and ok_mask_vol > 0:
                slices = tuple(slice(tgt_origin[i], tgt_origin[i] + sub_data.shape[i]) for i in range(3))
                if len(resampled_vol.shape) > 3:
                    slices += (slice(None),) * (len(resampled_vol.shape) - 3)
                resampled_vol[slices][sub_data >= ok_thr] = val
            else:
                raise ValueError(f"Could not find a suitable threshold to resample cluster with label {val} without losing it. "
                                 "Consider using continuous resampling or adjusting the thresholds.")
    return nib.Nifti1Image(resampled_vol, target_img.affine)
