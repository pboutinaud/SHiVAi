import gc
import numpy as np
from skimage import measure
#from joblib import Parallel, delayed
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (
    distance_transform_edt, binary_erosion,
    generate_binary_structure
)


# --------------------------------------------------------------------------
def get_metrics(
    type_labels,
    model_id,
    ids, truths, predictions,
    cluster_range=[0],
    threshold_range=np.arange(0.1, 1.0, 0.1),
    compute_hd95=False,
    n_jobs=4,
    prediction_label=None,
):
    vrs_v_metrics = []
    vrs_c_metrics = []
    wmh_v_metrics = []
    wmh_c_metrics = []

    if cluster_range is None:
        cluster_range = [0]

    if 'VRS_WMH' in type_labels:
        # Parallel computation of metrics for VRS
        vrs_v_metrics, vrs_c_metrics = compute_metrics(
                    model_id, 'VRS',
                    ids, truths, predictions, pred_channel=0,
                    cluster_range=cluster_range,
                    threshold_range=threshold_range,
                    compute_hd95=compute_hd95,
                    n_jobs=n_jobs
                )
        # Parallel computation of metrics for WMH
        wmh_v_metrics, wmh_c_metrics = compute_metrics(
                    model_id, 'WMH',
                    ids, truths, predictions, pred_channel=1,
                    cluster_range=cluster_range,
                    threshold_range=threshold_range,
                    compute_hd95=compute_hd95,
                    n_jobs=n_jobs
                )
    elif 'WMH' in type_labels:
        # Parallel computation of metrics for WMH
        wmh_v_metrics, wmh_c_metrics = compute_metrics(
                    model_id, 'WMH',
                    ids, truths, predictions, pred_channel=0,
                    cluster_range=cluster_range,
                    threshold_range=threshold_range,
                    compute_hd95=compute_hd95,
                    n_jobs=n_jobs
                )
    elif 'VRS' in type_labels:
        # Parallel computation of metrics for VRS
        vrs_v_metrics, vrs_c_metrics = compute_metrics(
                    model_id, 'VRS',
                    ids, truths, predictions, pred_channel=0,
                    cluster_range=cluster_range,
                    threshold_range=threshold_range,
                    compute_hd95=compute_hd95,
                    n_jobs=n_jobs
                )
    else:
        # Parallel computation of metrics for ???
        vrs_v_metrics, vrs_c_metrics = compute_metrics(
                    model_id, 
                    '???' if prediction_label is None else prediction_label,
                    ids, truths, predictions, pred_channel=0,
                    cluster_range=cluster_range,
                    threshold_range=threshold_range,
                    compute_hd95=compute_hd95,
                    n_jobs=n_jobs
                )
    return vrs_v_metrics, vrs_c_metrics, wmh_v_metrics, wmh_c_metrics


# --------------------------------------------------------------------------
def compute_metrics(
    model_id, output_label,
    ids, truths, preds, pred_channel=0,
    cluster_range=[0],
    threshold_range=np.arange(0.1, 1.0, 0.1),
    compute_hd95=False,
    n_jobs=-1
):
    """ Compute metrics values for a set of subjects, truths predictions for a
    range of cluster sizes and vocels values.

    Args:
    - model_id (str): descriptive id for the model that gave the predictions
    - output_label (str): descriptive ouput label for the predictions
    - ids (list): list of subject ids,
    - truths (np.array of float): np.array of truths values for all subjects,
      must be binary truth values 0 or 1
    - preds (np.array of float): np.array of truths values for all subjects
    - pred_channel (int): channel (int the last dimension of preds) for the
      values that will be compared with truths
    - cluster_range (list, optional): list or range of cluster filters. All
      clusters of size larger than each successive values will be filtered.
      Defaults to [0].
    - threshold_range (list, optional): list or range of voxel filters.
      Defaults to np.arange(0.1, 1.0, 0.1).
    - n_jobs (int, optional): number of available cores for parallel
      procecessing. Defaults to -1.

    Returns:
    voxel_metrics, cluster_metrics (dict, dict): for each for all subjects,
    cluster_sizes and thresholds returns the metrics
    """
    # Run all thresholds / cluster sizes in //
    res = Parallel(n_jobs=n_jobs, backend="threading")(delayed(image_metrics_proxy)(  # noqa: E501
        (
            truths[0],  # post proc fn
            [p[i_subject] for p in truths[1]]
        ) if must_be_processed(truths) else truths[i_subject],
        (
            preds[0],  # post proc fn
            [p[i_subject] for p in preds[1]]
        ) if must_be_processed(preds) else preds[i_subject],
        pred_channel,
        threshold,
        model_id,
        output_label,
        ids[i_subject],
        cluster_filter=cluster_size,
        compute_hd95=compute_hd95,
    ) for threshold in threshold_range for cluster_size in cluster_range for i_subject in range(len(ids)))  # noqa: E501
    # separate voxel and cluster_metrics
    voxel_metrics = [sub[0] for sub in res]
    cluster_metrics = [sub[1] for sub in res]
    # Filter null results from metrics
    voxel_metrics = [x for x in voxel_metrics if isinstance(x, dict)]
    cluster_metrics = [x for x in cluster_metrics if isinstance(x, dict)]
    gc.collect()
    return voxel_metrics, cluster_metrics


# --------------------------------------------------------------------------
def image_metrics_proxy(
    mask, pred, channel, threshold, rname, output_label,
    subject_id, cluster_filter=0, compute_hd95=False
):
    voxel, cluster = image_metrics(
            mask,
            pred,
            channel=channel,
            threshold=threshold,
            cluster_filter=cluster_filter,
            compute_hd95=compute_hd95,
        )
    sensitivity, precision, f1, tp, fp, fn, _hd95 = voxel
    v = {
        'run': rname,
        'subject_id': subject_id,
        'output_type': 'Voxel',
        'output_label': output_label,
        'threshold': threshold,
        'cluster_filter': cluster_filter,
        'precision': precision,
        'sensitivity': sensitivity,
        'f1': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'HD95': _hd95
    }
    sensitivity, precision, f1, tp, fp, fn, _hd95 = cluster
    c = {
        'run': rname,
        'subject_id': subject_id,
        'output_type': 'Cluster',
        'output_label': output_label,
        'threshold': threshold,
        'cluster_filter': cluster_filter,
        'precision': precision,
        'sensitivity': sensitivity,
        'f1': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'HD95': _hd95
    }

    return [v, c]


# --------------------------------------------------------------------------
def get_clusters_and_filter_image(image, cluster_filter=0):
    # Compute clusters
    clusters, num_clusters = measure.label(
        image, background=0, connectivity=2, return_num=True)
    # do we have to remove some clusters ?
    if cluster_filter is not None and cluster_filter > 0:
        # find the size of each cluster except for the background
        clusnum, counts = np.unique(clusters[clusters > 0], return_counts=True)
        # find the ids of the cluster with a size < threshold
        avirer = clusnum[counts <= cluster_filter]
        # set to 0 the voxels where the cluster size is les than threshold
        image = image.copy()
        image[np.isin(clusters, avirer)] = 0
        # rebuild clusters (must be some more efficient way to do that)
        clusters_f, num_clusters_f = measure.label(
            image, background=0, connectivity=3, return_num=True)
    else:  # filtered clusters are the same
        clusters_f, num_clusters_f = clusters, num_clusters
    return image, clusters, num_clusters, clusters_f, num_clusters_f


# --------------------------------------------------------------------------
def image_metrics(
    y_true, y_pred, channel=0, threshold=0.5, cluster_filter=0,
    compute_hd95=False
):
    """
    Function that does the actual computation for mono-channel images.
    Returns a tuple of metrics.
    """
    y_true_t = postprocess_images(y_true, channel, threshold)
    y_true_t_f, cci_GT, num_GT, cci_GT_f, num_GT_f = get_clusters_and_filter_image(  # noqa: E501
        y_true_t,
        cluster_filter
    )
    y_pred_t = postprocess_images(y_pred, channel, threshold)
    y_pred_t_f, cci_DL, num_DL, cci_DL_f, num_DL_f = get_clusters_and_filter_image(  # noqa: E501
        y_pred_t,
        cluster_filter
    )
    # Voxel metrics
    vox_tp = np.sum(y_true_t_f*y_pred_t_f)
    vox_fn = np.sum(y_true_t_f*(1.-y_pred_t_f))
    vox_fp = np.sum((1.-y_true_t_f)*y_pred_t_f)
    if vox_tp == 0:
        if vox_fp == 0 and vox_fn == 0:  # there was nothing to find and we found nothing
            vox_precision = 1.
            vox_sensitivity = 1.
            vox_f1 = 1.
        else:  # there was something to find and we found nothing or something else
            vox_precision = 0.
            vox_sensitivity = 0.
            vox_f1 = 0.
    else:
            vox_sensitivity = vox_tp / (vox_tp+vox_fn)
            vox_precision = vox_tp / (vox_tp+vox_fp)
            vox_f1 = (2*vox_tp)/(2*vox_tp+vox_fp+vox_fn)  # noqa: E501

    # Cluster metrics
    # compute True Positive from Ground Truth perspective
    c = cci_GT * y_pred_t if cluster_filter == 0 else cci_GT_f * y_pred_t
    uniq = [x for x in np.unique(c) if x > 0]
    clus_tp1 = len(uniq)
    # compute False Negative
    clus_fn = num_GT - clus_tp1 if cluster_filter == 0 else num_GT_f - clus_tp1

    # compute True Positive from Prediction map perspective
    c = cci_GT * y_pred_t if cluster_filter == 0 else cci_GT * y_pred_t_f
    uniq = [x for x in np.unique(c) if x > 0]
    clus_tp2 = len(uniq)
    # compute False Positive
    allDL = np.unique(cci_DL).tolist() if cluster_filter == 0 else np.unique(cci_DL_f).tolist()  # noqa: E501
    allDL = [x for x in allDL if x > 0]
    c = cci_DL * y_true_t if cluster_filter == 0 else cci_DL_f * y_true_t
    uniq = [x for x in np.unique(c) if x > 0]
    uniq2 = [int(x) for x in allDL if x not in uniq]
    clus_fp = len(uniq2)

    if clus_tp1 == 0 and clus_tp2 == 0:
        if clus_fn == 0 and clus_fp == 0:  # there was nothing to find and we found nothing
            clus_precision = 1.
            clus_sensitivity = 1.
            clus_f1 = 1.
        else:  # there was something to find and we found nothing or something else
            clus_precision = 0.
            clus_sensitivity = 0.
            clus_f1 = 0.
    elif clus_tp1 == 0:
        if clus_fn == 0 and clus_fp == 0:  # there was nothing to find and we found nothing
            clus_sensitivity = 1
        else:  # there was something to find and we found nothing or something else
            clus_sensitivity = 0
        clus_precision = clus_tp2 / (clus_tp2+clus_fp)
        clus_f1 = 2*(clus_precision*clus_sensitivity)/(clus_precision+clus_sensitivity)
    elif clus_tp2 == 0:
        if clus_fn == 0 and clus_fp == 0:  # there was nothing to find and we found nothing
            clus_precision = 1
        else:  # there was something to find and we found nothing or something else
            clus_precision = 0
        clus_sensitivity = clus_tp1 / (clus_tp1+clus_fn)
        clus_f1 = 2*(clus_precision*clus_sensitivity)/(clus_precision+clus_sensitivity)
    else:
        clus_precision = clus_tp2 / (clus_tp2+clus_fp)
        clus_sensitivity = clus_tp1 / (clus_tp1+clus_fn)
        clus_f1 = 2*(clus_precision*clus_sensitivity)/(clus_precision+clus_sensitivity)

    if compute_hd95:
        try:
            c = cci_GT_f * y_pred_t_f
            uniq = [x for x in np.unique(c) if x > 0]
            TP_in_GT = cci_GT_f.copy()
            TP_in_GT[np.isin(TP_in_GT, uniq)] = -1
            TP_in_GT[TP_in_GT > 0] = 0
            TP_in_GT = -1 * TP_in_GT
            c = cci_DL_f * y_true_t_f
            uniq = [x for x in np.unique(c) if x > 0]
            TP_in_DL = cci_DL_f.copy()
            TP_in_DL[np.isin(TP_in_DL, uniq)] = -1
            TP_in_DL[TP_in_DL > 0] = 0
            TP_in_DL = -1 * TP_in_DL
            _hd95 = hd95(TP_in_DL, TP_in_GT)
        except ZeroDivisionError:
            _hd95 = np.nan
        except RuntimeError:
            _hd95 = np.nan
    else:
        _hd95 = np.nan
    return (
        (vox_sensitivity, vox_precision, vox_f1, vox_tp, vox_fp, vox_fn, _hd95),  # noqa: E501
        (clus_sensitivity, clus_precision, clus_f1, clus_tp1, clus_fp, clus_fn, _hd95)  # noqa: E501
    )


# --------------------------------------------------------------------------
def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD)
    between the binary objects in two images. Compared to the Hausdorff
    Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`. Note that the connectivity influences the result in the
        case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result```
        and the object(s) in ```reference```. The distance unit is the same as
        for the spacing of elements along each dimension, which is usually
        given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any
    order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and
    their nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)  # noqa: E501
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')  # noqa: E501
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')  # noqa: E501

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(
        result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of
    # the foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


# --------------------------------------------------------------------------
def must_be_processed(images) -> bool:
    """[summary]

    Args:
        images (np.array or (fn, np.array)): if a np.array then False, else
        test if the first element is a callable to be applied on the second

    Returns:
        bool: True if the image includes a post proc function
    """
    try:
        return callable(images[0])
    except Exception:
        return False


def postprocess_images(images, channel=0, threshold=0.5):
    """ Threshold and apply some post processing to a set of images if needed.
    This is useful if some other function than averaging is needed to
    'ensemble' a set of predictions from different predictors.

    Args:
        - images (np.array or (fn, np.array)): if it is a np.array then only
        thresholding is done and no other post processing is applied. If it is
        a tuple (function, images) then the function is applied to the images
        and must returns an image with at least the thresholding done ; the
        function must have 2 parameters f(images:np.array, threshold: float)->
        np.array.
        - channel (int): which channel (last index of the image) to use.
        - threshold (float): the threshold.

    Returns:
        np.ndarray of int: post processed image of (0,1)
    """
    if must_be_processed(images):
        return images[0](images[1], channel, threshold)
    else:
        return (images[..., channel] > threshold).astype(int)


def hilo(images, channel=0, threshold=0.5):
    """ used to generate predictions from 'hilo' model, the first half
    predictions will be dedicated to low volume predictions, the second half to
    high volume predictions. Then first half is averaged & thresholded, as for
    the second half, finally the union of first and second halves is returned.

    Args:
        - images (np.array): images
        - channel (int): which channel (last index of the image) to use.
        - threshold (float): threshold
    Returns:
        np.ndarray of int: post processed image of (0,1)
    """
    half = len(images) // 2
    return (
        (np.stack(images[:half]).mean(axis=0) > threshold) |
        (np.stack(images[half+1:]).mean(axis=0) > threshold)
        )[..., channel].astype(int)
