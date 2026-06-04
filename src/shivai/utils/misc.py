"""
Miscellaneous functions usefull in multiple scripts
"""

import hashlib
import os
from pathlib import Path
import json


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN
from jinja2 import Template
from functools import reduce

from skimage import measure

from nipype.pipeline.engine import Workflow


def md5(fname: Path):
    """
    Create a md5 hash for a file or a folder

    Args:
        fname (str): file/folder path

    Returns:
        str: hexadecimal hash for the file/folder
    """
    if isinstance(fname, str):
        fname = Path(fname)
    hash_md5 = hashlib.md5()
    if fname.is_file():
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    elif fname.is_dir():
        file_list = [f for f in fname.rglob('*') if f.is_file()]
        file_list.sort()
        for sub_file in file_list:
            with open(sub_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    else:
        raise ValueError(f'The input is neither a file nor a folder: {fname}')
    return hash_md5.hexdigest()


def get_md5_from_json(json_path, get_url=False):
    with open(json_path) as f:
        meta_data = json.load(f)
    uid_model = {}
    for i, model_file in enumerate(meta_data["files"]):
        filename = os.path.basename(model_file["name"])
        uid_model[f'Model file {i + 1}'] = (filename, model_file["md5"])
    if get_url:
        if 'url' in meta_data:
            url = meta_data["url"]
        else:
            url = None
        return uid_model, url
    else:
        return uid_model


def set_wf_shapers(predictions):
    """
    Set with_t1, with_flair, and with_swi with the corresponding value depending on the
    segmentations (predictions) that will be done.
    The tree boolean variables are used to shape the main and postproc workflows
    (e.g. if doing PVS and CMB, the wf will use T1 and SWI)
    """
    # Setting up the different cases to build the workflows (should clarify things up)
    if any(pred in predictions for pred in ['PVS', 'PVS2', 'WMH', 'LAC']):  # all which requires T1
        with_t1 = True
    else:
        with_t1 = False
    if any(pred in predictions for pred in ['PVS2', 'WMH', 'LAC']):
        with_flair = True
    else:
        with_flair = False
    if 'CMB' in predictions:
        with_swi = True
    else:
        with_swi = False
    return with_t1, with_flair, with_swi


def as_list(arg_in):
    return [arg_in]


def file_selector(inArg, fileNum):
    if isinstance(inArg, list) or isinstance(inArg, tuple):
        if len(inArg) <= fileNum:
            raise IndexError(f'The node where "file_selector" was used gave a list of {len(inArg)} elements '
                             f'but the function tried to get the element {fileNum} (out of range)')
        return inArg[fileNum]
    else:
        return inArg


def get_mode(hist: np.array,
             edges: np.array,
             bins: int):
    """Get most frequent value in an numpy histogram composed by
    frequence (hist) and values (edges)

    Args:
        hist (np.array): frequence of values
        edges (np.array): different values possible in histogram
        bins (int): number of batchs

    Returns:
        mode (int): most frequent value
    """
    inf = int(0.2 * bins)
    sup = int(0.9 * bins)
    index = np.where(hist[inf:sup] == hist[inf:sup].max())
    mode = edges[inf+index[0]][0]

    return mode


def histogram(array, percentile, bins):
    """Create an histogram with a numpy array. Retrieves the largest value in
    the first axis of of the histogram and returns the corresponding value on
    the 2nd axe (that of the voxel intensity value) between two bounds of the
    histogram to be defined

    Args:
        array (array): histogram with 2 axes: one for the number of voxels and
                       the other for the intensity value of the voxels
        bins (int): number of batchs to gather data

    Returns:
        mode (int): most frequent value of histogram
    """
    x = array.reshape(-1)
    hist, edges = np.histogram(x, bins=bins)

    mode = get_mode(hist, edges, bins)

    p = figure(title="Histogram of intensities voxel values", width=400,
               height=400, y_axis_type='log')
    p.quad(top=hist, bottom=1, left=edges[:-1], right=edges[1:],
           line_color=None)
    html = file_html(p, CDN, "histogram of voxel intensities")

    tm = Template(
        """<!DOCTYPE html>
                 <html lang="en">
                 <head>
                    <meta charset="UTF-8">
                    <title>My template</title>
                 </head>
                 <body>
                 <div class="test">
                    <object data="data:application/html;base64,{{ pa }}"></object>
                    <h2>Percentile : {{ percentile}}</h2>
                    <h2>Mode : {{ mode }}</h2>
                 </div>
                 </body>
                 </html>"""
    )

    template_hist = tm.render(pa=html, percentile=percentile, mode=mode)

    return template_hist, mode


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


def fisin(arr, vals):
    '''
    Fast np.isin function using reccursive bitwise_or function 
    (here represented with the lambda function, because slightly faster(?))
    '''
    try:
        arrl = [arr == val for val in vals]
    except TypeError:  # Typically if there is only 1 value
        arrl = [arr == vals]
    return reduce(lambda x, y: x | y, arrl)


def salient_slices(vol: np.ndarray, slices_ind: tuple[int, int, int] = None) -> tuple[int, int, int]:
    """Get the most salient slices of a 3D volume (i.e. those with the most non-zero voxels)
    and display them with nilearn plot_anat function.
    Mostly for QC / tests purposes, not really for the pipeline itself
    Args:
        vol (np.ndarray): 3D volume

    Returns:
        tuple[int, int, int]: the indices of the most salient slice of each plane
    """
    if slices_ind is None:
        # Salient slices are those with the most non-zero voxels. Find the max for each dim
        X_slice = np.argmax(np.sum(vol > 0, axis=(1, 2)))
        Y_slice = np.argmax(np.sum(vol > 0, axis=(0, 2)))
        Z_slice = np.argmax(np.sum(vol > 0, axis=(0, 1)))
        slices_ind = (X_slice, Y_slice, Z_slice)
    else:
        X_slice, Y_slice, Z_slice = slices_ind
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(vol[X_slice, :, :].T, cmap='gray', origin='lower')
    plt.title(f'X slice {X_slice}')
    plt.subplot(1, 3, 2)
    plt.imshow(vol[:, Y_slice, :].T, cmap='gray', origin='lower')
    plt.title(f'Y slice {Y_slice}')
    plt.subplot(1, 3, 3)
    plt.imshow(vol[:, :, Z_slice].T, cmap='gray', origin='lower')
    plt.title(f'Z slice {Z_slice}')
    plt.show()
    return slices_ind


def _export_workflow_compat(workflow: Workflow, filename: str = None, prefix: str = "output_"):
    """Export workflow with a compatibility patch for Nipype 1.9.x string handling bugs."""
    from nipype.pipeline.engine import utils as pe_utils
    from nipype.pipeline.engine import workflows as pe_workflows
    from nipype.interfaces.base.traits_extension import isdefined
    from nipype.utils.functions import create_function_from_source

    original_write_inputs = pe_utils._write_inputs
    original_pickle_loads = pe_workflows.pickle.loads

    def _compat_pickle_loads(value):
        if isinstance(value, str):
            return value
        return original_pickle_loads(value)

    def _write_inputs_compat(node):
        lines = []
        nodename = node.fullname.replace('.', '_')
        for key, _ in list(node.inputs.items()):
            val = getattr(node.inputs, key)
            if isdefined(val):
                if isinstance(val, (str, bytes)):
                    try:
                        func = create_function_from_source(val)
                    except (RuntimeError, AssertionError, TypeError):
                        lines.append(f"{nodename}.inputs.{key} = {val!r}")
                    else:
                        funcname = [name for name in func.__globals__ if name != '__builtins__'][0]
                        # Nipype 1.9.x expects pickled bytes in this branch but Function stores source as str.
                        # try:
                        #     lines.append(pickle.loads(val))
                        # except Exception:
                        #     lines.append(val.decode() if isinstance(val, bytes) else val)
                        lines.append(_compat_pickle_loads(val).rstrip())
                        if funcname == nodename:
                            lines[-1] = lines[-1].replace(
                                f" {funcname}(", f" {funcname}_1("
                            )
                            funcname = f"{funcname}_1"
                        lines.append('from nipype.utils.functions import getsource')
                        lines.append(f"{nodename}.inputs.{key} = getsource({funcname})")
                else:
                    lines.append(f"{nodename}.inputs.{key} = {val}")
        return lines

    pe_utils._write_inputs = _write_inputs_compat
    pe_workflows.pickle.loads = _compat_pickle_loads
    try:
        return workflow.export(filename, prefix)
    finally:
        pe_utils._write_inputs = original_write_inputs
        pe_workflows.pickle.loads = original_pickle_loads
