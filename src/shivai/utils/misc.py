"""
Miscellaneous functions usefull in multiple scripts
"""

import hashlib
import os
from pathlib import Path
import json


import numpy as np
import nibabel as nib
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN
from jinja2 import Template
from functools import reduce

from skimage import measure


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
        raise FileNotFoundError(f'The input is neither a file nor a folder: {fname}')
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


def get_clusters_and_filter_image(image, cluster_filter=0):
    """ 
    Compute clusters and filter out those of size "cluster_filter" and smaller

    """
    clusters, num_clusters = measure.label(
        image, return_num=True)
    if cluster_filter:
        clusnum, counts = np.unique(clusters[clusters > 0], return_counts=True)
        to_remove = clusnum[counts <= cluster_filter]
        nums_left = [i for i in clusnum if i not in to_remove]

        image_f = image.copy()
        clusters_f = clusters.copy()
        if to_remove.size:
            image_f[fisin(clusters, to_remove)] = 0
            clusters_f[fisin(clusters, to_remove)] = 0
        num_clusters_f = num_clusters - len(to_remove)

        for new_i, old_i in enumerate(nums_left):
            new_i += 1  # because starts at 0
            clusters_f[clusters == old_i] = new_i
    else:  # filtered clusters are the same
        image_f, clusters_f, num_clusters_f = image, clusters, num_clusters
    return image_f, clusters, num_clusters, clusters_f, num_clusters_f


def label_clusters(pred_vol, brain_seg_vol, threshold, cluster_filter):
    """Threshold and labelize the clusters from a prediction map

    Args:
        pred_vol (np.ndarray): Prediction map from the AI model
        brain_seg_vol (np.ndarray): Brain seg delimiting the brain
        threshold (float): Value to threshold the prediction map
        cluster_filter (int): size up to which (including) small clusters are removed

    Returns:
        labelled_clusters (np.ndarray): Labelled clusters volume
    """
    if len(pred_vol.shape) > 3:
        pred_vol = pred_vol.squeeze()
    if len(brain_seg_vol.shape) > 3:
        brain_seg_vol = brain_seg_vol.squeeze()
    brain_mask = (brain_seg_vol > 0)
    thresholded_img = (pred_vol > threshold).astype(int)*brain_mask
    _, _, _, labelled_clusters, _ = get_clusters_and_filter_image(thresholded_img, cluster_filter)
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
