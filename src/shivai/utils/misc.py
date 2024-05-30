"""
Miscellaneous functions usefull in multiple scripts
"""

import hashlib
import os
import pathlib
import json

from shivai.utils.stats import get_mode
from shivai.utils.metrics import get_clusters_and_filter_image

import numpy as np
import nibabel as nib
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN
from jinja2 import Template


def md5(fname):
    """
    Create a md5 hash for a file or a folder

    Args:
        fname (str): file/folder path

    Returns:
        str: hexadecimal hash for the file/folder
    """
    hash_md5 = hashlib.md5()
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    elif os.path.isdir(fname):
        fpath = pathlib.Path(fname)
        file_list = [f for f in fpath.rglob('*') if os.path.isfile(f)]
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
