from statistics import mean
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from jinja2 import Template
import pandas as pd
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import median
import os
from scipy.spatial.transform import Rotation
from scipy.io import loadmat

from shivautils.utils.metrics import get_clusters_and_filter_image


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


def set_percentile(list_node: list,
                   list_percentile: list,
                   bins: int,
                   list_img_cohort: list) -> int:
    """Compared mode distribution for different percentile
    modalities in normalization and return percentile value most adapted
    at the given cohort image

    Args:
        list_node (list): list of nodes obtained after
        workflow with normalization
        list_percentile (list): lists of different percentile values to test
        bins (int) : number of batchs in cohorte image histogram
        list_img_cohort : list of nibabel image to compare with preprocessed image

    Returns:
        int: percentile value most adapted at the mode distribution cohort image
    """
    list_mode = []
    for i in list_node:
        # filters only nodes resulting from normalization
        if i.name == 'normalization':
            list_mode.append(i.result.outputs.mode)

    mode_by_percentile = []
    for i in range(len(list_percentile)):
        # create a table, create inside a table for each percentile value,
        # put the mode values of each image for the percentile value in
        # the corresponding table
        mode_by_percentile.append([])
        for j in range(i, len(list_mode), len(list_percentile)):
            mode_by_percentile[i].append(list_mode[j])

    for i in mode_by_percentile:
        mode_by_percentile[mode_by_percentile.index(i)] = mean(i)

    list_array_cohort = []
    for i in list_img_cohort:
        list_array_cohort.append(i.get_fdata().reshape(-1))

    list_mode_cohort = []
    for i in list_array_cohort:
        hist, edges = np.histogram(i, bins)
        mode = get_mode(hist, edges, bins)
        list_mode_cohort.append(mode)
    mode_mean_cohort = mean(list_mode_cohort)

    mode_fit = mode_by_percentile[min(range(len(mode_by_percentile)), key=lambda i: abs(mode_by_percentile[i]-mode_mean_cohort))]
    percentile_fit = list_percentile[mode_by_percentile.index(mode_fit)]

    return percentile_fit


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


def save_histogram(img_normalized,
                   bins=64):
    """Save histogram of intensity normalization voxels on the disk

    Args:
        img_normalized (nb.Nifti1Image): intensity normalize images to obtain histogram
        bins (int, optional): Number of element on histogram file. Defaults to 64.

    Returns:
        path: file path of histogram intensity voxels
    """

    import nibabel as nb
    import matplotlib.pyplot as plt
    import numpy as np
    import os.path as op
    from shivautils.utils.stats import get_mode

    if not isinstance(img_normalized, nb.Nifti1Image):
        if op.isfile(img_normalized):
            img_normalized = nb.load(img_normalized)
        else:
            raise ValueError('The input should have been a Nifti1Image or a nifti file, but was neither.')
    array = img_normalized.get_fdata()
    x = array.flatten()
    _, ax = plt.subplots()
    (hist_val, hist_edges, _) = plt.hist(x, bins=bins)
    mode = get_mode(hist_val, hist_edges, bins)  # Peak intensity away from min and max intensities
    ax.set_yscale('log')
    ax.set_title("Histogram of intensities voxel values")
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("Number of Voxels")

    histogram = 'hist.svg'
    plt.savefig('hist.svg')
    return op.abspath(histogram), mode


def bounding_crop(img_apply_to: str,
                  brainmask: str,
                  bbox1: tuple,
                  bbox2: tuple,
                  cdg_ijk: tuple):
    """Overlay of brainmask over nifti images

    Args:
        img_apply_to (path): t1 nifti images overlay with brainmask and crop box
        brainmask (path): nifti brainmask file
        bbox1 (tuple): first coordoninates point of cropping box
        bbox2 (tuple): second coordonaites point of cropping box
        cdg_ijk (tuple): t1 nifti image center of mass used to calculate cropping box

    Returns:
        svg: svg file of cropping box with overlay brainmask
    """

    import os.path as op
    import matplotlib.pyplot as plt
    import numpy as np
    import nibabel as nb
    from matplotlib.colors import ListedColormap

    img_apply_to = nb.load(img_apply_to).get_fdata()
    brainmask = nb.load(brainmask).get_fdata().astype(bool)
    original_dims = img_apply_to.shape
    # Coordonnées de la boîte de recadrage
    x_start, y_start, z_start = (bbox1[0], bbox1[1], bbox1[2])
    x_end, y_end, z_end = (bbox2[0], bbox2[1], bbox2[2])

    # Création d'une matrice tridimensionnelle pour les contours de la boîte de recadrage
    contour = np.zeros(original_dims, dtype=np.uint8)
    contour[x_start:x_end, y_start:y_end, z_start:z_end] = original_dims[0] - 1

    contour[cdg_ijk[0], cdg_ijk[1], cdg_ijk[2]] = 10

    # Affichage dans les trois axes
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # fig.patch.set_facecolor('k')
    alpha = 0.7

    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    # Axe x
    ax[0].imshow(np.rot90(contour[cdg_ijk[0], :, :]), alpha=0.8)
    ax[0].imshow(np.rot90(img_apply_to[cdg_ijk[0], :, :]), cmap='gray', alpha=0.85)
    ax[0].imshow(np.rot90(brainmask[cdg_ijk[0], :, :]), cmap=my_cmap, alpha=alpha)
    ax[0].axis('off')
    ax[0].set_title('Sagittal Axis')

    # Axe y
    ax[1].imshow(np.rot90(contour[:, cdg_ijk[1], :]), alpha=0.8)
    ax[1].imshow(np.rot90(img_apply_to[:, cdg_ijk[1], :]), cmap='gray', alpha=0.85)
    ax[1].imshow(np.rot90(brainmask[:, cdg_ijk[1], :]), cmap=my_cmap, alpha=alpha)
    ax[1].axis('off')
    ax[1].set_title('Coronal Axis')

    # Axe z
    ax[2].imshow(np.rot90(contour[:, :, cdg_ijk[2]]), alpha=0.8)
    ax[2].imshow(np.rot90(img_apply_to[:, :, cdg_ijk[2]]), cmap='gray', alpha=0.85)
    ax[2].imshow(np.rot90(brainmask[:, :, cdg_ijk[2]]), cmap=my_cmap, alpha=alpha)
    ax[2].axis('off')
    ax[2].set_title('Axial Axis')

    fig.tight_layout()
    plt.savefig('bounding_crop.svg', bbox_inches='tight', pad_inches=0.1)
    return op.abspath('bounding_crop.svg')


def overlay_brainmask(img_ref,
                      brainmask):
    """Overlay brainmask on t1 images for 40 slices

    Args:
        img_ref (nb.Nifti1Image): t1 nifti images overlay with brainmask
        brainmask (nb.Nifti1Image): brainmask file to overlay with t1 nifti file

    Returns:
        png: png file with 40 slices of overlay brainmask on t1 nifti file
    """

    import nibabel as nb
    from scipy import ndimage
    import numpy as np
    import os.path as op
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from math import ceil

    ref_im = nb.load(img_ref)
    ref_vol = ref_im.get_fdata()
    brainmask_im = nb.load(brainmask)
    brainmask_vol = brainmask_im.get_fdata()

    nb_of_slices = 6  # define the number of columns in the image

    # creating cmap to overlay reference image and given image
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    # Getting slices in each dim
    crop_ratio = 0.1  # 10% of the slice
    border_crop_X = ceil(crop_ratio * ref_vol.shape[0])
    croped_shape_X = (1-2*crop_ratio)*ref_vol.shape[0]  # can be non-integer at this step, not important
    slices_ind_X = np.arange(border_crop_X, ref_vol.shape[0]-border_crop_X, croped_shape_X/nb_of_slices).astype(int)
    slices_X = [(ref_vol[ind, :, :], brainmask_vol[ind, :, :], ind) for ind in slices_ind_X]

    border_crop_Y = ceil(crop_ratio * ref_vol.shape[1])
    croped_shape_Y = (1-2*crop_ratio)*ref_vol.shape[1]
    slices_ind_Y = np.arange(border_crop_Y, ref_vol.shape[1]-border_crop_Y, croped_shape_Y/nb_of_slices).astype(int)
    slices_Y = [(ref_vol[:, ind, :], brainmask_vol[:, ind, :], ind) for ind in slices_ind_Y]

    border_crop_Z = ceil(crop_ratio * ref_vol.shape[2])
    croped_shape_Z = (1-2*crop_ratio)*ref_vol.shape[2]
    slices_ind_Z = np.arange(border_crop_Z, ref_vol.shape[2]-border_crop_Z, croped_shape_Z/nb_of_slices).astype(int)
    slices_Z = [(ref_vol[:, :, ind], brainmask_vol[:, :, ind], ind) for ind in slices_ind_Z]

    # Affichage dans les trois axes
    fig, ax = plt.subplots(3, nb_of_slices, figsize=(nb_of_slices, 3), dpi=300)
    fig.patch.set_facecolor('k')
    alpha = 0.5

    for col in range(nb_of_slices):
        # X
        ax[0, col].imshow(slices_X[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[0, col].imshow(slices_X[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_X = ax[0, col].set_xlabel(f'k = {slices_X[col][2]}')
        label_X.set_fontsize(2)  # Définir la taille de la police du label
        label_X.set_color('white')
        ax[0, col].get_xaxis().set_ticks([])
        ax[0, col].get_yaxis().set_ticks([])
        # Y
        ax[1, col].imshow(slices_Y[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[1, col].imshow(slices_Y[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_Y = ax[1, col].set_xlabel(f'k = {slices_Y[col][2]}')
        label_Y.set_fontsize(2)  # Définir la taille de la police du label
        label_Y.set_color('white')
        ax[1, col].get_xaxis().set_ticks([])
        ax[1, col].get_yaxis().set_ticks([])
        # Z
        ax[2, col].imshow(slices_Z[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[2, col].imshow(slices_Z[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_Z = ax[2, col].set_xlabel(f'k = {slices_Z[col][2]}')
        label_Z.set_fontsize(2)  # Définir la taille de la police du label
        label_Z.set_color('white')
        ax[2, col].get_xaxis().set_ticks([])
        ax[2, col].get_yaxis().set_ticks([])

    fig.tight_layout()

    plt.savefig('qc_overlay_brainmask_T1.png')
    return (op.abspath('qc_overlay_brainmask_T1.png'))


def prediction_metrics(array_vol, threshold, cluster_filter, brain_seg_vol,
                       region_dict={'Whole brain': [-1]},
                       prio_labels=[]):
    """Get metrics on sgmented biomarkers by brain region

    Args:
        array_vol (array): Biomarker segmentation array (from the nifti file)
        threshold (float): Threshold to compute clusters metrics
        cluster_filter (int): number of voxels (strictly) above which the cluster is counted
        brain_seg_vol (array): Brain segmentation (or brain mask), should be filled with integers
        region_dict (dict): Dict pairing the brain regions' name (key) with the numeric label (as int) to use when
                            counting the biomarkers.
                            '-1' denotes the region encompassing the whole brain segmentation > 0
                            (must be associated with the 'Whole brain' key)
    Returns:
        Dataframe: Labels and size of each cluster
        Dataframe: Summary metrics of the clusters 
            (sum voxel segmented, number of cluster, mean size cluster, 
            median size cluster, min size cluster, max size cluster)
        Array: Labelled clusters
    """

    if len(array_vol.shape) > 3:
        array_vol = array_vol.squeeze()
    if len(brain_seg_vol.shape) > 3:
        brain_seg_vol = brain_seg_vol.squeeze()

    # Associate a label with a region name
    swaped_region_dict = {val: reg for reg, val in region_dict.items()}
    if prio_labels:
        prio_dict = {reg: region_dict[reg] for reg in prio_labels}

    # Threshold and mask the biomarker segmentation image
    brain_mask = (brain_seg_vol > 0)
    thresholded_img = (array_vol > threshold).astype(int)*brain_mask

    _, _, _, clusters_vol, _ = get_clusters_and_filter_image(thresholded_img, cluster_filter)
    clust_labels, clust_size = np.unique(clusters_vol[clusters_vol > 0], return_counts=True)

    cluster_measures = pd.DataFrame(
        {'Biomarker_labels': clust_labels,
         'Biomarker_size': clust_size})

    regions = []
    biom_num = []
    biom_tot = []
    biom_mean = []
    biom_med = []
    biom_std = []
    biom_min = []
    biom_max = []
    # Default case: get whole brain metrics
    if 'Whole brain' in region_dict.keys():
        del region_dict['Whole brain']
        regions.append('Whole brain')
        biom_num.append(cluster_measures.shape[0])
        biom_tot.append(cluster_measures['Biomarker_size'].sum())
        biom_mean.append(cluster_measures['Biomarker_size'].mean())
        biom_med.append(cluster_measures['Biomarker_size'].median())
        biom_std.append(cluster_measures['Biomarker_size'].std())
        biom_min.append(cluster_measures['Biomarker_size'].min())
        biom_max.append(cluster_measures['Biomarker_size'].max())

    # Attribution of one region per cluster (i.e. the most represented region in each cluster = winner-takes-all)
    if len(region_dict):
        clust_reg = []
        for clust in clust_labels:
            seg_clust = brain_seg_vol[clusters_vol == clust]
            reg_in_clust, reg_count = np.unique(seg_clust, return_counts=True)  # There shouldn't be any 0 here
            max_count = reg_count.max()
            seg_attributed_label = np.random.choice(reg_in_clust[reg_count == max_count])  # takes equalities into account
            if prio_labels:  # Overriding the w-t-a approach for priority labels
                prio_ind = np.isin(reg_in_clust, list(prio_dict.values()))
                if prio_ind.any():
                    seg_attributed_label = np.random.choice(reg_in_clust[prio_ind])  # takes equalities into account

            if seg_attributed_label in swaped_region_dict.keys():
                seg_attributed = swaped_region_dict[seg_attributed_label]
            else:  # keeping the raw label if it's not part of the investigated regions
                if seg_attributed_label:  # Non-zero label
                    seg_attributed = f'Label_{seg_attributed_label}'
            clust_reg.append(seg_attributed)

        cluster_measures['Biomarker_region'] = clust_reg

        regions_seg = [reg for reg in region_dict.keys() if reg in clust_reg]  # Sorted like in the input dict
        regions += regions_seg

        for reg in regions_seg:
            clusts_in_reg = cluster_measures.loc[cluster_measures['Biomarker_region'] == reg]
            biom_num.append(clusts_in_reg.shape[0])
            biom_tot.append(clusts_in_reg['Biomarker_size'].sum())
            biom_mean.append(clusts_in_reg['Biomarker_size'].mean())
            biom_med.append(clusts_in_reg['Biomarker_size'].median())
            biom_std.append(clusts_in_reg['Biomarker_size'].std())
            biom_min.append(clusts_in_reg['Biomarker_size'].min())
            biom_max.append(clusts_in_reg['Biomarker_size'].max())

    cluster_stats = pd.DataFrame(
        {'Region': regions,
         'Number_of_biomarkers': biom_num,
         'Total_biomarker_volume': biom_tot,
         'Mean_biomarker_volume': biom_mean,
         'Median_biomarker_volume': biom_med,
         'StD_biomarker_volume': biom_std,
         'Min_biomarker_volume': biom_min,
         'Max_biomarker_volume': biom_max})

    return cluster_measures, cluster_stats, clusters_vol


def swarmplot_from_census(census_csv: str, pred: str):
    census_df = pd.read_csv(census_csv)
    save_name = f'{pred}_census_plot.svg'
    plt.ioff()
    if 'Region_names' not in census_df.columns:
        fig = plt.figure()
        sns.stripplot(census_df, y='Biomarker_size')  # replaced swamplot
        plt.savefig(save_name, format='svg')
        plt.close(fig)
    else:
        # Change all non-identified regions by "Other"
        FS_id = ['FreeSurfer_' in reg for reg in census_df['Region_names']]
        census_df.loc[FS_id, 'Region_names'] = 'Other'
        fig = plt.figure()
        sns.stripplot(census_df, y='Biomarker_size', hue='Region_names')
        plt.savefig(save_name, format='svg')
        plt.close(fig)
    return os.path.abspath(save_name)


def get_mask_regions(img: nb.Nifti1Image,
                     list_labels_regions: list):
    """Filter in a regions segmented images regions of interest specified
    in 'list_labels_regions'

    Args:
        img (nb.Nifti1Image): image segmented by regions 
        list_labels_regions (list): list of int corresponding Fresurfer numerotation

    Returns:
        nb.Nifti1Image:segmented image with specific regions
    """

    import nibabel as nb
    import numpy as np

    pred_img_array = img.get_fdata()
    mask = np.isin(pred_img_array, list_labels_regions).astype(int)
    mask_regions = nb.Nifti1Image(mask, img.affine, img.header)

    return mask_regions


def transf_from_affine(mat_file: str):
    """
    Import the affine transformation from a .mat file generated by an ANTs registration
    and output the norm of its rotation (in degrees) and translation
    """
    # import numpy as np
    # from scipy.spatial.transform import Rotation
    # from scipy.io import loadmat

    mat = loadmat(mat_file)
    rot = mat['AffineTransform_double_3_3'][0:9].reshape((3, 3))
    trans = mat['AffineTransform_double_3_3'][9:12]
    r = Rotation.from_matrix(rot)
    r_tot = np.abs(r.as_euler('xyz', degrees=True)).sum()
    t_norm = np.linalg.norm(trans)
    return r_tot, t_norm
