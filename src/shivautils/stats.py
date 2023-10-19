from statistics import mean
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from jinja2 import Template
import pandas as pd
import nibabel as nb

from statistics import median

from shivautils.metrics import get_clusters_and_filter_image


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


def save_histogram(img_normalized: nb.Nifti1Image,
                   bins: int = 64):
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

    array = img_normalized.get_fdata()
    x = array.reshape(-1)
    hist, edges = np.histogram(x, bins=bins)

    fig, ax = plt.subplots()
    ax.hist(x, bins=edges, color=(0.3, 0.5, 0.8))
    ax.set_yscale('log')
    ax.set_title("Histogram of intensities voxel values")
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("Number of Voxels")

    histogram = 'hist.svg'
    plt.savefig('hist.svg')
    return (op.abspath(histogram))


def bounding_crop(img_apply_to: nb.Nifti1Image,
                  brainmask: nb.Nifti1Image,
                  bbox1: tuple,
                  bbox2: tuple,
                  cdg_ijk: tuple):
    """Overlay of brainmask over nifti images

    Args:
        img_apply_to (nb.Nifti1Image): t1 nifti images overlay with brainmask and crop box
        brainmask (nb.Nifti1Image): nifti brainmask file
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

    img_apply_to = img_apply_to.get_fdata()
    brainmask = brainmask.get_fdata()
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
    return (op.abspath('bounding_crop.svg'))


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

    img_ref = nb.load(img_ref)
    img_ref = img_ref.get_fdata()
    brainmask = nb.load(brainmask)
    brainmask = brainmask.get_fdata()

    # creating cmap to overlay reference image and given image
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    sli_ref = []
    sli_brainmask = []
    list_sli_Z = []

    for j in range(10, img_ref.shape[2] - 6, 4):
        sli_ref.append(img_ref[:, :, int(j)])
        sli_brainmask.append(brainmask[:, :, int(j)])
        list_sli_Z.append(int(j))

    # Affichage dans les trois axes
    fig, ax = plt.subplots(5, 8, figsize=(160, 80))
    fig.patch.set_facecolor('k')
    alpha = 0.5
    count = 0

    for j in range(8):

        for i in range(5):
            # Axe z
            ax[i, j].imshow(ndimage.rotate(sli_ref[count], 90),
                            cmap='gray')
            ax[i, j].imshow(ndimage.rotate(sli_brainmask[count], 90),
                            alpha=0.5, cmap=my_cmap)

            label = ax[i, j].set_xlabel('k = ' + str(list_sli_Z[count]))
            label.set_fontsize(30)  # Définir la taille de la police du label
            label.set_color('white')
            ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].get_yaxis().set_ticks([])

            count += 1

    fig.tight_layout()

    plt.savefig('qc_overlay_brainmask_T1.png', bbox_inches='tight', pad_inches=0.1)
    return (op.abspath('qc_overlay_brainmask_T1.png'))


def prediction_metrics(array_vol, threshold, cluster_filter):
    """Get metrics on array Nifti prediction file

    Args:
        array_vol (array): array Nifti prediction file
        threshold (float): Threshold to compute clusters metrics
        cluster_filter (int): number of voxels (strictly) above which the cluster is counted

    Returns:
        Dataframe: Labels and size of each cluster
        Dataframe: Summary metrics of the clusters 
            (sum voxel segmented, number of cluster, mean size cluster, 
            median size cluster, min size cluster, max size cluster)
        Array: Labelled clusters
    """

    if len(array_vol.shape) > 3:
        array_vol = array_vol.squeeze()
    thresholded_img = (array_vol > threshold).astype(int)
    _, _, _, clusters_vol, _ = get_clusters_and_filter_image(thresholded_img, cluster_filter)
    clust_labels, clust_size = np.unique(clusters_vol, return_counts=True)
    cluster_measures = pd.DataFrame(
        {'Biomarker_labels': clust_labels,
         'Biomarker_size': clust_size})
    cluster_stats = pd.DataFrame(
        {'Total_biomarker_volume':cluster_measures['Biomarker_size'].sum(),
         'Mean_biomarker_volume':cluster_measures['Biomarker_size'].mean(),
         'Median_biomarker_volume':cluster_measures['Biomarker_size'].median(),
         'StD_biomarker_volume': cluster_measures['Biomarker_size'].std(),
         'Min_biomarker_volume': cluster_measures['Biomarker_size'].min(),
         'Max_biomarker_volume': cluster_measures['Biomarker_size'].max()})

    return cluster_measures, cluster_stats, clusters_vol


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
