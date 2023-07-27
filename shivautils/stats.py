from statistics import mean
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from jinja2 import Template
import base64
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from os import listdir
import os
import csv
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

    print(mode_by_percentile)

    list_array_cohort = []
    for i in list_img_cohort:
        list_array_cohort.append(i.get_fdata().reshape(-1))

    list_mode_cohort = []
    for i in list_array_cohort:
        hist, edges = np.histogram(i, bins)
        mode = get_mode(hist, edges, bins)
        list_mode_cohort.append(mode)
    mode_mean_cohort = mean(list_mode_cohort)
    print(mode_mean_cohort)

    mode_fit = mode_by_percentile[min(range(len(mode_by_percentile)), key = lambda i: abs(mode_by_percentile[i]-mode_mean_cohort))]
    print(mode_fit)
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


def bounding_crop(img_apply_to,
                  brainmask,
                  bbox1,
                  bbox2,
                  cdg_ijk):
    
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
    #fig.patch.set_facecolor('k')

    # Gestion de la transparence pour les images superposées
    alpha = 0.7

    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
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


def overlay_brainmask(img_ref, brainmask):

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
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
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
            ax[i,j].imshow(ndimage.rotate(sli_ref[count],90), 
                           cmap='gray')
            ax[i,j].imshow(ndimage.rotate(sli_brainmask[count],90), 
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




def metrics_prediction(array_img, threshold, cluster_filter):
    """Get metrics on array Nifti prediction file

    Args:
        array_img (array): array Nifti prediction file

    Returns:
        tuple: sum voxel segmented, mean size cluster, 
        median size cluster, min size cluster, max size cluster
    """

    if len(array_img.shape) > 3:
        array_img = array_img.squeeze()
    # Il faut bien faire un seuillage avant de calculer les clusters
    threshold_img = array_img > threshold
    cluster_img = get_clusters_and_filter_image(threshold_img, cluster_filter)
    number_of_cluster = cluster_img[4]
    size_cluster = list(np.unique(cluster_img[3], return_counts=True)[1])
    if len(size_cluster) != 1:
        del size_cluster[0]
    else:
        size_cluster[0] = 0
    sum_voxel_segmented = sum(size_cluster)
    mean_size_cluster = round(np.mean(size_cluster))
    median_size_cluster = float(median(size_cluster))
    min_size_cluster = min(size_cluster)
    max_size_cluster = max(size_cluster)

    return sum_voxel_segmented, number_of_cluster, mean_size_cluster, median_size_cluster, min_size_cluster, max_size_cluster


def get_mask_regions(img, list_labels_regions):

    import nibabel as nb
    import numpy as np

    pred_img_array = img.get_fdata()
    mask = np.isin(pred_img_array, list_labels_regions).astype(int)
    mask_regions = nb.Nifti1Image(mask, img.affine, img.header)

    return mask_regions

        
def make_report(img_normalized,
                brainmask,
                bbox1,
                bbox2,
                cdg_ijk,
                isocontour_slides_path_FLAIR_T1, 
                qc_overlay_brainmask_main,
                metrics_clusters_path,
                subject_id=None, 
                image_size=(160, 214, 176),
                resolution=(1.0, 1.0, 1.0),
                percentile=99,
                threshold=0.5,
                sum_workflow_path=None, 
                metrics_clusters_2_path=None,
                clusters_bg_pvs_path=None,
                predictions_latventricles_DWMH_path=None,
                swi='False'):
    """

    Args:
        histogram_intensity ():
        isocontour_slides ():
        metrics_clusters (pandas array):
    """
    from PIL import Image
    import base64

    if swi == 'True':
        modality = 'SWI'
        title_metrics_clusters = "Predictions results for Cerebral MicroBleeds (CMB)"
    else:
        modality = 'T1w'
        title_metrics_clusters = "Prediction results for PeriVascular Spaces (PVS)"

    try:
        with open(sum_workflow_path, 'rb') as f:
            image_data = f.read()
        sum_workflow_data = base64.b64encode(image_data).decode()
    except:
        sum_workflow_data = None

    histogram_intensity_path = save_histogram(img_normalized)
    with open(histogram_intensity_path, 'rb') as f:
        image_data = f.read()
    histogram_intensity_data = base64.b64encode(image_data).decode()

    with open(bbox1, 'r') as file:
        bbox1 = eval(file.readline().strip())
    with open(bbox2, 'r') as file:
        bbox2 = eval(file.readline().strip())
    with open(cdg_ijk, 'r') as file:
        cdg_ijk = eval(file.readline().strip())

    bounding_crop_path = bounding_crop(img_normalized,
                                       brainmask,
                                       bbox1,
                                       bbox2,
                                       cdg_ijk)
    with open(bounding_crop_path, 'rb') as f:
        image_data = f.read()
    bounding_crop_data = base64.b64encode(image_data).decode()

    try:
        with open(isocontour_slides_path_FLAIR_T1, 'rb') as f:
            image_data = f.read()
        isocontour_slides_path_FLAIR_T1 = base64.b64encode(image_data).decode()
         
    except:
        isocontour_slides_path_FLAIR_T1 = None
    try:
        with open(qc_overlay_brainmask_main, 'rb') as f:
            image_data = f.read()
        qc_overlay_brainmask_main = base64.b64encode(image_data).decode()
    except:
        qc_overlay_brainmask_main = None

    metrics_clusters_orig = pd.read_csv(metrics_clusters_path)
    metrics_clusters = metrics_clusters_orig[['Number of voxels', 'Number of clusters', 'Mean clusters size',
                                              'Median clusters size', 'Minimal clusters size', 'Maximal clusters size']].copy()
    cluster_filter = metrics_clusters_orig['Cluster Filter'].values[0]
    cluster_threshold = metrics_clusters_orig['Cluster Threshold'].values[0]
    columns = metrics_clusters.columns.tolist()
    if metrics_clusters_2_path:
        metrics_clusters_2_orig = pd.read_csv(metrics_clusters_2_path)
        metrics_clusters_2 = metrics_clusters_2_orig[['Number of voxels', 'Number of clusters',
                                                      'Mean clusters size', 'Median clusters size',
                                                      'Minimal clusters size', 'Maximal clusters size']].copy()
        clusters_threshold_2 = metrics_clusters_2_orig['Cluster Threshold'].values[0]
        clusters_filter_2 = metrics_clusters_2_orig['Cluster Filter'].values[0]
        columns_2 = metrics_clusters_2.columns.tolist()
    else:
        metrics_clusters_2 = None
        clusters_threshold_2 = None
        clusters_filter_2 = None
        columns_2 = None

    clusters_threshold_bg = None
    clusters_filter_bg = None
    if clusters_bg_pvs_path:
        clusters_bg_pvs_orig = pd.read_csv(clusters_bg_pvs_path)
        clusters_bg_pvs = clusters_bg_pvs_orig[['DWM num clusters', 'DWM num voxels', 
                                                'BG num clusters', 'BG num voxels', 
                                                'Total num clusters', 'Total num voxels']].copy()
        clusters_threshold_bg = clusters_bg_pvs_orig['Threshold'].values[0]
        clusters_filter_bg = clusters_bg_pvs_orig['Cluster filter DWM'].values[0]
        clusters_bg_pvs.columns = [col.replace('_', ' ') for col in clusters_bg_pvs.columns]
    else:
        clusters_bg_pvs = None
        clusters_threshold_bg = None
        clusters_filter_bg = None
        columns_bg = None

    predictions_latventricles_DWMH = None
    columns_latventricles = None
    clusters_threshold_latventricles = None
    if predictions_latventricles_DWMH_path:
        predictions_latventricles_DWMH_orig = pd.read_csv(predictions_latventricles_DWMH_path)
        predictions_latventricles_DWMH = predictions_latventricles_DWMH_orig[['DWMH clusters number', 'DWMH voxels number',
                                                                              'Lateral Ventricles clusters number', 'Lateral ventricles voxels number',
                                                                              'Total clusters number', 'Total voxels number']]
        clusters_threshold_latventricles = predictions_latventricles_DWMH_orig['Cluster Threshold'].values[0]
        columns_latventricles = predictions_latventricles_DWMH.columns.tolist()
    else:
        predictions_latventricles_DWMH = None
        columns_latventricles = None
        clusters_threshold_latventricles = None


    tm = Template(
            """<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Report</title>
                    <style>
                        .table {
                            display: table;
                            width: 100%;
                            border-collapse: collapse;
                        }

                        .row {
                            display: table-row;
                        }

                        .cell {
                            display: table-cell;
                            padding: 8px;
                            border: 1px solid black;
                        }
                        p {
                            font-size: 12px;
                            font-weight: normal;
                        }
                    </style>
                </head>
                <body>
                <h1>Results report {% if subject_id %} subject {{ subject_id }} {% endif %}</h1>
                <div class="table">
                    {% if metrics_clusters_2 is defined and metrics_clusters_2 is not none %}
                    <h2>Prediction results for White Matter Hyperintensities (WMH)</h2>
                    <table>
                        <tr class="row">
                            {% for col in columns_2 %}
                            <th class="cell">{{ col }}</th>
                            {% endfor %}
                        </tr>
                        <div class="row">
                        {% for _, row in metrics_clusters_2.iterrows() %}
                        <tr>
                            {% for col in columns_2 %}
                            <td class="cell">{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                        </div>
                    </table>
                    <p>Clusters Threshold : {{ clusters_threshold_2 }}</p>
                    <p>Clusters Filter : {{ clusters_filter_2 }}</p>
                    {% endif %}
                </div>
                <div class="table">
                    {% if predictions_latventricles_DWMH is defined and predictions_latventricles_DWMH is not none %}
                    <h2>Predictions results clusters WMH in Lateral Ventricles and DWMH</h2>
                    <table>
                        <tr class="row">
                            {% for col in columns_latventricles %}
                            <th class="cell">{{ col }}</th>
                            {% endfor %}
                        </tr>
                        <div class="row">
                        {% for _, row in predictions_latventricles_DWMH.iterrows() %}
                        <tr>
                            {% for col in columns_latventricles %}
                            <td class="cell">{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                        </div>
                    </table>
                    <p>Clusters Threshold : {{ clusters_threshold_latventricles }}</p>
                    {% endif %}
                </div>
                <div class="table">
                    <h2>{{ title_metrics_clusters }}</h2>
                    <table>
                        <tr class="row">
                            {% for col in columns %}
                            <th class ="cell">{{ col }}</th>
                            {% endfor %}
                        </tr>
                        <div class="row">
                        {% for _, row in metrics_clusters.iterrows() %}
                        <tr>
                            {% for col in columns %}
                            <td class="cell">{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                        </div>
                    </table>
                    <p>Clusters Threshold : {{ cluster_threshold }}</p>
                    <p>Clusters Filter : {{ cluster_filter  }}</p>
                </div>
                <div class="table">
                    {% if clusters_bg_pvs is defined and clusters_bg_pvs is not none %}
                    <h2>Predictions results clusters PVS in Basal Ganglia and DWM</h2>
                    <table>
                        <tr class="row">
                            {% for col in clusters_bg_pvs.columns %}
                            <th class="cell">{{ col }}</th>
                            {% endfor %}
                        </tr>
                        <div class="row">
                        {% for _, row in clusters_bg_pvs.iterrows() %}
                        <tr>
                            {% for col in clusters_bg_pvs.columns %}
                            <td class="cell">{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                        </div>
                    </table>
                    <p>Clusters Threshold : {{ clusters_threshold_bg }}</p>
                    <p>Clusters Filter DWM: {{ clusters_filter_bg }}</p>
                    {% endif %}
                </div>
                <div class="test">
                    <h1>Quality control</h1>
                    <h2>Preprocessed {{ modality }} image histogram</h2>
                    <p>Histogram of the {{ modality }} image that enters the classifier: {{ resolution }} mm<sup>3</sup> with {{ image_size }} shape, (within brain mask and with censoring for voxels outside the brain mask)</p>
                    <object type = 'image/svg+xml' data='data:image/svg+xml;base64, {{ hist_intensity }}' width="400" height="400"></object>
                    <h2>Crop box</h2>Display of the cropping region on the conformed image (256x256x256 at 1.0 mm<sup>3</sup> resolution).</p>
                    <object type = 'image/svg+xml' data='data:image/svg+xml;base64, {{ bounding_crop }}'></object>
                    {% if isocontour_slides_FLAIR_T1 %}
                    <h2>Isocontour Slides for coregistration FLAIR on T1w</h2>
                    <p>Isocontour of the FLAIR image coregister on T1w image that enters the classifier: {{ resolution }} mm<sup>3</sup> with {{ image_size }}, (within brain mask and with censoring for voxels outside the brain mask)</p>
                    <img src = 'data:image/png;base64, {{ isocontour_slides_FLAIR_T1 }}' width="800" "height="400"></img>
                    {% endif %}
                    {% if qc_overlay_brainmask_main %}
                    <h2>Overlay of final brainmask over cropped {{ modality }}</h2>
                    <p>Overlay of the brainmask on {{ modality }} image : {{ resolution }} mm<sup>3</sup> with {{ image_size }}, (with censoring for voxels outside the brain mask)</p>
                    <img src = 'data:image/png;base64, {{ qc_overlay_brainmask_main }}' width="800" "height="400"></img>
                    {% endif %}
                </div>
                {% if sum_workflow %}
                <h2>Preprocessing workflow diagram<h2>
                {% endif %}
                <p>Parameters : </p>
                <p>value percentile : {{ percentile }}</p>
                <p>value threshold : {{ threshold }}</p>
                <p>final resolution : {{ resolution }} mm<sup>3</p>
                <p>final dimensions crop : {{ image_size }}<p>
                {% if sum_workflow %}
                <object type = 'image/svg+xml' data='data:image/svg+xml;base64, {{ sum_workflow }}' width="1000" height="600"></object>
                {% endif %}
                </body>
                </html>"""
                )

    template_report = tm.render(subject_id=subject_id,
                                hist_intensity=histogram_intensity_data, 
                                bounding_crop=bounding_crop_data,
                                isocontour_slides_FLAIR_T1=isocontour_slides_path_FLAIR_T1, 
                                qc_overlay_brainmask_main=qc_overlay_brainmask_main,
                                sum_workflow=sum_workflow_data,
                                metrics_clusters= metrics_clusters,
                                columns=columns,
                                cluster_filter=cluster_filter,
                                cluster_threshold=cluster_threshold,
                                columns_2=columns_2,
                                metrics_clusters_2=metrics_clusters_2,
                                clusters_threshold_2=clusters_threshold_2,
                                clusters_filter_2=clusters_filter_2,
                                clusters_bg_pvs=clusters_bg_pvs,
                                clusters_threshold_bg=clusters_threshold_bg,
                                clusters_filter_bg=clusters_filter_bg,
                                predictions_latventricles_DWMH=predictions_latventricles_DWMH,
                                columns_latventricles=columns_latventricles,
                                clusters_threshold_latventricles=clusters_threshold_latventricles,
                                percentile=percentile,
                                threshold=threshold,
                                image_size=image_size,
                                resolution=resolution,
                                modality=modality,
                                title_metrics_clusters=title_metrics_clusters)

    return template_report