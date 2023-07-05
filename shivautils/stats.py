from statistics import mean
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from jinja2 import Template
import base64
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
import os
import csv
from statistics import median

from shivautils.metrics import get_clusters_and_filter_image
from shivautils.quantification_WMH_Ventricals_Maps import create_distance_map

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


def save_histogram(img_normalized: str,
                   bins=100):
    
    import nibabel as nb
    import matplotlib.pyplot as plt
    import numpy as np
    import os.path as op

    img = nb.load(img_normalized)
    array = img.get_fdata()
    x = array.reshape(-1)
    hist, edges = np.histogram(x, bins=bins)

    fig, ax = plt.subplots()
    ax.hist(x, bins=edges, color=(0.3, 0.5, 0.8))
    ax.set_yscale('log')
    ax.set_title("Histogram of intensities voxel values")
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("Number of Voxels")

    histogram = 'hist.png' 
    plt.savefig('hist.png')
    return (op.abspath(histogram))



def metrics_prediction(array_img):
    """Get metrics on array Nifti prediction file

    Args:
        array_img (array): array Nifti prediction file

    Returns:
        tuple: sum voxel segmented, mean size cluster, 
        median size cluster, min size cluster, max size cluster
    """

    if len(array_img.shape) > 3:
        array_img = array_img.squeeze()
    cluster_img = get_clusters_and_filter_image(array_img)
    number_of_cluster = cluster_img[2]
    size_cluster = list(np.unique(cluster_img[1], return_counts=True)[1])
    if len(size_cluster) != 1:
        del size_cluster[0]
    else:
        size_cluster[0] = 0
    sum_voxel_segmented = sum(size_cluster)
    mean_size_cluster = np.mean(size_cluster)
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

#img = "/homes_unix/yrio/Documents/test_report_fonctionnalities/_subject_id_subject_21/synthseg/segmentation_regions.nii.gz"
#list_labels_regions = [4, 5, 43, 44]
#mask_regions(img, list_labels_regions)

        
def make_report(histogram_intensity_path, isocontour_slides_path, metrics_clusters_path, metrics_clusters_2_path=None):
    """

    Args:
        histogram_intensity ():
        isocontour_slides ():
        metrics_clusters (pandas array):
    """
    try:
        from PIL import Image
        image = Image.open(isocontour_slides_path)

    except:
        isocontour_slides_path = None

    metrics_clusters = pd.read_csv(metrics_clusters_path)
    columns = metrics_clusters.columns.tolist()

    if metrics_clusters_2_path:
        metrics_clusters_2 = pd.read_csv(metrics_clusters_2_path)
        columns = metrics_clusters_2.columns.tolist()
    else:
        metrics_clusters_2 = None

    tm = Template(
            """<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Report</title>
                </head>
                <body>
                <div class="test">
                    <h1>Report Summary</h1>
                    <h2>T1 final intensity normalization step</h2>
                    <img src = {{ hist_intensity }} width="400" height="400"></img>
                    {% if isocontour_slides %}
                    <h2>Isocontour Slides for coregistration FLAIR on T1</h2>
                    <img src = {{ isocontour_slides }} width="800" "height="400"></img>
                    {% endif %}
                </div>
                <div>
                    <h2>Metrics Prediction clusters PVS</h2>
                    <table>
                        <tr>
                            {% for col in columns %}
                            <th>{{ col }}</th>
                            {% endfor %}
                        </tr>
                        {% for _, row in metrics_clusters.iterrows() %}
                        <tr>
                            {% for col in columns %}
                            <td>{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                <div>
                    {% if metrics_clusters_2 is defined and metrics_clusters_2 is not none %}
                    <h2>Metrics Prediction clusters WMH</h2>
                    <table>
                        <tr>
                            {% for col in columns %}
                            <th>{{ col }}</th>
                            {% endfor %}
                        </tr>
                        {% for _, row in metrics_clusters_2.iterrows() %}
                        <tr>
                            {% for col in columns %}
                            <td>{{ row[col] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                    {% endif %}
                </div>
                </body>
                </html>"""
                )

    template_report = tm.render(hist_intensity=histogram_intensity_path, 
                                isocontour_slides=isocontour_slides_path, 
                                metrics_clusters= metrics_clusters,
                                columns=columns,
                                metrics_clusters_2=metrics_clusters_2)

    return template_report