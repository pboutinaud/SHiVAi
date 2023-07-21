from statistics import mean
import numpy as np
import nibabel as nb
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from jinja2 import Template

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
