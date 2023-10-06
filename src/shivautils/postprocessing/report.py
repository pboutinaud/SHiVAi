import nibabel as nb
import pandas as pd
from shivautils.stats import save_histogram, bounding_crop
from jinja2 import Template


def make_report(img_normalized: nb.Nifti1Image,
                brainmask: nb.Nifti1Image,
                bbox1: tuple,
                bbox2: tuple,
                cdg_ijk: tuple,
                isocontour_slides_path_FLAIR_T1: str,
                qc_overlay_brainmask_t1: str,
                metrics_clusters_path: str,
                subject_id: int = None,
                image_size: tuple = (160, 214, 176),
                resolution: tuple = (1.0, 1.0, 1.0),
                percentile: int = 99,
                threshold: float = 0.5,
                sum_workflow_path: str = None,
                metrics_clusters_2_path: str = None,
                clusters_bg_pvs_path: str = None,
                predictions_latventricles_DWMH_path: str = None,
                swi: bool = 'False'):
    """
    Individual HTML report:

    - Summary of metrics clusters per subject
    - Histogram of voxel intensity during t1 normalization
    - Display of the cropping region on the conformed image
    - T1 on FLAIR isocontour slides 
    - Overlay of final brainmask over cropped t1 images
    - Preprocessing workflow diagram

    Args:
        img_normalized (nb.Nifti1Image): t1 nifti file to compute histogram voxels intensity
        brainmask (nb.Nifti1Image): brainmask nifti file 
        bbox1 (tuple): first coordoninates point of cropping box
        bbox2 (tuple): second coordonaites point of cropping box
        cdg_ijk (tuple): t1 nifti image center of mass used to calculate cropping box
        isocontour_slides_path_FLAIR_T1 (path): PNG file with the reference image in the background and the edges of the given image on top
        qc_overlay_brainmask_t1 (path): svg file of cropping box with overlay brainmask
        metrics_clusters_path (str): csv file about predictions results and clusters metrics (for PVS or CMB)
        subject_id (int): identified number of subject report
        image_size (tuple): Final image array size in i, j, k in tuple 
        resolution (tuple): Voxel size of the final image in tuple
        percentile (int): value to threshold above this percentile
        threshold (float): Value of the treshold to apply to the image for brainmask
        sum_worflow (path): summary of preprocessing step in a svg file
        metrics_clusters_2_path (path): csv file about predictions results and clusters metrics (for WMH)
        clusters_bg_pvs_path (path): csv file about predictions results in basal ganglia and deep white matter
        predictions_latventricles_DWMH_path (path): csv file about predictions results in lateral ventricles and deep white matter hyperintensities
        swi (bool): boolean value to indicate if the report is about cmb clusters predictions or not

    Returns:
        html file with completed report
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
        with open(qc_overlay_brainmask_t1, 'rb') as f:
            image_data = f.read()
        qc_overlay_brainmask_t1 = base64.b64encode(image_data).decode()
    except:
        qc_overlay_brainmask_t1 = None

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
                    <object type = 'image/svg+xml' data='data:image/svg+xml;base64, {{ bounding_crop }}' width="600" height="400"></object>
                    {% if isocontour_slides_FLAIR_T1 %}
                    <h2>Isocontour Slides for coregistration FLAIR on T1w</h2>
                    <p>Isocontour of the FLAIR image coregister on T1w image that enters the classifier: {{ resolution }} mm<sup>3</sup> with {{ image_size }}, (within brain mask and with censoring for voxels outside the brain mask)</p>
                    <img src = 'data:image/png;base64, {{ isocontour_slides_FLAIR_T1 }}' width="600" "height="300"></img>
                    {% endif %}
                    {% if qc_overlay_brainmask_t1 %}
                    <h2>Overlay of final brainmask over cropped {{ modality }}</h2>
                    <p>Overlay of the brainmask on {{ modality }} image : {{ resolution }} mm<sup>3</sup> with {{ image_size }}, (with censoring for voxels outside the brain mask)</p>
                    <img src = 'data:image/png;base64, {{ qc_overlay_brainmask_t1 }}' width="600" "height="300"></img>
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
                <object type = 'image/svg+xml' data='data:image/svg+xml;base64, {{ sum_workflow }}' width="700" height="500"></object>
                {% endif %}
                </body>
                </html>"""
    )

    template_report = tm.render(subject_id=subject_id,
                                hist_intensity=histogram_intensity_data,
                                bounding_crop=bounding_crop_data,
                                isocontour_slides_FLAIR_T1=isocontour_slides_path_FLAIR_T1,
                                qc_overlay_brainmask_t1=qc_overlay_brainmask_t1,
                                sum_workflow=sum_workflow_data,
                                metrics_clusters=metrics_clusters,
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
