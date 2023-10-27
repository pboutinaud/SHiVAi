# from shivautils.stats import save_histogram, bounding_crop
from jinja2 import Environment, PackageLoader
import base64


def make_report(
        pred_metrics_dict: dict,
        pred_census_im_dict: dict,
        brain_vol: float,
        thr_cluster_val: float,
        min_seg_size: dict,
        bounding_crop_path: str,
        qc_overlay_brainmask_t1: str = None,
        isocontour_slides_FLAIR_T1: str = None,
        subject_id: int = None,
        image_size: tuple = (160, 214, 176),
        resolution: tuple = (1.0, 1.0, 1.0),
        percentile: int = 99,
        threshold: float = 0.5,
        wf_graph: str = None):
    """
    Individual HTML report:

    - Summary of segmentation metrics per subject
    - Swarmplot of the size of each segmented biomarker
    - Display of the cropping region on the conformed image
    - T1 on FLAIR isocontour slides 
    - Overlay of final brainmask over cropped t1 images
    - Processing workflow diagram

    Args:
        pred_metrics_dict (dict): Dict of the dataframes holding statistics for each studied biomaerker (keys)
        pred_census_im_dict (dic): Dict of the image path to the swarmplot showing each biomarker size repartition
        brain_vol (float): Intracranial brain volume
        thr_cluster_val (float): Threshold applied to raw predictions to binarise them
        min_seg_size (dict): Dict holding the minimal size used to filter each type of biomarker segmentation 
        bounding_crop_path (path): PNG file showing the crop box.
        qc_overlay_brainmask_t1 (path): SVG file of cropping box with overlay brainmask
        isocontour_slides_FLAIR_T1 (path): PNG file with the reference image in the background and the edges of the given image on top
        subject_id (int): Participant identificator
        image_size (tuple): Final image dimensions
        resolution (tuple): Voxel size of the final image
        percentile (int): Range of values (in percentile) kept during intensity normalisation
        threshold (float): Treshold applied to binarise the brainmask
        wf_graph (path): graph of the workflow in an svg file

    Returns:
        html file with completed report
    """

    # Preparing of prediction tables and stats in html
    seg_full_name = {
        'PVS': 'Perivasculaire spaces',
        'WMH': 'White-matter hyperintensities',
        'CMB': 'Cerebral microbleeds'
    }
    vol_mm3_per_voxel = resolution[0] * resolution[1] * resolution[2]  # Should be 1.0 mm3 by default
    brain_vol *= vol_mm3_per_voxel
    pred_stat_dict = {}
    for seg, stat_df in pred_metrics_dict.items():
        metrics = ['Region',
                   f'Number of {seg}',
                   f'Total volume of all {seg} (mm<sup>3</sup>)',
                   'Mean volume (mm<sup>3</sup>)',
                   'Median volume (mm<sup>3</sup>)',
                   'StD of the volume (mm<sup>3</sup>)',
                   'Min volume (mm<sup>3</sup>)',
                   'Max volume (mm<sup>3</sup>)',]
        col_maper = {col: metric for col, metric in zip(stat_df.columns, metrics)}
        stat_df.rename(col_maper, axis=1, inplace=True)
        stat_df[f'Total volume of all {seg} (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df['Mean volume (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df['Median volume (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df['StD of the volume (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df['Min volume (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df['Max volume (mm<sup>3</sup>)'] *= vol_mm3_per_voxel
        stat_df.set_index('Region', inplace=True)
        stat_df_html = stat_df.to_html(justify='center', escape=False)

        with open(pred_census_im_dict[seg], 'rb') as f:
            image_data = f.read()
        pred_census_fig = base64.b64encode(image_data).decode()

        pred_stat_dict[seg] = {'title': f'Brain charge statistics for {seg_full_name[seg]} ({seg})',
                               'metrics_table': stat_df_html,
                               'brain_volume': brain_vol,
                               'cluster_threshold': thr_cluster_val,
                               'cluster_min_vol': min_seg_size[seg],
                               'census_figure': pred_census_fig
                               }

    if 'CMB' in pred_metrics_dict.keys() and len(pred_metrics_dict.keys()) == 1:
        modality = 'SWI'
    else:  # TODO : make this more adaptative
        modality = 'T1w'

    # Conversion of images in base64 objects
    if qc_overlay_brainmask_t1 is not None:
        with open(qc_overlay_brainmask_t1, 'rb') as f:
            image_data = f.read()
        qc_overlay_brainmask_t1 = base64.b64encode(image_data).decode()

    with open(bounding_crop_path, 'rb') as f:
        image_data = f.read()
    bounding_crop = base64.b64encode(image_data).decode()

    if wf_graph is not None:
        with open(wf_graph, 'rb') as f:
            image_data = f.read()
        wf_graph = base64.b64encode(image_data).decode()

    if isocontour_slides_FLAIR_T1 is not None:
        with open(isocontour_slides_FLAIR_T1, 'rb') as f:
            image_data = f.read()
        isocontour_slides_FLAIR_T1 = base64.b64encode(image_data).decode()

    env = Environment(loader=PackageLoader('shivautils', 'postprocessing'))
    tm = env.get_template('report_template.html')

    filled_template_report = tm.render(
        data_origin=subject_id,
        pred_stat_dict=pred_stat_dict,
        qc_overlay_brainmask_t1=qc_overlay_brainmask_t1,
        bounding_crop=bounding_crop,
        modality=modality,
        image_size=image_size,
        resolution=resolution,
        isocontour_slides_FLAIR_T1=isocontour_slides_FLAIR_T1,
        wf_graph=wf_graph,
        percentile=percentile,
        threshold=threshold,
    )

    return filled_template_report
