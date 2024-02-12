# from shivautils.utils.stats import save_histogram, bounding_crop
from shivautils import __version__ as version
from jinja2 import Environment, PackageLoader
import base64
import os
from shivautils.postprocessing import __file__ as postproc_init


def make_report(
        pred_metrics_dict: dict,
        pred_census_im_dict: dict,
        pred_overlay_im_dict: dict,
        pred_and_acq: dict,
        brain_vol_vox: float,
        thr_cluster_vals: float,
        min_seg_size: dict,
        models_uid: dict,
        bounding_crop: str = None,
        overlayed_brainmask_1: str = None,
        overlayed_brainmask_2: str = None,
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
        pred_overlay_im_dict (dic): Dict of the image path to the overlau showing biomarkers on the brain
        brain_vol (float): Intracranial brain volume in voxels
        pred_and_acq (dict):
        brain_vol_vox (float):
        thr_cluster_vals (dict): Thresholds applied to raw predictions to binarise them
        min_seg_size (dict): Dict holding the minimal size used to filter each type of biomarker segmentation 
        models_uid (dict): Dict with keys = predictions, and values = dict with keys = 'url' and 'id'. 'url' is the url to the pred model files
                           and 'id' is a dict with keys = generic model file id, and vales = tuple(filename, md5)
        bounding_crop (path): PNG file showing the crop box.
        overlayed_brainmask_1 (path): PNG file of cropping box with overlay brainmask
        overlayed_brainmask_2 (path): PNG file of cropping box with overlay brainmask (for SWI if done at the same time as non-CMB predictions)
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
        'PVS': 'PeriVascular Spaces',
        'WMH': 'White-Matter Hyperintensities',
        'CMB': 'Cerebral MicroBleeds',
        'LAC': 'Lacunas'
    }
    vol_mm3_per_voxel = resolution[0] * resolution[1] * resolution[2]  # Should be 1.0 mm3 by default
    brain_vol = brain_vol_vox * vol_mm3_per_voxel / 1000  # in cm3
    pred_stat_dict = {}
    for seg, stat_df in pred_metrics_dict.items():
        metrics = ['Region',
                   f'Number of {seg}',
                   'Total volume (mm<sup>3</sup>)',
                   'Mean volume (mm<sup>3</sup>)',
                   'Median volume (mm<sup>3</sup>)',
                   'Standard Dev. (mm<sup>3</sup>)',
                   'Min volume (mm<sup>3</sup>)',
                   'Max volume (mm<sup>3</sup>)',]
        col_maper = {col: metric for col, metric in zip(stat_df.columns, metrics)}
        stat_df.rename(col_maper, axis=1, inplace=True)
        # Putting the voxel results in mm3 and in str with 2 decimal floats
        mm3_cols = metrics[2:]
        for col in mm3_cols:
            stat_df[col] = (stat_df[col] * vol_mm3_per_voxel).map('{:.2f}'.format)
        # stat_df.set_index('Region', inplace=True)
        stat_df_html = stat_df.to_html(justify='center', escape=False, index=False)

        # quick fix to a weird formatting
        stat_df_html = stat_df_html.replace('table border="1"', 'table')

        # Some finer formatting
        stat_df_html_list = stat_df_html.splitlines()
        row_ind = [i for i, line in enumerate(stat_df_html_list) if '<tr>' in line]
        for ind in row_ind:
            # getting the first col as <th> to have it bold
            stat_df_html_list[ind + 1] = stat_df_html_list[ind + 1].replace('td', 'th')
            # Finding rows with 0 biomarkers and setting a special tr class to format with css
            if '<td>0</td>' in stat_df_html_list[ind + 2]:
                stat_df_html_list[ind] = stat_df_html_list[ind].replace('<tr>', '<tr class="empty_reg">')
                # Replace the NaN with empty cells
                ind2 = ind + 2
                while '</tr>' not in stat_df_html_list[ind2]:
                    stat_df_html_list[ind2] = stat_df_html_list[ind2].replace('<td>NaN</td>', '<td></td>')
                    stat_df_html_list[ind2] = stat_df_html_list[ind2].replace('<td>nan</td>', '<td></td>')
                    ind2 += 1
        stat_df_html = '\n'.join(stat_df_html_list)

        with open(pred_census_im_dict[seg], 'rb') as f:
            image_data = f.read()
        pred_census_fig = base64.b64encode(image_data).decode()

        with open(pred_overlay_im_dict[seg], 'rb') as f:
            image_data = f.read()
        pred_overlay_fig = base64.b64encode(image_data).decode()

        pred_stat_dict[seg] = {'title': f'Brain charge statistics for {seg_full_name[seg]} ({seg})',
                               'metrics_table': stat_df_html,
                               'brain_volume': brain_vol,
                               'census_figure': pred_census_fig,
                               'overlay_figure': pred_overlay_fig,
                               'acquisitions': pred_and_acq[seg],
                               'cluster_threshold': thr_cluster_vals[seg],
                               'cluster_min_vol': min_seg_size[seg],
                               }

    if 'CMB' in pred_metrics_dict.keys() and len(pred_metrics_dict.keys()) == 1:
        modality = 'SWI'
    else:  # TODO : make this more adaptative
        modality = 'T1w'

    # Conversion of images in base64 objects
    if overlayed_brainmask_1 is not None:
        with open(overlayed_brainmask_1, 'rb') as f:
            image_data = f.read()
        overlayed_brainmask_1 = base64.b64encode(image_data).decode()

    if overlayed_brainmask_2 is not None:
        with open(overlayed_brainmask_2, 'rb') as f:
            image_data = f.read()
        overlayed_brainmask_2 = base64.b64encode(image_data).decode()

    if bounding_crop is not None:
        with open(bounding_crop, 'rb') as f:
            image_data = f.read()
        bounding_crop = base64.b64encode(image_data).decode()

    if wf_graph is not None:
        with open(wf_graph, 'rb') as f:
            image_data = f.read()
        wf_graph = base64.b64encode(image_data).decode()

    if isocontour_slides_FLAIR_T1:
        with open(isocontour_slides_FLAIR_T1, 'rb') as f:
            image_data = f.read()
        isocontour_slides_FLAIR_T1 = base64.b64encode(image_data).decode()

    env = Environment(loader=PackageLoader('shivautils', 'postprocessing'))
    tm = env.get_template('report_template.html')

    filled_template_report = tm.render(
        data_origin=subject_id,
        pred_stat_dict=pred_stat_dict,
        pred_and_acq=pred_and_acq,
        models_uid=models_uid,
        overlayed_brainmask_1=overlayed_brainmask_1,
        overlayed_brainmask_2=overlayed_brainmask_2,
        bounding_crop=bounding_crop,
        modality=modality,
        image_size=image_size,
        resolution=resolution,
        isocontour_slides_FLAIR_T1=isocontour_slides_FLAIR_T1,
        wf_graph=wf_graph,
        percentile=percentile,
        threshold=threshold,
        version=version
    )

    return filled_template_report
