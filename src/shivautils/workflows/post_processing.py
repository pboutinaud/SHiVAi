#!/usr/bin/env python
"""Nipype workflow for post-processing.

    Required connections to set outside of the workflow:

    (summary_report, 'subject_id')
    (summary_report, 'brainmask')
    (summary_report, 'wf_graph')
    (summary_report, 'crop_brain_img')
    (summary_report, 'overlayed_brainmask_1')
    
    if with_swi and with_t1:
        (summary_report, 'overlayed_brainmask_2')
    if with_flair:
        (summary_report, 'isocontour_slides_FLAIR_T1')

    if PVS:
        (prediction_metrics_pvs, 'img')
        (prediction_metrics_pvs, 'brain_seg') can be brainmask or synthseg
    if WMH:
        (prediction_metrics_wmh, 'img')
        (prediction_metrics_wmh, 'brain_seg') can be brainmask or synthseg
    if CMB:
        (prediction_metrics_cmb, 'img')
        (prediction_metrics_cmb, 'brain_seg') can be brainmask or synthseg
    if LAC:
        (prediction_metrics_lac, 'img')
        (prediction_metrics_lac, 'brain_seg') can be brainmask or synthseg

   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces import ants

from shivautils.interfaces.post import SummaryReport
from shivautils.interfaces.image import Regionwise_Prediction_metrics
from shivautils.utils.misc import set_wf_shapers


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def get_maps_from_dict(subject_id,
                       preproc_dict,
                       pvs_pred_dict=None,
                       wmh_pred_dict=None,
                       cmb_pred_dict=None,
                       lac_pred_dict=None,
                       wf_graph=None,  # Placeholder
                       brain_seg=None  # SynthSeg
                       ):
    if pvs_pred_dict is not None:
        segmentation_pvs = pvs_pred_dict[subject_id]
    else:
        segmentation_pvs = None
    if wmh_pred_dict is not None:
        segmentation_wmh = wmh_pred_dict[subject_id]
    else:
        segmentation_wmh = None
    if cmb_pred_dict is not None:
        segmentation_cmb = cmb_pred_dict[subject_id]
    else:
        segmentation_cmb = None
    if lac_pred_dict is not None:
        segmentation_lac = lac_pred_dict[subject_id]
    else:
        segmentation_lac = None
    T1_cropped = preproc_dict[subject_id]['T1_cropped']
    brainmask = preproc_dict[subject_id]['brainmask']
    pre_brainmask = preproc_dict[subject_id]['pre_brainmask']
    T1_conform = preproc_dict[subject_id]['T1_conform']
    bbox1 = preproc_dict[subject_id]['bbox1']
    bbox2 = preproc_dict[subject_id]['bbox2']
    cdg_ijk = preproc_dict[subject_id]['cdg_ijk']
    FLAIR_cropped = preproc_dict[subject_id]['FLAIR_cropped']
    SWI_cropped = preproc_dict[subject_id]['SWI_cropped']
    return (segmentation_pvs, segmentation_wmh, segmentation_cmb, segmentation_lac, T1_cropped, brainmask, pre_brainmask,
            T1_conform, bbox1, bbox2, cdg_ijk, wf_graph,
            FLAIR_cropped, SWI_cropped, brain_seg)


def genWorkflow(**kwargs) -> Workflow:
    """
    Generate a nipype workflow to produce statistics and reports for the segmentations

    Probably not adapted for SWOMed right now

    Returns:
        workflow
    """
    # Setting up the different cases to build the workflows (should clarify things)
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs['PREDICTION'])

    name_workflow = "post_processing_workflow"
    workflow = Workflow(name_workflow)
    workflow.base_dir = kwargs['BASE_DIR']

    # Preparing stats and figures for the report
    # Segmentation part
    preds = []
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:  # WARN: None of this is SWOMed compatible
        preds.append('PVS')
        prediction_metrics_pvs = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_pvs")
        prediction_metrics_pvs.inputs.biomarker_type = 'pvs'
        prediction_metrics_pvs.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_pvs.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1  # "- 1 because thr removes up to given value"
        # TODO: This is when using only brainmask, we need synthseg for BG
        # if not synthseg:
        prediction_metrics_pvs.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_pvs.inputs.region_list = ['Whole_brain']
        # else:
        # prediction_metrics_pvs.inputs.brain_seg_type = 'synthseg'
        # prediction_metrics_pvs.inputs.region_list = ['Whole_brain', 'Basal_ganglia']

    if 'WMH' in kwargs['PREDICTION']:
        preds.append('WMH')
        prediction_metrics_wmh = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_wmh")
        prediction_metrics_wmh.inputs.biomarker_type = 'wmh'
        prediction_metrics_wmh.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_wmh.inputs.thr_cluster_size = kwargs['MIN_WMH_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_wmh.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_wmh.inputs.region_list = ['Whole_brain']

    if 'LAC' in kwargs['PREDICTION']:
        preds.append('LAC')
        prediction_metrics_lac = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_lac")
        prediction_metrics_lac.inputs.biomarker_type = 'lac'
        prediction_metrics_lac.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_lac.inputs.thr_cluster_size = kwargs['MIN_LAC_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_lac.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_lac.inputs.region_list = ['Whole_brain']

    if 'CMB' in kwargs['PREDICTION']:
        preds.append('CMB')
        prediction_metrics_cmb = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_cmb")
        prediction_metrics_cmb.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_cmb.inputs.thr_cluster_size = kwargs['MIN_CMB_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_cmb.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_cmb.inputs.region_list = ['Whole_brain']
        if with_t1:  # The metrics are computed on the segmentation put in T1 space, for coherence
            swi_pred_to_t1 = Node(ants.ApplyTransforms(), name="swi_pred_to_t1")
            swi_pred_to_t1.inputs.out_postfix = '_t1-space'
            workflow.connect(swi_pred_to_t1, 'output_image', prediction_metrics_cmb, 'img')
            prediction_metrics_cmb.inputs.biomarker_type = 'cmb_t1-space'
        else:
            prediction_metrics_cmb.inputs.biomarker_type = 'cmb'

    # QC part
    # Initialising the QC sub-workflow
    # qc_wf = gen_qc_wf('preproc_qc_workflow')
    # if with_flair:  # dual predictions
    #     qc_wf = qc_wf_add_flair(qc_wf)
    # if with_swi and with_t1:
    #     qc_wf = qc_wf_add_swi(qc_wf)
    # workflow.add_nodes([qc_wf])

    # Building the actual report (html then pdf)
    summary_report = Node(SummaryReport(), name="summary_report")
    # Segmentation section
    if 'PVS' in preds:
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', summary_report, 'pvs_metrics_csv')
        workflow.connect(prediction_metrics_pvs, 'biomarker_census_csv', summary_report, 'pvs_census_csv')
    if 'WMH' in preds:
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', summary_report, 'wmh_metrics_csv')
        workflow.connect(prediction_metrics_wmh, 'biomarker_census_csv', summary_report, 'wmh_census_csv')
    if 'CMB' in preds:
        workflow.connect(prediction_metrics_cmb, 'biomarker_stats_csv', summary_report, 'cmb_metrics_csv')
        workflow.connect(prediction_metrics_cmb, 'biomarker_census_csv', summary_report, 'cmb_census_csv')
    if 'LAC' in preds:
        workflow.connect(prediction_metrics_lac, 'biomarker_stats_csv', summary_report, 'lac_metrics_csv')
        workflow.connect(prediction_metrics_lac, 'biomarker_census_csv', summary_report, 'lac_census_csv')

    # QC section
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']
    summary_report.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
    summary_report.inputs.min_seg_size = {
        'PVS': kwargs['MIN_PVS_SIZE'],
        'WMH': kwargs['MIN_WMH_SIZE'],
        'CMB': kwargs['MIN_CMB_SIZE'],
        'LAC': kwargs['MIN_LAC_SIZE']}
    summary_report.inputs.pred_list = preds

    # workflow.connect(qc_wf, 'qc_crop_box.crop_brain_img', summary_report, 'crop_brain_img')
    # workflow.connect(qc_wf, 'qc_overlay_brainmask.overlayed_brainmask', summary_report, 'overlayed_brainmask_1')
    # if with_swi and with_t1:
    #     workflow.connect(qc_wf, 'qc_overlay_brainmask_swi.overlayed_brainmask', summary_report, 'overlayed_brainmask_2')
    # if with_flair:
    #     workflow.connect(qc_wf, 'qc_coreg_FLAIR_T1.qc_coreg', summary_report, 'isocontour_slides_FLAIR_T1')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
