#!/usr/bin/env python
"""Nipype workflow for post-processing.

Required connections to set outside of the workflow:

(summary_report, 'subject_id')
(qc_crop_box, 'img_apply_to')
(qc_crop_box, 'brainmask')
(qc_crop_box, 'bbox1')
(qc_crop_box, 'bbox2')
(qc_crop_box, 'cdg_ijk')
(qc_overlay_brainmask, 'brainmask')
(qc_overlay_brainmask, 'img_ref')
(summary_report, 'brainmask')
(summary_report, 'wf_graph')

if dual:
        (qc_coreg_FLAIR_T1, 'path_image')
        (qc_coreg_FLAIR_T1, 'path_ref_image')
        (qc_coreg_FLAIR_T1, 'path_brainmask')


if PVS:
    (prediction_metrics_pvs, 'img')
    (prediction_metrics_pvs, 'brain_seg') can be brainmask or synthseg
if WMH:
    (prediction_metrics_wmh, 'img')
    (prediction_metrics_wmh, 'brain_seg') can be brainmask or synthseg
if CMB:
    (prediction_metrics_cmb, 'img')
    (prediction_metrics_cmb, 'brain_seg') can be brainmask or synthseg


   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.image import (Apply_mask, Regionwise_Prediction_metrics,
                                         Join_Prediction_metrics, SummaryReport)
from shivautils.postprocessing.isocontour import create_edges
from shivautils.stats import overlay_brainmask, bounding_crop


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def as_list(input):
    return [input]


def get_maps_from_dict(subject_id,
                       preproc_dict,
                       pvs_pred_dict=None,
                       wmh_pred_dict=None,
                       cmb_pred_dict=None,
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
    T1_cropped = preproc_dict[subject_id]['T1_cropped']
    brainmask = preproc_dict[subject_id]['brainmask']
    pre_brainmask = preproc_dict[subject_id]['pre_brainmask']
    T1_conform = preproc_dict[subject_id]['T1_conform']
    bbox1 = preproc_dict[subject_id]['bbox1']
    bbox2 = preproc_dict[subject_id]['bbox2']
    cdg_ijk = preproc_dict[subject_id]['cdg_ijk']
    FLAIR_cropped = preproc_dict[subject_id]['FLAIR_cropped']
    SWI_cropped = preproc_dict[subject_id]['SWI_cropped']
    return (segmentation_pvs, segmentation_wmh, segmentation_cmb, T1_cropped, brainmask, pre_brainmask,
            T1_conform, bbox1, bbox2, cdg_ijk, wf_graph,
            FLAIR_cropped, SWI_cropped, brain_seg)


def genWorkflow(**kwargs) -> Workflow:
    """
    Generate a nipype workflow to produce statistics and reports for the segmentations

    Probably not adapted for SWOMed right now

    Returns:
        workflow
    """
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
        prediction_metrics_pvs.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_pvs.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1  # "- 1 because thr removes up to given value"
        # TODO: This is when using only brainmask, we need synthseg for BG
        # if not synthseg:
        prediction_metrics_pvs.inputs.region_list = ['Whole_brain']
        # else:
        # prediction_metrics_pvs.inputs.region_list = ['Whole_brain', 'Basal_ganglia']
        prediction_metrics_pvs_generale = JoinNode(Join_Prediction_metrics(),
                                                   joinsource='subject_list',
                                                   joinfield='csv_files',
                                                   name="prediction_metrics_pvs_generale")
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', prediction_metrics_pvs_generale, 'csv_files')

    if 'WMH' in kwargs['PREDICTION']:
        preds.append('WMH')
        prediction_metrics_wmh = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_wmh")
        prediction_metrics_wmh.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_wmh.inputs.thr_cluster_size = kwargs['MIN_WMH_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_wmh.inputs.region_list = ['Whole_brain']
        prediction_metrics_wmh_generale = JoinNode(Join_Prediction_metrics(),
                                                   joinsource='subject_list',
                                                   joinfield='csv_files',
                                                   name="prediction_metrics_wmh_generale")
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', prediction_metrics_wmh_generale, 'csv_files')

    # if 'CMB' in kwargs['PREDICTION']:  # TODO
    #     preds.append('CMB')
    #     prediction_metrics_cmb = Node(Regionwise_Prediction_metrics(),
    #                                   name="prediction_metrics_cmb")
    #     prediction_metrics_cmb.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
    #     prediction_metrics_cmb.inputs.thr_cluster_size = kwargs['MIN_CMB_SIZE'] - 1
    #     # if not synthseg:
    #     prediction_metrics_cmb.inputs.region_list = ['Whole_brain']
    #     prediction_metrics_cmb_generale = JoinNode(Join_Prediction_metrics(),
    #                                                joinsource='subject_list',
    #                                                joinfield='csv_files',
    #                                                name="prediction_metrics_cmb_generale")
    #     workflow.connect(prediction_metrics_cmb, 'biomarker_stats_csv', prediction_metrics_cmb_generale, 'csv_files')

    # QC part
    qc_crop_box = Node(Function(input_names=['img_apply_to',
                                             'brainmask',
                                             'bbox1',
                                             'bbox2',
                                             'cdg_ijk'],
                                output_names=['crop_brain_img'],
                                function=bounding_crop),
                       name='qc_crop_box')

    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:  # dual
        qc_coreg_FLAIR_T1 = Node(Function(input_names=['path_image', 'path_ref_image', 'path_brainmask', 'nb_of_slices'],
                                          output_names=['qc_coreg'],
                                          function=create_edges),
                                 name='qc_coreg_FLAIR_T1')
        qc_coreg_FLAIR_T1.inputs.nb_of_slices = 5  # Should be enough

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                                         output_names=['qc_overlay_brainmask_t1'],
                                         function=overlay_brainmask),
                                name='overlay_brainmask')

    # Building the actual report (html then pdf)
    summary_report = Node(SummaryReport(), name="summary_report")
    # Segmentation section
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', summary_report, 'pvs_metrics_csv')
        workflow.connect(prediction_metrics_pvs, 'biomarker_census_csv', summary_report, 'pvs_census_csv')
    if 'WMH' in kwargs['PREDICTION']:
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', summary_report, 'wmh_metrics_csv')
        workflow.connect(prediction_metrics_wmh, 'biomarker_census_csv', summary_report, 'wmh_census_csv')
    # if 'CMB' in kwargs['PREDICTION']:  # TODO: Add SWI
    #     workflow.connect(prediction_metrics_cmb, 'biomarker_stats_csv', summary_report, 'cmb_metrics')

    # QC section
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']
    summary_report.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
    summary_report.inputs.min_seg_size: {
        'PVS': kwargs['MIN_PVS_SIZE'],
        'WMH': kwargs['MIN_WMH_SIZE'],
        'CMB': kwargs['MIN_CMB_SIZE']}
    summary_report.inputs.pred_list = preds

    workflow.connect(qc_crop_box, 'crop_brain_img', summary_report, 'crop_brain_img')
    workflow.connect(qc_overlay_brainmask, 'qc_overlay_brainmask_t1', summary_report, 'qc_overlay_brainmask_t1')
    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:
        workflow.connect(qc_coreg_FLAIR_T1, 'qc_coreg', summary_report, 'isocontour_slides_FLAIR_T1')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
