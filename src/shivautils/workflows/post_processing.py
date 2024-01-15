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
from nipype.interfaces import ants

from shivautils.interfaces.post import SummaryReport
from shivautils.interfaces.image import Regionwise_Prediction_metrics, Brain_Seg_for_PVS
from shivautils.utils.misc import set_wf_shapers


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


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
        if 'PVS' in kwargs['PREDICTION']:
            pvs_descriptor = kwargs['PVS_DESCRIPTOR']
        elif 'PVS2' in kwargs['PREDICTION']:
            pvs_descriptor = kwargs['PVS2_DESCRIPTOR']
        preds.append('PVS')
        prediction_metrics_pvs = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_pvs")
        prediction_metrics_pvs.inputs.biomarker_type = 'pvs'
        prediction_metrics_pvs.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_pvs.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1  # "- 1 because thr removes up to given value"

        if kwargs['BRAIN_SEG'] == 'synthseg':
            prediction_metrics_pvs.inputs.brain_seg_type = 'synthseg'
            custom_pvs_parc = Node(Brain_Seg_for_PVS, name='custom_pvs_parc')
            workflow.connect(custom_pvs_parc, 'brain_seg_pvs', prediction_metrics_pvs, 'brain_seg')
            workflow.connect(custom_pvs_parc, 'pvs_region_dict', prediction_metrics_pvs, 'region_dict')
        else:
            prediction_metrics_pvs.inputs.brain_seg_type = 'brain_mask'
            prediction_metrics_pvs.inputs.region_list = ['Whole brain']

    if 'WMH' in kwargs['PREDICTION']:
        preds.append('WMH')
        prediction_metrics_wmh = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_wmh")
        prediction_metrics_wmh.inputs.biomarker_type = 'wmh'
        prediction_metrics_wmh.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_wmh.inputs.thr_cluster_size = kwargs['MIN_WMH_SIZE'] - 1
        if kwargs['BRAIN_SEG'] == 'synthseg':  # TODO do the real thing
            prediction_metrics_wmh.inputs.brain_seg_type = 'synthseg'
            prediction_metrics_wmh.inputs.region_list = ['Whole brain',
                                                         'Left cerebral WM',
                                                         'Right cerebral WM',
                                                         'Left cerebellum WM',
                                                         'Right cerebellum WM']
            # Connect the "brain_seg" input externally with synthseg parcelation
        else:
            prediction_metrics_wmh.inputs.brain_seg_type = 'brain_mask'
            prediction_metrics_wmh.inputs.region_list = ['Whole brain']

    if 'LAC' in kwargs['PREDICTION']:
        preds.append('LAC')
        prediction_metrics_lac = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_lac")
        prediction_metrics_lac.inputs.biomarker_type = 'lac'
        prediction_metrics_lac.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_lac.inputs.thr_cluster_size = kwargs['MIN_LAC_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_lac.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_lac.inputs.region_list = ['Whole brain']

    if 'CMB' in kwargs['PREDICTION']:
        preds.append('CMB')
        prediction_metrics_cmb = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_cmb")
        prediction_metrics_cmb.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_cmb.inputs.thr_cluster_size = kwargs['MIN_CMB_SIZE'] - 1
        # if not synthseg:  # TODO
        prediction_metrics_cmb.inputs.brain_seg_type = 'brain_mask'
        prediction_metrics_cmb.inputs.region_list = ['Whole brain']
        if with_t1:  # The metrics are computed on the segmentation put in T1 space, for coherence
            swi_pred_to_t1 = Node(ants.ApplyTransforms(), name="swi_pred_to_t1")
            swi_pred_to_t1.inputs.out_postfix = '_t1-space'
            workflow.connect(swi_pred_to_t1, 'output_image', prediction_metrics_cmb, 'img')
            prediction_metrics_cmb.inputs.biomarker_type = 'cmb_t1-space'
        else:
            prediction_metrics_cmb.inputs.biomarker_type = 'cmb'

    # Building the actual report (html then pdf)
    summary_report = Node(SummaryReport(), name="summary_report")
    # Segmentation section
    if 'PVS' in preds:
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', summary_report, 'pvs_metrics_csv')
        workflow.connect(prediction_metrics_pvs, 'biomarker_census_csv', summary_report, 'pvs_census_csv')
        summary_report.inputs.pvs_model_descriptor = os.path.join(kwargs['MODELS_PATH'], pvs_descriptor)
    if 'WMH' in preds:
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', summary_report, 'wmh_metrics_csv')
        workflow.connect(prediction_metrics_wmh, 'biomarker_census_csv', summary_report, 'wmh_census_csv')
        summary_report.inputs.wmh_model_descriptor = os.path.join(kwargs['MODELS_PATH'], kwargs['WMH_DESCRIPTOR'])
    if 'CMB' in preds:
        workflow.connect(prediction_metrics_cmb, 'biomarker_stats_csv', summary_report, 'cmb_metrics_csv')
        workflow.connect(prediction_metrics_cmb, 'biomarker_census_csv', summary_report, 'cmb_census_csv')
        summary_report.inputs.cmb_model_descriptor = os.path.join(kwargs['MODELS_PATH'], kwargs['CMB_DESCRIPTOR'])
    if 'LAC' in preds:
        workflow.connect(prediction_metrics_lac, 'biomarker_stats_csv', summary_report, 'lac_metrics_csv')
        workflow.connect(prediction_metrics_lac, 'biomarker_census_csv', summary_report, 'lac_census_csv')
        summary_report.inputs.lac_model_descriptor = os.path.join(kwargs['MODELS_PATH'], kwargs['LAC_DESCRIPTOR'])

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

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
