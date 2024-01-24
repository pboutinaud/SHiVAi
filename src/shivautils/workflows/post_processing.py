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
from shivautils.interfaces.image import (Regionwise_Prediction_metrics,
                                         Brain_Seg_for_biomarker,
                                         Label_clusters)
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
    summary_report = Node(SummaryReport(), name="summary_report")
    for pred in kwargs['PREDICTION']:
        model_descriptor = os.path.join(kwargs['MODELS_PATH'], kwargs[f'{pred}_DESCRIPTOR'])
        if pred == 'CMB' and with_t1:  # Requires registrations to T1 and SWI
            preds.append('CMB')
            lpred = pred.lower()
            cluster_labelling_cmb = Node(Label_clusters(),
                                         name='cluster_labelling_cmb')
            cluster_labelling_cmb.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
            cluster_labelling_cmb.inputs.thr_cluster_size = kwargs['MIN_CMB_SIZE'] - 1
            cluster_labelling_cmb.inputs.out_name = 'labelled_cmb.nii.gz'

            prediction_metrics = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_cmb")
            prediction_metrics.inputs.biomarker_type = 'cmb_swi-space'

            if kwargs['BRAIN_SEG'] == 'synthseg':
                prediction_metrics.inputs.brain_seg_type = 'synthseg'

                seg_to_swi = Node(ants.ApplyTransforms(), name="seg_to_swi")  # Register custom parc to swi space
                seg_to_swi.inputs.interpolation = 'NearestNeighbor'
                seg_to_swi.inputs.out_postfix = '_swi-space'
                seg_to_swi.inputs.invert_transform_flags = [True]  # original transform is swi to t1

                custom_cmb_parc = Node(Brain_Seg_for_biomarker(), name='custom_cmb_parc')
                custom_cmb_parc.inputs.custom_parc = 'mars'
                custom_cmb_parc.inputs.out_file = 'Brain_Seg_for_CMB_swi-space.nii.gz'

                workflow.connect(seg_to_swi, 'output_image', custom_cmb_parc, 'brain_seg')
                # seg_to_swi also needs external connection for 'reference_image', 'transforms' and 'input_image'
                workflow.connect(custom_cmb_parc, 'brain_seg', prediction_metrics, 'brain_seg')
                workflow.connect(custom_cmb_parc, 'region_dict', prediction_metrics, 'region_dict')
                workflow.connect(custom_cmb_parc, 'brain_seg', cluster_labelling_cmb, 'brain_seg')
            else:
                # Requires external connection for (prediction_metrics, 'brain_seg') and (cluster_labelling_cmb, 'brain_seg')
                prediction_metrics.inputs.brain_seg_type = 'brain_mask'
                prediction_metrics.inputs.region_list = ['Whole brain']

            workflow.connect(cluster_labelling_cmb, 'labelled_biomarkers', prediction_metrics, 'labelled_clusters')
        else:
            if pred == 'PVS2':
                pred = 'PVS'
            preds.append(pred)
            lpred = pred.lower()
            cluster_labelling = Node(Label_clusters(),
                                     name=f'cluster_labelling_{lpred}')
            cluster_labelling.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
            cluster_labelling.inputs.thr_cluster_size = kwargs[f'MIN_{pred}_SIZE'] - 1  # "- 1 because thr removes up to given value"
            cluster_labelling.inputs.out_name = f'labelled_{lpred}.nii.gz'
            prediction_metrics = Node(Regionwise_Prediction_metrics(),
                                      name=f"prediction_metrics_{lpred}")
            prediction_metrics.inputs.biomarker_type = lpred
            if kwargs['BRAIN_SEG'] == 'synthseg':
                prediction_metrics.inputs.brain_seg_type = 'synthseg'
                custom_parc = Node(Brain_Seg_for_biomarker(), name=f'custom_{lpred}_parc')
                custom_parc.inputs.out_file = f'Brain_Seg_for_{pred}.nii.gz'
                if pred in ('PVS', 'WMH'):  # Specific parcellation scheme for those two
                    custom_parc.inputs.custom_parc = lpred
                else:
                    custom_parc.inputs.custom_parc = 'mars'
                workflow.connect(custom_parc, 'brain_seg', cluster_labelling, 'brain_seg')
                workflow.connect(custom_parc, 'brain_seg', prediction_metrics, 'brain_seg')
                workflow.connect(custom_parc, 'region_dict', prediction_metrics, 'region_dict')
            else:  # external connection for (cluster_labelling_*, 'brain_seg')
                prediction_metrics.inputs.brain_seg_type = 'brain_mask'
                prediction_metrics.inputs.region_list = ['Whole brain']
            workflow.connect(cluster_labelling, 'labelled_biomarkers', prediction_metrics, 'labelled_clusters')

        # Building the actual report (html then pdf)
        setattr(summary_report.inputs, f'{lpred}_model_descriptor', model_descriptor)
        workflow.connect(prediction_metrics, 'biomarker_stats_csv', summary_report, f'{lpred}_metrics_csv')
        workflow.connect(prediction_metrics, 'biomarker_census_csv', summary_report, f'{lpred}_census_csv')

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
