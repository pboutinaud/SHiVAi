#!/usr/bin/env python
"""Nipype workflow for post-processing.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.image import (Apply_mask, Regionwise_Prediction_metrics,
                                         Join_Prediction_metrics, SummaryReport)
from shivautils.postprocessing.isocontour import create_edges
from shivautils.stats import overlay_brainmask


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
        cmb_map = cmb_pred_dict[subject_id]
    else:
        cmb_map = None
    T1_cropped = preproc_dict[subject_id]['T1_cropped']
    brainmask = preproc_dict[subject_id]['brainmask']
    pre_brainmask = preproc_dict[subject_id]['pre_brainmask']
    T1_conform = preproc_dict[subject_id]['T1_conform']
    BBOX1 = preproc_dict[subject_id]['BBOX1']
    BBOX2 = preproc_dict[subject_id]['BBOX2']
    CDG_IJK = preproc_dict[subject_id]['CDG_IJK']
    FLAIR_cropped = preproc_dict[subject_id]['FLAIR_cropped']
    SWI_cropped = preproc_dict[subject_id]['SWI_cropped']
    return (segmentation_pvs, segmentation_wmh, cmb_map, T1_cropped, brainmask, pre_brainmask,
            T1_conform, BBOX1, BBOX2, CDG_IJK, wf_graph,
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

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    if 'SUB_WF' in kwargs.keys() and kwargs['SUB_WF']:  # From previous workflow
        input_node = Node(
            Function(
                input_names=[
                    'subject_id',
                    'preproc_dict',
                    'pvs_pred_dict',
                    'wmh_pred_dict',
                    'cmb_pred_dict',
                    'wf_graph',
                    'brain_seg'],
                output_names=[
                    'segmentation_pvs',
                    'segmentation_wmh',
                    'cmb_map',
                    'T1_cropped',
                    'brainmask',
                    'pre_brainmask',
                    'T1_conform',
                    'BBOX1',
                    'BBOX2',
                    'CDG_IJK',
                    'wf_graph',
                    'FLAIR_cropped',
                    'SWI_cropped',
                    'brain_seg'],
                function=get_maps_from_dict),
            name='post_proc_input_node'
        )
        input_node.inputs.wf_graph = os.path.join(
            kwargs['BASE_DIR'], 'full_workflow', 'graph.svg')
    else:

        # file selection
        input_node = Node(DataGrabber(infields=['subject_id'],
                                      outfields=['segmentation_pvs', 'segmentation_wmh', 'cmb_map', 'brainmask',
                                                 'pre_brainmask', 'T1_cropped', 'FLAIR_cropped', 'T1_conform',
                                                 'CDG_IJK', 'BBOX1', 'BBOX2', 'wf_graph']
                                      ),
                          name='dataGrabber')
        input_node.inputs.template = '%s/%s/*.nii*'
        input_node.inputs.raise_on_empty = False
        input_node.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', input_node, 'subject_id')

    # Preparing stats and figures for the report
    # Segmentation part
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:  # WARN: None of this is SWOMed compatible
        prediction_metrics_pvs = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_pvs")
        prediction_metrics_pvs.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_pvs.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1  # "- 1 because thr removes up to given value"
        # TODO: This is when using only brainmask, we need synthseg for BG
        workflow.connect(input_node, 'segmentation_pvs', prediction_metrics_pvs, 'img')
        # if not synthseg:
        prediction_metrics_pvs.inputs.region_list = ['Whole_brain']
        workflow.connect(input_node, 'brainmask', prediction_metrics_pvs, 'brain_seg')
        # else:
        # prediction_metrics_pvs.inputs.region_list = ['Whole_brain', 'Basal_ganglia']
        # workflow.connect(input_node, 'brain_seg', prediction_metrics_pvs, 'brain_seg')
        prediction_metrics_pvs_generale = JoinNode(Join_Prediction_metrics(),
                                                   joinsource='subject_list',
                                                   joinfield='csv_files',
                                                   name="prediction_metrics_pvs_generale")
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', prediction_metrics_pvs_generale, 'csv_files')

    if 'WMH' in kwargs['PREDICTION']:
        mask_on_pred_wmh = Node(Apply_mask(),
                                name='mask_on_pred_wmh')
        prediction_metrics_wmh = Node(Regionwise_Prediction_metrics(),
                                      name="prediction_metrics_wmh")
        prediction_metrics_wmh_generale = JoinNode(Join_Prediction_metrics(),
                                                   joinsource='subject_list',
                                                   joinfield='csv_files',
                                                   name="prediction_metrics_wmh_generale")
        prediction_metrics_wmh.inputs.thr_cluster_val = kwargs['THRESHOLD_CLUSTERS']
        prediction_metrics_wmh.inputs.thr_cluster_size = kwargs['MIN_WMH_SIZE'] - 1
        workflow.connect(input_node, 'segmentation_wmh', mask_on_pred_wmh, 'segmentation')
        workflow.connect(input_node, 'brainmask', mask_on_pred_wmh, 'brainmask')
        workflow.connect(mask_on_pred_wmh, 'segmentation_filtered', prediction_metrics_wmh, 'img')
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', prediction_metrics_wmh_generale, 'csv_files')

    # QC part
    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:  # dual
        qc_coreg_FLAIR_T1 = Node(Function(input_names=['path_image', 'path_ref_image', 'path_brainmask', 'nb_of_slices'],
                                          output_names=['qc_coreg'],
                                          function=create_edges),
                                 name='qc_coreg_FLAIR_T1')
        qc_coreg_FLAIR_T1.inputs.nb_of_slices = 5  # Should be enough
        workflow.connect(input_node, 'FLAIR_cropped', qc_coreg_FLAIR_T1, 'path_image')
        workflow.connect(input_node, 'T1_cropped', qc_coreg_FLAIR_T1, 'path_ref_image')
        workflow.connect(input_node, 'brainmask', qc_coreg_FLAIR_T1, 'path_brainmask')

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                                         output_names=['qc_overlay_brainmask_t1'],
                                         function=overlay_brainmask),
                                name='overlay_brainmask')
    workflow.connect(input_node, 'brainmask', qc_overlay_brainmask, 'brainmask')
    workflow.connect(input_node, 'T1_cropped', qc_overlay_brainmask, 'img_ref')

    # Building the actual report (html then pdf)
    summary_report = Node(SummaryReport(), name="summary_report")
    workflow.connect(subject_list, 'subject_id', summary_report, 'subject_id')
    # Segmentation section
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        workflow.connect(prediction_metrics_pvs, 'biomarker_stats_csv', summary_report, 'metrics_clusters')
    if 'WMH' in kwargs['PREDICTION']:
        workflow.connect(prediction_metrics_wmh, 'biomarker_stats_csv', summary_report, 'metrics_clusters_2')
    # TODO: Add SWI, metrics_bg_pvs

    # QC section
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']

    workflow.connect(input_node, 'wf_graph', summary_report, 'wf_graph')
    workflow.connect(input_node, 'BBOX1', summary_report, 'bbox1')
    workflow.connect(input_node, 'BBOX2', summary_report, 'bbox2')
    workflow.connect(input_node, 'CDG_IJK', summary_report, 'cdg_ijk')
    workflow.connect(input_node, 'T1_conform', summary_report, 'img_normalized')
    workflow.connect(input_node, 'pre_brainmask', summary_report, 'brainmask')
    workflow.connect(qc_overlay_brainmask, 'qc_overlay_brainmask_t1', summary_report, 'qc_overlay_brainmask_t1')
    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:
        workflow.connect(qc_coreg_FLAIR_T1, 'qc_coreg', summary_report, 'isocontour_slides_FLAIR_T1')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
