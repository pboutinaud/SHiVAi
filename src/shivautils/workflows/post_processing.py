#!/usr/bin/env python
"""Nipype workflow for post-processing.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.image import (ApplyMask, MetricsPredictions,
                                         JoinMetricsPredictions, SummaryReport)
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
                       ):
    if pvs_pred_dict is not None:
        pvs_map = pvs_pred_dict[subject_id]
    else:
        pvs_map = None
    if wmh_pred_dict is not None:
        wmh_map = wmh_pred_dict[subject_id]
    else:
        wmh_map = None
    if cmb_pred_dict is not None:
        cmb_map = cmb_pred_dict[subject_id]
    else:
        cmb_map = None
    t1 = preproc_dict[subject_id].t1
    brainmask = preproc_dict[subject_id].brainmask
    pre_brainmask = preproc_dict[subject_id].pre_brainmask
    T1_cropped = preproc_dict[subject_id].T1_cropped
    T1_conform = preproc_dict[subject_id].T1_conform
    BBOX1 = preproc_dict[subject_id].BBOX1
    BBOX2 = preproc_dict[subject_id].BBOX2
    CDG_IJK = preproc_dict[subject_id].CDG_IJK
    wf_graph = None  # Placeholder
    flair = preproc_dict[subject_id].flair
    swi = preproc_dict[subject_id].swi
    return (pvs_map, wmh_map, cmb_map, t1, brainmask, pre_brainmask,
            T1_cropped, T1_conform, BBOX1, BBOX2, CDG_IJK, wf_graph,
            flair, swi)


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
                    'cmb_pred_dict'],
                output_names=[
                    'pvs_map',
                    'wmh_map',
                    'cmb_map',
                    't1',
                    'brainmask',
                    'pre_brainmask',
                    'T1_cropped',
                    'T1_conform',
                    'BBOX1',
                    'BBOX2',
                    'CDG_IJK',
                    'wf_graph',
                    'flair',
                    'swi'],
                function=get_maps_from_dict),
            name='post_proc_input_node'
        )
        input_node.inputs.wf_graph = os.path.join(
            kwargs['BASE_DIR'], 'full_workflow', 'graph.svg')
    else:

        # file selection
        input_node = Node(DataGrabber(infields=['subject_id'],
                                      outfields=['pvs_map', 'wmh_map', 'cmb_map', 'brainmask',
                                                 'pre_brainmask', 'T1_cropped', 'FLAIR_cropped',
                                                 'T1_conform', 'CDG_IJK', 'BBOX1', 'BBOX2',
                                                 'wf_graph']
                                      ),
                          name='dataGrabber')
        input_node.inputs.template = '%s/%s/*.nii*'
        input_node.inputs.raise_on_empty = False
        input_node.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', input_node, 'subject_id')

    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        mask_on_pred_pvs = Node(ApplyMask(), name='mask_on_pred_pvs')

    workflow.connect(input_node, 'segmentation_pvs', mask_on_pred_pvs, 'segmentation')
    workflow.connect(input_node, 'brainmask', mask_on_pred_pvs, 'brainmask')

    if 'WMH' in kwargs['PREDICTION']:
        mask_on_pred_wmh = Node(ApplyMask(), name='mask_on_pred_wmh')

        workflow.connect(input_node, 'segmentation_wmh', mask_on_pred_wmh, 'segmentation')
        workflow.connect(input_node, 'brainmask', mask_on_pred_wmh, 'brainmask')

    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:
        qc_coreg_FLAIR_T1 = Node(Function(input_names=['path_image', 'path_ref_image', 'path_brainmask', 'nb_of_slices'],
                                          output_names=['qc_coreg'], function=create_edges), name='qc_coreg_FLAIR_T1')
        # Default number of slices - 8
        qc_coreg_FLAIR_T1.inputs.nb_of_slices = 5

        # connecting image, ref_image and brainmask to create_edges function
        workflow.connect(input_node, 'FLAIR_cropped', qc_coreg_FLAIR_T1, 'path_image')
        workflow.connect(input_node, 'T1_cropped', qc_coreg_FLAIR_T1, 'path_ref_image')
        workflow.connect(input_node, 'brainmask', qc_coreg_FLAIR_T1, 'path_brainmask')

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                                         output_names=['qc_overlay_brainmask_t1'], function=overlay_brainmask), name='overlay_brainmask')

    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(input_node, 'brainmask', qc_overlay_brainmask, 'brainmask')
    workflow.connect(input_node, 'T1_cropped', qc_overlay_brainmask, 'img_ref')

    if 'WMH' in kwargs['PREDICTION']:
        metrics_predictions_wmh = Node(MetricsPredictions(),
                                       name="metrics_predictions_wmh")
        metrics_predictions_wmh.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']

        workflow.connect(mask_on_pred_wmh, 'segmentation_filtered', metrics_predictions_wmh, 'img')

    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        metrics_predictions_pvs = Node(MetricsPredictions(),
                                       name="metrics_predictions_pvs")
        metrics_predictions_pvs.pvs = True
        metrics_predictions_pvs.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']

        workflow.connect(mask_on_pred_pvs, 'segmentation_filtered', metrics_predictions_pvs, 'img')

    summary_report = Node(SummaryReport(), name="summary_report")
    # TODO: Add SWI, metrics_bg_pvs
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']

    workflow.connect(input_node, 'wf_graph', summary_report, 'sum_workflow')
    workflow.connect(input_node, 'BBOX1', summary_report, 'bbox1')
    workflow.connect(input_node, 'BBOX2', summary_report, 'bbox2')
    workflow.connect(input_node, 'CDG_IJK', summary_report, 'cdg_ijk')
    workflow.connect(input_node, 'T1_conform', summary_report, 'img_normalized')
    workflow.connect(input_node, 'pre_brainmask', summary_report, 'brainmask')
    workflow.connect(subject_list, 'subject_id', summary_report, 'subject_id')
    workflow.connect(qc_overlay_brainmask, 'qc_overlay_brainmask_t1', summary_report, 'qc_overlay_brainmask_t1')
    if 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:
        workflow.connect(qc_coreg_FLAIR_T1, 'qc_coreg', summary_report, 'isocontour_slides_FLAIR_T1')
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        workflow.connect(metrics_predictions_pvs, 'metrics_predictions_csv', summary_report, 'metrics_clusters')
    if 'WMH' in kwargs['PREDICTION']:
        workflow.connect(metrics_predictions_wmh, 'metrics_predictions_csv', summary_report, 'metrics_clusters_2')

    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        metrics_predictions_pvs_generale = JoinNode(JoinMetricsPredictions(),
                                                    joinsource='subject_list',
                                                    joinfield='csv_files',
                                                    name="metrics_predictions_pvs_generale")
        workflow.connect(metrics_predictions_pvs, 'metrics_predictions_csv', metrics_predictions_pvs_generale, 'csv_files')

    if 'WMH' in kwargs['PREDICTION']:
        metrics_predictions_wmh_generale = JoinNode(JoinMetricsPredictions(),
                                                    joinsource='subject_list',
                                                    joinfield='csv_files',
                                                    name="metrics_predictions_wmh_generale")

        workflow.connect(metrics_predictions_wmh, 'metrics_predictions_csv', metrics_predictions_wmh_generale, 'csv_files')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
