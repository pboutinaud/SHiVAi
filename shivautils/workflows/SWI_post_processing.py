#!/usr/bin/env python
"""Nipype workflow for post-processing.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.image import (ApplyMask, MetricsPredictions, 
                            JoinMetricsPredictions, SummaryReport, MaskRegions,
                            QuantificationWMHLatVentricles, BGMask, PVSQuantificationBG)
from shivautils.interfaces.interfaces_post_processing import MakeDistanceMap
from shivautils.isocontour_board import create_edges
from shivautils.stats import save_histogram, bounding_crop, overlay_brainmask


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def as_list(input):
    return [input]
    

def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    name_workflow = "SWI_post_processing"
    workflow = Workflow(name_workflow)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['segmentation_cmb', 'brainmask', 
                                              'pre_brainmask', 'SWI_cropped',
                                              'SWI_conform', 'CDG_IJK', 'BBOX1', 'BBOX2',
                                              'sum_preproc_wf']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')



    mask_on_pred_cmb = Node(ApplyMask(), name='mask_on_pred_pvs')

    workflow.connect(datagrabber, 'segmentation_cmb', mask_on_pred_cmb, 'segmentation')
    workflow.connect(datagrabber, 'brainmask', mask_on_pred_cmb, 'brainmask')

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                             output_names=['qc_overlay_brainmask_main'], function=overlay_brainmask), name='overlay_brainmask')
                    
    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(datagrabber, 'brainmask', qc_overlay_brainmask, 'brainmask')
    workflow.connect(datagrabber, 'SWI_cropped', qc_overlay_brainmask, 'img_ref')

    metrics_predictions_cmb = Node(MetricsPredictions(),
                                   name="metrics_predictions_cmb")
    metrics_predictions_cmb.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']

    workflow.connect(mask_on_pred_cmb, 'segmentation_filtered', metrics_predictions_cmb, 'img')

    summary_report = Node(SummaryReport(), name="summary_report")
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.swi = kwargs['SWI']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']

    workflow.connect(datagrabber, 'sum_preproc_wf', summary_report, 'sum_workflow')
    workflow.connect(datagrabber, 'BBOX1', summary_report, 'bbox1')
    workflow.connect(datagrabber, 'BBOX2', summary_report, 'bbox2')
    workflow.connect(datagrabber, 'CDG_IJK', summary_report, 'cdg_ijk')
    workflow.connect(datagrabber, 'SWI_conform', summary_report, 'img_normalized')
    workflow.connect(datagrabber, 'pre_brainmask', summary_report, 'brainmask')
    workflow.connect(subject_list, 'subject_id', summary_report, 'subject_id')
    workflow.connect(qc_overlay_brainmask, 'qc_overlay_brainmask_main', summary_report, 'qc_overlay_brainmask_main')
    workflow.connect(metrics_predictions_cmb, 'metrics_predictions_csv', summary_report, 'metrics_clusters')

    metrics_predictions_cmb_generale = JoinNode(JoinMetricsPredictions(),
                                   joinsource = 'subject_list',
                                   joinfield= 'csv_files',
                                   name="metrics_predictions_cmb_generale")

    workflow.connect(metrics_predictions_cmb, 'metrics_predictions_csv', metrics_predictions_cmb_generale, 'csv_files')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')