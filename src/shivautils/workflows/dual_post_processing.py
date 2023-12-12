#!/usr/bin/env python
"""Nipype workflow for post-processing.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.shiva import SynthSeg
from shivautils.interfaces.image import (ApplyMask, MetricsPredictions, MakeDistanceMap,
                                         JoinMetricsPredictions, SummaryReport, MaskRegions,
                                         QuantificationWMHLatVentricles, BGMask, PVSQuantificationBG)
from shivautils.postprocessing.isocontour import create_edges
from shivautils.utils.stats import overlay_brainmask


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def as_list(input):
    return [input]


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

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

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['segmentation_pvs', 'segmentation_wmh', 'brainmask',
                                              'pre_brainmask', 'T1_cropped', 'FLAIR_cropped',
                                              'T1_conform', 'CDG_IJK', 'BBOX1', 'BBOX2',
                                              'sum_preproc_wf']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    mask_on_pred_pvs = Node(ApplyMask(), name='mask_on_pred_pvs')

    workflow.connect(datagrabber, 'segmentation_pvs', mask_on_pred_pvs, 'segmentation')
    workflow.connect(datagrabber, 'brainmask', mask_on_pred_pvs, 'brainmask')

    mask_on_pred_wmh = Node(ApplyMask(), name='mask_on_pred_wmh')

    workflow.connect(datagrabber, 'segmentation_wmh', mask_on_pred_wmh, 'segmentation')
    workflow.connect(datagrabber, 'brainmask', mask_on_pred_wmh, 'brainmask')

    qc_coreg_FLAIR_T1 = Node(Function(input_names=['path_image', 'path_ref_image', 'path_brainmask', 'nb_of_slices'],
                             output_names=['qc_coreg'], function=create_edges), name='qc_coreg_FLAIR_T1')
    # Default number of slices - 8
    qc_coreg_FLAIR_T1.inputs.nb_of_slices = 5

    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(datagrabber, 'FLAIR_cropped', qc_coreg_FLAIR_T1, 'path_image')
    workflow.connect(datagrabber, 'T1_cropped', qc_coreg_FLAIR_T1, 'path_ref_image')
    workflow.connect(datagrabber, 'brainmask', qc_coreg_FLAIR_T1, 'path_brainmask')

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                                         output_names=['qc_overlay_brainmask_t1'], function=overlay_brainmask), name='overlay_brainmask')

    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(datagrabber, 'brainmask', qc_overlay_brainmask, 'brainmask')
    workflow.connect(datagrabber, 'T1_cropped', qc_overlay_brainmask, 'img_ref')

    synthseg = Node(SynthSeg(), name='synthseg')
    synthseg.inputs.out_filename = 'segmentation_regions.nii.gz'
    workflow.connect(datagrabber, 'T1_cropped', synthseg, 'i')

    mask_latventricles_regions = Node(MaskRegions(), name='mask_latventricles_regions')
    mask_latventricles_regions.inputs.list_labels_regions = [4, 5, 43, 44]
    workflow.connect(synthseg, 'segmentation', mask_latventricles_regions, 'img')

    bg_mask = Node(BGMask(), name="bg_mask")
    workflow.connect(synthseg, "segmentation", bg_mask, "segmented_regions")

    # Creating a distance map for each ventricle mask
    make_distance_latventricles_map = Node(MakeDistanceMap(), name="make_distance_latventricles_map")
    make_distance_latventricles_map.inputs.out_file = 'distance_map.nii.gz'
    workflow.connect(mask_latventricles_regions, 'mask_regions', make_distance_latventricles_map, "in_file")

    wmh_quantification_latventricles = Node(QuantificationWMHLatVentricles(), name='wmh_quantification_latventricles')
    wmh_quantification_latventricles.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']
    workflow.connect(datagrabber, 'segmentation_wmh', wmh_quantification_latventricles, 'wmh')
    workflow.connect(make_distance_latventricles_map, 'out_file', wmh_quantification_latventricles, 'latventricles_distance_maps')
    workflow.connect(subject_list, 'subject_id', wmh_quantification_latventricles, 'subject_id')

    pvs_quantification_bg = Node(PVSQuantificationBG(), name="pvs_quantification_bg")
    pvs_quantification_bg.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']
    workflow.connect(datagrabber, 'segmentation_pvs', pvs_quantification_bg, "img")
    workflow.connect(bg_mask, 'bg_mask', pvs_quantification_bg, "bg_mask")

    metrics_predictions_wmh = Node(MetricsPredictions(),
                                   name="metrics_predictions_wmh")
    metrics_predictions_wmh.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']

    workflow.connect(mask_on_pred_wmh, 'segmentation_filtered', metrics_predictions_wmh, 'img')

    metrics_predictions_pvs = Node(MetricsPredictions(),
                                   name="metrics_predictions_pvs")
    metrics_predictions_pvs.pvs = True
    metrics_predictions_pvs.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']

    workflow.connect(mask_on_pred_pvs, 'segmentation_filtered', metrics_predictions_pvs, 'img')

    summary_report = Node(SummaryReport(), name="summary_report")
    summary_report.inputs.anonymized = kwargs['ANONYMIZED']
    summary_report.inputs.percentile = kwargs['PERCENTILE']
    summary_report.inputs.threshold = kwargs['THRESHOLD']
    summary_report.inputs.image_size = kwargs['IMAGE_SIZE']
    summary_report.inputs.resolution = kwargs['RESOLUTION']

    workflow.connect(datagrabber, 'sum_preproc_wf', summary_report, 'sum_workflow')
    workflow.connect(datagrabber, 'BBOX1', summary_report, 'bbox1')
    workflow.connect(datagrabber, 'BBOX2', summary_report, 'bbox2')
    workflow.connect(datagrabber, 'CDG_IJK', summary_report, 'cdg_ijk')
    workflow.connect(datagrabber, 'T1_conform', summary_report, 'img_normalized')
    workflow.connect(datagrabber, 'pre_brainmask', summary_report, 'brainmask')
    workflow.connect(subject_list, 'subject_id', summary_report, 'subject_id')
    workflow.connect(qc_coreg_FLAIR_T1, 'qc_coreg', summary_report, 'isocontour_slides_FLAIR_T1')
    workflow.connect(qc_overlay_brainmask, 'qc_overlay_brainmask_t1', summary_report, 'qc_overlay_brainmask_t1')
    workflow.connect(metrics_predictions_pvs, 'metrics_predictions_csv', summary_report, 'metrics_clusters')
    workflow.connect(metrics_predictions_wmh, 'metrics_predictions_csv', summary_report, 'metrics_clusters_2')
    workflow.connect(pvs_quantification_bg, 'metrics_bg_pvs', summary_report, 'metrics_bg_pvs')
    workflow.connect(wmh_quantification_latventricles, 'predictions_latventricles_DWMH', summary_report, 'predictions_latventricles_DWMH')

    metrics_predictions_pvs_generale = JoinNode(JoinMetricsPredictions(),
                                                joinsource='subject_list',
                                                joinfield='csv_files',
                                                name="metrics_predictions_pvs_generale")

    workflow.connect(metrics_predictions_pvs, 'metrics_predictions_csv', metrics_predictions_pvs_generale, 'csv_files')

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
