#!/usr/bin/env python
"""Nipype workflow for post-processing.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivai.interfaces.image import (Apply_mask, MetricsPredictions,
                                     JoinMetricsPredictions)
from shivai.interfaces.post import SummaryReport
from shivai.interfaces.image import Isocontour
from shivai.utils.quality_control import overlay_brainmask


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    name_workflow = "post_processing_workflow_container"
    workflow = Workflow(name_workflow)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    preprocDir = 'shiva_dual_preprocessing'
    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['segmentation_pvs', 'segmentation_wmh', 'brainmask',
                                              'pre_brainmask', 'T1_cropped', 'FLAIR_cropped',
                                              'T1_conform', 'CDG_IJK', 'BBOX1', 'BBOX2',
                                              'sum_preproc_wf']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'dual_predictor_workflow')
    datagrabber.inputs.template = '%s/%s/*.nii*'
    datagrabber.inputs.template_args = {'segmentation_pvs': [['subject_id']],
                                        'segmentation_wmh': [['subject_id']],
                                        'brainmask': [['subject_id']],
                                        'pre_brainmask': [['subject_id']],
                                        'T1_cropped': [['subject_id', 'subject_id']],
                                        'FLAIR_cropped': [['subject_id']],
                                        'T1_conform': [['subject_id', 'subject_id']],
                                        'BBOX1': [['subject_id']],
                                        'BBOX2': [['subject_id']],
                                        'CDG_IJK': [['subject_id']],
                                        'sum_preproc_wf': [[]]}
    datagrabber.inputs.field_template = {'segmentation_pvs': '_subject_id_%s/predict_pvs/pvs_map.nii.gz',
                                         'segmentation_wmh': '_subject_id_%s/predict_wmh/wmh_map.nii.gz',
                                         'brainmask': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/hard_post_brain_mask/post_brain_mask_thresholded.nii.gz'),
                                         'pre_brainmask': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/hard_brain_mask/pre_brain_maskresampled_thresholded.nii.gz'),
                                         'T1_cropped': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/t1_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz'),
                                         'FLAIR_cropped': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/flair_final_intensity_normalization/flair_to_t1__Warped_img_normalized.nii.gz'),
                                         'T1_conform': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/conform/%s_T1_rawresampled.nii.gz'),
                                         'BBOX1': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/crop/bbox1.txt'),
                                         'BBOX2': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/crop/bbox2.txt'),
                                         'CDG_IJK': os.path.join(kwargs['BASE_DIR'], preprocDir, '_subject_id_%s/crop/cdg_ijk.txt'),
                                         'sum_preproc_wf': os.path.join(kwargs['BASE_DIR'], preprocDir, 'graph.svg')}
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    mask_on_pred_pvs = Node(Apply_mask(), name='mask_on_pred_pvs')

    workflow.connect(datagrabber, 'segmentation_pvs', mask_on_pred_pvs, 'segmentation')
    workflow.connect(datagrabber, 'brainmask', mask_on_pred_pvs, 'brainmask')

    mask_on_pred_wmh = Node(Apply_mask(), name='mask_on_pred_wmh')

    workflow.connect(datagrabber, 'segmentation_wmh', mask_on_pred_wmh, 'segmentation')
    workflow.connect(datagrabber, 'brainmask', mask_on_pred_wmh, 'brainmask')

    qc_coreg_FLAIR_T1 = Node(Isocontour(),
                             name='qc_coreg_FLAIR_T1')
    qc_coreg_FLAIR_T1.inputs.nb_of_slices = 12

    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(datagrabber, 'FLAIR_cropped', qc_coreg_FLAIR_T1, 'path_image')
    workflow.connect(datagrabber, 'T1_cropped', qc_coreg_FLAIR_T1, 'path_ref_image')
    workflow.connect(datagrabber, 'brainmask', qc_coreg_FLAIR_T1, 'path_brainmask')

    qc_overlay_brainmask = Node(Function(input_names=['img_ref', 'brainmask'],
                                         output_names=['qc_overlay_brainmask_t1'], function=overlay_brainmask), name='overlay_brainmask')

    # connecting image, ref_image and brainmask to create_edges function
    workflow.connect(datagrabber, 'brainmask', qc_overlay_brainmask, 'brainmask')
    workflow.connect(datagrabber, 'T1_cropped', qc_overlay_brainmask, 'img_ref')

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
