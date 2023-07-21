#!/usr/bin/env python
"""Nipype workflow for image conformation and preparation before deep
   learning"""
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.utility import Function
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.ants import ApplyTransforms

from shivautils.interfaces.shiva import PredictDirect, SynthSeg
from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop, ApplyMask, MetricsPredictions,
                            JoinMetricsPredictions, SummaryReport, MaskRegions,
                            QuantificationWMHVentricals)
from shivautils.interfaces.interfaces_post_processing import MakeDistanceVentricleMap
from shivautils.isocontour_board import create_edges
from shivautils.stats import save_histogram


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def swiWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_processing_swi")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['SWI']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    # conform main to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = (1.0, 1.0, 1.0)
    conform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, 'SWI', conform, 'img')

    # preconform main to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(), name="preconform")
    preconform.inputs.dimensions = (160, 214, 176)
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "SWI", preconform, 'img')


    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile = 99), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    # brain mask from tensorflow
    pre_brain_mask = Node(PredictDirect(), name="pre_brain_mask")
    pre_brain_mask.inputs.model = kwargs['MODELS_PATH']
    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.out_filename = 'brain_mask_map.nii.gz'

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 'swi')
    
    # send mask from preconformed space to
    # conformed space 256 256 256, same as anatomical conformed image
    unconform = Node(Conform(), name="unpreconform")
    unconform.inputs.dimensions = (256, 256, 256)
    unconform.inputs.voxel_size = (1.0, 1.0, 1.0)
    unconform.inputs.orientation = 'RAS'

    workflow.connect(pre_brain_mask, 'segmentation', unconform, 'img')

    # binarize unpreconformed brain mask
    hard_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_brain_mask")
    workflow.connect(unconform, 'resampled', hard_brain_mask, 'img')

    # normalize intensities between 0 and 1 for Tensorflow
    post_normalization = Node(Normalization(percentile = 99), name="post_intensity_normalization")
    workflow.connect(conform, 'resampled',
                     post_normalization, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
                    post_normalization, 'brain_mask')
    
    # crop main centered on mask origin
    crop_normalized = Node(Crop(final_dimensions=(160, 214, 176)),
                           name="crop_normalized")
    workflow.connect(post_normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop_normalized, 'roi_mask')
    
    # crop raw
    # crop main centered on mask
    crop = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop, 'roi_mask')
    
    # brain mask from tensorflow
    post_brain_mask = Node(PredictDirect(), "post_brain_mask")
    post_brain_mask.inputs.model = kwargs['MODELS_PATH']
    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    post_brain_mask.inputs.out_filename = 'brain_mask_map.nii.gz'
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 'swi')
    
    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_post_brain_mask")
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    # compute 3-dof (translations) coregistration parameters of cropped to native main
    crop_to_main = Node(ants.Registration(),
                 name='crop_to_main')
    crop_to_main.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    crop_to_main.inputs.transforms = ['Rigid']
    crop_to_main.inputs.restrict_deformation=[[1,0,0,],[1,0,0,],[1,0,0]]
    crop_to_main.inputs.transform_parameters = [(0.1,)]
    crop_to_main.inputs.metric = ['MI']
    crop_to_main.inputs.radius_or_number_of_bins = [64]
    crop_to_main.inputs.shrink_factors = [[8,4,2,1]]
    crop_to_main.inputs.output_warped_image = False
    crop_to_main.inputs.smoothing_sigmas = [[3,2,1,0]]
    crop_to_main.inputs.num_threads = 8
    crop_to_main.inputs.number_of_iterations = [[1000,500,250,125]]
    crop_to_main.inputs.sampling_strategy = ['Regular']
    crop_to_main.inputs.sampling_percentage = [0.25]
    crop_to_main.inputs.output_transform_prefix = "cropped_to_source_"
    crop_to_main.inputs.verbose = True
    crop_to_main.inputs.winsorize_lower_quantile = 0.0
    crop_to_main.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(datagrabber, "SWI",
                     crop_to_main, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_main, 'moving_image')


    # write brain seg on main in native space
    mask_to_main = Node(ants.ApplyTransforms(), name="mask_to_main")
    mask_to_main.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_main, 'forward_transforms', mask_to_main, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_main, 'input_image')
    workflow.connect(datagrabber, "SWI", mask_to_main, 'reference_image')

    # write original image into main crop space
    main_to_mask = Node(ants.ApplyTransforms(), name="main_to_mask")
    main_to_mask.inputs.invert_transform_flags = [True]
    main_to_mask.interpolation = 'WelchWindowedSinc'

    workflow.connect(crop_to_main, 'forward_transforms', main_to_mask, 'transforms')
    workflow.connect(datagrabber, "SWI", main_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', main_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    final_norm = Node(Normalization(percentile = 99), name="final_intensity_normalization")
    workflow.connect(main_to_mask, 'output_image',
                     final_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     final_norm, 'brain_mask')
    
    save_hist_final = Node(Function(input_names=['img_normalized'],
                             output_names=['hist'], function=save_histogram), name='save_hist_final')
    
    workflow.connect(final_norm, 'intensity_normalized', save_hist_final, 'img_normalized')


    predict_cmb = Node(PredictDirect(), name="predict_cmb")
    predict_cmb.inputs.model = kwargs['MODELS_PATH']
    predict_cmb.inputs.descriptor = kwargs['CMB_DESCRIPTOR']
    predict_cmb.inputs.out_filename = 'map.nii.gz'

    workflow.connect(final_norm, "intensity_normalized", predict_cmb, 'swi')

    # warp back on seg PVS or CMB native space
    cmb_seg_to_native = Node(ants.ApplyTransforms(), name="cmb_seg_to_native")
    cmb_seg_to_native.inputs.interpolation = 'NearestNeighbor'
    cmb_seg_to_native.inputs.invert_transform_flags = [True]

    workflow.connect(crop_to_main, 'forward_transforms',
                     cmb_seg_to_native, 'transforms')
    workflow.connect(predict_cmb, 'segmentation',
                    cmb_seg_to_native, 'input_image')
    workflow.connect(datagrabber, 'SWI',
                     cmb_seg_to_native, 'reference_image')
    
    mask_on_pred_cmb = Node(ApplyMask(), name='mask_on_pred_cmb')

    workflow.connect(predict_cmb, 'segmentation', mask_on_pred_cmb, 'segmentation')
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_on_pred_cmb, 'brainmask')

    qc_coreg_brainmask_SWI = Node(Function(input_names=['path_image', 'path_ref_image', 'path_brainmask','nb_of_slices'],
                                           output_names=['qc_coreg'], function=create_edges), name='qc_coreg_brainmask_T1')
    # Default number of slices - 8
    qc_coreg_brainmask_T1.inputs.nb_of_slices =  8


    metrics_predictions_cmb = Node(MetricsPredictions(),
                                   name="metrics_predictions_cmb")

    workflow.connect(subject_list, 'subject_id', metrics_predictions_cmb, 'subject_id')
    workflow.connect(mask_on_pred_cmb, 'segmentation_filtered', metrics_predictions_cmb, 'img')

    summary_report = Node(SummaryReport(), name="summary_report")

    workflow.connect(save_hist_final, 'hist', summary_report, 'histogram_intensity')
    workflow.connect(metrics_predictions_cmb, 'metrics_predictions_csv', summary_report, 'metrics_clusters')


    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = 'output'
    workflow.connect(save_hist_final, 'hist', datasink, 'img_histogram')
    workflow.connect(preconf_normalization, 'report', datasink, 'report_preconf_normalization')
    workflow.connect(final_norm, 'report', datasink, 'report_main_final_normalization')
    workflow.connect(metrics_predictions_cmb, 'metrics_predictions_csv', datasink, 'metrics_prediction_cmb')
    workflow.connect(predict_cmb, 'segmentation', datasink, 'segmentation')
    workflow.connect(cmb_seg_to_native, 'output_image', datasink, 'segmentation_original_space')
    workflow.connect(post_brain_mask, 'segmentation', datasink, 'brain_mask')
    workflow.connect(final_norm, 'intensity_normalized', datasink, 'img_cropped_normalize')
    workflow.connect(summary_report, "summary_report", datasink, "summary_report")
    workflow.connect(summary_report, 'summary', datasink, 'summary_pdf')

    metrics_predictions_generale = JoinNode(JoinMetricsPredictions(),
                                   joinsource = 'subject_list',
                                   joinfield= 'csv_files',
                                   name="metrics_predictions_generale")

    workflow.connect(metrics_predictions_cmb, 'metrics_predictions_csv', metrics_predictions_generale, 'csv_files')

    return workflow


if __name__ == '__main__':
    wf = swiWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')