#!/usr/bin/env python
"""Nipype workflow for conformation and preparation before deep
   learning, with accessory image coregistation to cropped space (through ANTS). 
   This also handles back-registration from conformed-crop to main.
   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.ants import ApplyTransforms

from shivautils.interfaces.shiva import PredictDirect
from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


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
    workflow = Workflow("shiva_predictor_Nagahama_dual_interpolation")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['T1', 'FLAIR']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'T1': [['subject_id', 'T1']],
                                        'FLAIR': [['subject_id', 'FLAIR']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')


    # conform main to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = (1.0, 1.0, 1.0)
    conform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "T1", conform, 'img')
    
    # preconform main to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(),
                   name="preconform")
    preconform.inputs.dimensions = (160, 214, 176)
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "T1", preconform, 'img')

    
    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile = 99), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled',
                     preconf_normalization, 'input_image')

    # brain mask from tensorflow
    pre_brain_mask = Node(PredictDirect(), name="pre_brain_mask")
    pre_brain_mask.inputs.model = kwargs['MODELS_PATH']
    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.out_filename = 'brain_mask_map.nii.gz'

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')
                     
    
    # send mask from preconformed space to 
    # conformed space 256 256 256 , same as anatomical conformed image
    unconform = Node(Conform(),
                name="unpreconform")
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
                     post_brain_mask, 't1')

    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_post_brain_mask")
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    # compute 6-dof coregistration parameters of accessory scan 
    # to cropped  main image
    coreg = Node(ants.Registration(),
                 name='coregister')
    coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    coreg.inputs.transforms = ['Rigid']
    coreg.inputs.transform_parameters = [(0.1,)]
    coreg.inputs.metric = ['MI']
    coreg.inputs.radius_or_number_of_bins = [64]
    coreg.inputs.interpolation = 'WelchWindowedSinc'
    coreg.inputs.shrink_factors = [[8,4,2,1]]
    coreg.inputs.output_warped_image = True
    coreg.inputs.smoothing_sigmas = [[3,2,1,0]]
    coreg.inputs.num_threads = 8
    coreg.inputs.number_of_iterations = [[1000,500,250,125]]
    coreg.inputs.sampling_strategy = ['Regular']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.output_transform_prefix = "main_to_acc_"
    coreg.inputs.verbose = True
    coreg.inputs.winsorize_lower_quantile = 0.005
    coreg.inputs.winsorize_upper_quantile = 0.995

    workflow.connect(datagrabber, "FLAIR",
                     coreg, 'moving_image')
    workflow.connect(crop, 'cropped',
                     coreg, 'fixed_image')
                
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

    workflow.connect(datagrabber, "T1",
                     crop_to_main, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_main, 'moving_image')

    

    # write brain seg on main in native space
    mask_to_main = Node(ants.ApplyTransforms(), name="mask_to_main")
    mask_to_main.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_main, 'forward_transforms', mask_to_main, 'transforms' )
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_main, 'input_image')
    workflow.connect(datagrabber, "T1", mask_to_main, 'reference_image')
    
    # write mask to acc in native space
    mask_to_acc = Node(ants.ApplyTransforms(), name="mask_to_acc")
    mask_to_acc.inputs.interpolation = 'NearestNeighbor'
    mask_to_acc.inputs.invert_transform_flags = [True]
    
    workflow.connect(coreg, 'forward_transforms',
                     mask_to_acc, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     mask_to_acc, 'input_image')
    workflow.connect(datagrabber, "FLAIR",
                     mask_to_acc, 'reference_image')
    
    # write original image into main crop space
    main_to_mask = Node(ants.ApplyTransforms(), name="main_to_mask")
    main_to_mask.inputs.invert_transform_flags = [True]
    # itere value
    main_to_mask.iterables = ('interpolation', ['Linear', 'HammingWindowedSinc'])

    workflow.connect(crop_to_main, 'forward_transforms', main_to_mask, 'transforms')
    workflow.connect(datagrabber, "T1", main_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', main_to_mask, 'reference_image')

    # write original image into acc crop space
    acc_to_mask = Node(ants.ApplyTransforms(), name="acc_to_mask")
    acc_to_mask.inputs.interpolation = 'Linear'

    workflow.connect(coreg, 'forward_transforms', acc_to_mask, 'transforms')
    workflow.connect(datagrabber, "FLAIR", acc_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', acc_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    main_norm =  Node(Normalization(percentile = 99), name="main_final_intensity_normalization")
    workflow.connect(main_to_mask, 'output_image',
    		         main_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
    	 	         main_norm, 'brain_mask')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 2)
    acc_norm =  Node(Normalization(percentile = 99), name="acc_final_intensity_normalization")
    workflow.connect(acc_to_mask, 'output_image',
    		         acc_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
    	 	         acc_norm, 'brain_mask')
    
    #predict_wmh = Node(PredictDirect(), name="predict_wmh")
    #predict_wmh.inputs.model = kwargs['MODELS_PATH']
    #predict_wmh.inputs.descriptor = kwargs['WMH_DESCRIPTOR']
    #predict_wmh.inputs.out_filename = 'wmh_map.nii.gz'

    #workflow.connect(main_norm, "intensity_normalized", predict_wmh, "t1")
    #workflow.connect(acc_norm, "intensity_normalized", predict_wmh, "flair")

    predict_pvs = Node(PredictDirect(), name="predict_pvs")
    predict_pvs.inputs.model = kwargs['MODELS_PATH']
    predict_pvs.inputs.descriptor = kwargs['PVS_DESCRIPTOR']
    predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'

    workflow.connect(main_norm, "intensity_normalized", predict_pvs, "t1")
    workflow.connect(acc_norm, "intensity_normalized", predict_pvs, "flair")

    # warp back on seg_WMH native space
    #wmh_seg_to_native = Node(ApplyTransforms(), name="wmh_seg_to_native")
    #wmh_seg_to_native.inputs.interpolation = 'NearestNeighbor'
    #wmh_seg_to_native.inputs.invert_transform_flags = [True]

    #workflow.connect(crop_to_main, 'forward_transforms',
                     #wmh_seg_to_native, 'transforms')
    #workflow.connect(predict_wmh, 'segmentation',
                     #wmh_seg_to_native, 'input_image')
    #workflow.connect(datagrabber, 't1',
                     #wmh_seg_to_native, 'reference_image')
    
    # warp back on seg_PVS native space
    pvs_seg_to_native = Node(ApplyTransforms(), name="pvs_seg_to_native")
    pvs_seg_to_native.inputs.interpolation = 'NearestNeighbor'
    pvs_seg_to_native.inputs.invert_transform_flags = [True]


    workflow.connect(crop_to_main, 'forward_transforms',
                     pvs_seg_to_native, 'transforms')
    workflow.connect(predict_pvs, 'segmentation',
                     pvs_seg_to_native, 'input_image')
    workflow.connect(datagrabber, 'T1',
                     pvs_seg_to_native, 'reference_image')


    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = 'output'
    #workflow.connect(predict_wmh, 'segmentation', datasink, 'segmentation_wmh')
    workflow.connect(predict_pvs, 'segmentation', datasink, 'segmentation_pvs')
    #workflow.connect(wmh_seg_to_native, 'output_image', datasink, 'segmentation_wmh_original_space')
    workflow.connect(pvs_seg_to_native, 'output_image', datasink, 'segmentation_pvs_original_space')
    workflow.connect(post_brain_mask, 'segmentation', datasink, 'brain_mask')
    workflow.connect(main_norm, 'intensity_normalized', datasink, 'T1_cropped_normalize')
    workflow.connect(acc_norm, 'intensity_normalized', datasink, 'FLAIR_cropped_normalize')


    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
