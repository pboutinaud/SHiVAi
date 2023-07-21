#!/usr/bin/env python
"""Nipype workflow for image conformation and preparation before deep
   learning"""
import os

from nipype import Function
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.shiva import Predict
from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("single_T1_trainset_preprocessing")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subjectList = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subjectList")
    subjectList.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['T1', 'GIN']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subjectList, 'subject_id', datagrabber, 'subject_id')

    # conform main to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = (1.0, 1.0, 1.0)
    conform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, 'T1', conform, 'img')

    # preconform main to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(), name="preconform")
    preconform.inputs.dimensions = (160, 214, 176)
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "T1", preconform, 'img')


    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile = 99), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    # brain mask from tensorflow
    pre_brain_mask = Node(Predict(), name="pre_brain_mask")
    pre_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    pre_brain_mask.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
    pre_brain_mask.inputs.model = '/mnt/model'
    pre_brain_mask.inputs.snglrt_enable_nvidia = True
    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'
    pre_brain_mask.inputs.out_filename = 'brain_mask_map.nii.gz'

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')
    
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
    post_brain_mask = Node(Predict(), "post_brain_mask")
    post_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    post_brain_mask.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
    post_brain_mask.inputs.model = '/mnt/model'
    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    post_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'
    post_brain_mask.inputs.out_filename = 'brain_mask_map.nii.gz'
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 't1')
    
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

    workflow.connect(datagrabber, "T1",
                     crop_to_main, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_main, 'moving_image')
    
    # compute 3-dof (translations) coregistration parameters of gin image to native main
    gin_to_final = Node(ants.Registration(),
                 name='gin_to_final')
    gin_to_final.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    gin_to_final.inputs.transforms = ['Rigid']
    gin_to_final.inputs.restrict_deformation=[[1,0,0,],[1,0,0,],[1,0,0]]
    gin_to_final.inputs.transform_parameters = [(0.1,)]
    gin_to_final.inputs.metric = ['MI']
    gin_to_final.inputs.radius_or_number_of_bins = [64]
    gin_to_final.inputs.shrink_factors = [[8,4,2,1]]
    gin_to_final.inputs.output_warped_image = False
    gin_to_final.inputs.smoothing_sigmas = [[3,2,1,0]]
    gin_to_final.inputs.num_threads = 8
    gin_to_final.inputs.number_of_iterations = [[1000,500,250,125]]
    gin_to_final.inputs.sampling_strategy = ['Regular']
    gin_to_final.inputs.sampling_percentage = [0.25]
    gin_to_final.inputs.output_transform_prefix = "cropped_to_source_"
    gin_to_final.inputs.verbose = True
    gin_to_final.inputs.winsorize_lower_quantile = 0.0
    gin_to_final.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(crop, 'cropped',
                     gin_to_final, 'fixed_image')
    workflow.connect(datagrabber, "GIN",
                     gin_to_final, 'moving_image')
    
    def combine_transforms(transform1, transform2):
        combined_transforms = [transform1[0], transform2[0]]
        return combined_transforms

    combine_transforms_node = Node(Function(input_names=['transform1', 'transform2'],
                                            output_names=['combined_transforms'],
                                            function=combine_transforms),
                                name='combine_transforms_node')

    workflow.connect(crop_to_main, 'forward_transforms', combine_transforms_node, 'transform2')
    workflow.connect(gin_to_final, 'forward_transforms', combine_transforms_node, 'transform1')


    # write brain seg on main in native space
    mask_to_main = Node(ants.ApplyTransforms(), name="mask_to_main")
    mask_to_main.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_main, 'forward_transforms', mask_to_main, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_main, 'input_image')
    workflow.connect(datagrabber, "T1", mask_to_main, 'reference_image')

    # write original image into main crop space
    main_to_mask = Node(ants.ApplyTransforms(), name="main_to_mask")
    main_to_mask.inputs.invert_transform_flags = [True, True]
    # itere value
    main_to_mask.iterables = ('interpolation', ['Linear', 'HammingWindowedSinc'])
    workflow.connect(combine_transforms_node, 'combined_transforms', main_to_mask, 'transforms')
    workflow.connect(datagrabber, "T1", main_to_mask, 'input_image')
    workflow.connect(datagrabber, 'GIN', main_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    final_norm = Node(Normalization(percentile = 99), name="final_intensity_normalization")
    workflow.connect(main_to_mask, 'output_image',
                     final_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     final_norm, 'brain_mask')
    
    predict = Node(Predict(), name="predict")
    predict.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    predict.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
    predict.inputs.model = '/mnt/model'
    predict.inputs.snglrt_enable_nvidia = True
    predict.inputs.descriptor = kwargs['PVS_DESCRIPTOR']
    predict.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'
    predict.inputs.out_filename = 'map.nii.gz'

    workflow.connect(final_norm, "intensity_normalized", predict, 't1')

    # warp back on seg PVS or CMB native space
    seg_to_native = Node(ants.ApplyTransforms(), name="seg_to_native")
    seg_to_native.inputs.interpolation = 'NearestNeighbor'
    seg_to_native.inputs.invert_transform_flags = [True]

    workflow.connect(crop_to_main, 'forward_transforms',
                     seg_to_native, 'transforms')
    workflow.connect(predict, 'segmentation',
                    seg_to_native, 'input_image')
    workflow.connect(datagrabber, 'T1',
                     seg_to_native, 'reference_image')
    

    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = 'output'
    workflow.connect(predict, 'segmentation', datasink, 'segmentation')
    workflow.connect(seg_to_native, 'output_image', datasink, 'segmentation_original_space')
    workflow.connect(post_brain_mask, 'segmentation', datasink, 'brain_mask')
    workflow.connect(final_norm, 'intensity_normalized', datasink, 'img_cropped_normalize')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')