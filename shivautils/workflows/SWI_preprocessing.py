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

from shivautils.interfaces.shiva import PredictSingularity, Predict
from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop, ApplyMask, MetricsPredictions,
                            JoinMetricsPredictions, SummaryReport)
from shivautils.isocontour_board import create_edges
from shivautils.stats import save_histogram


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_preprocessing_swi")
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
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, 'SWI', conform, 'img')

    # preconform main to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(), name="preconform")
    preconform.inputs.dimensions = kwargs['IMAGE_SIZE']
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "SWI", preconform, 'img')


    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    if kwargs['CONTAINER'] == True:
        pre_brain_mask = Node(Predict(), "pre_brain_mask")
        pre_brain_mask.inputs.model = kwargs['MODELS_PATH']
    else:
        # brain mask from tensorflow
        pre_brain_mask = Node(PredictSingularity(), "pre_brain_mask")
        pre_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 1 --partition GPU'}
        pre_brain_mask.inputs.snglrt_bind =  [
            (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
            ('`pwd`','/mnt/data','rw'),
            ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
            ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
            (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
        pre_brain_mask.inputs.model = '/mnt/model'
        pre_brain_mask.inputs.snglrt_enable_nvidia = True
        pre_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict.sif'
        
    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.out_filename = 'pre_brain_mask.nii.gz'
    
    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 'swi')
    
    # send mask from preconformed space to
    # conformed space 256 256 256, same as anatomical conformed image
    unconform = Node(Conform(), name="unpreconform")
    unconform.inputs.dimensions = (256, 256, 256)
    unconform.inputs.voxel_size = kwargs['RESOLUTION']
    unconform.inputs.orientation = 'RAS'

    workflow.connect(pre_brain_mask, 'segmentation', unconform, 'img')

    # binarize unpreconformed brain mask
    hard_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD'], binarize=True), name="hard_brain_mask")
    workflow.connect(unconform, 'resampled', hard_brain_mask, 'img')

    # normalize intensities between 0 and 1 for Tensorflow
    post_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="post_intensity_normalization")
    workflow.connect(conform, 'resampled',
                     post_normalization, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
                    post_normalization, 'brain_mask')
    
    # crop main centered on mask origin
    crop_normalized = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                           name="crop_normalized")
    workflow.connect(post_normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop_normalized, 'roi_mask')
    
    # crop raw
    # crop main centered on mask
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop, 'roi_mask')
    
    if kwargs['CONTAINER'] == True:
        post_brain_mask = Node(Predict(), "post_brain_mask")
        post_brain_mask.inputs.model = kwargs['MODELS_PATH']
    else:
        # brain mask from tensorflow
        post_brain_mask = Node(PredictSingularity(), "post_brain_mask")
        post_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 1 --partition GPU'}
        post_brain_mask.inputs.snglrt_bind =  [
            (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
            ('`pwd`','/mnt/data','rw'),
            ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
            ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
            (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
        post_brain_mask.inputs.model = '/mnt/model'
        post_brain_mask.inputs.snglrt_enable_nvidia = True
        post_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict.sif'
    
    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    post_brain_mask.inputs.out_filename = 'post_brain_mask.nii.gz'
    
    workflow.connect(crop_normalized, 'cropped',
                    post_brain_mask, 'swi')
    
    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD'], binarize=True), name="hard_post_brain_mask")
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
    main_to_mask.interpolation = kwargs['INTERPOLATION']

    workflow.connect(crop_to_main, 'forward_transforms', main_to_mask, 'transforms')
    workflow.connect(datagrabber, "SWI", main_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', main_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    final_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="final_intensity_normalization")
    workflow.connect(main_to_mask, 'output_image',
                     final_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     final_norm, 'brain_mask')
    
    workflow.write_graph(graph2use='orig', dotfilename='graph.svg', format='svg')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')