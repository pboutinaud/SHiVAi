#!/usr/bin/env python

"""
Creates a workflow to generate a brain mask based on the main (T1) image, using our models

Requires an external connection for the input image:
wf.connect(datagrabber, "img1", preconform, 'img') 
"""

from nipype.pipeline.engine import Node, Workflow

from shivai.interfaces.image import (Threshold, Normalization,
                                     Conform, Crop)
from shivai.interfaces.shiva import Predict, PredictSingularity
import os


def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow(name='brain_mask_creation')

    # preconform img1 to resampled image in model-imposed dim while keeping fov (for 1st brain mask)
    preconform = Node(Conform(),
                      name="preconform")
    preconform.inputs.dimensions = kwargs['IMAGE_SIZE']  # = final dim = 160x214x176
    preconform.inputs.orientation = 'RAS'
    preconform.inputs.correction_threshold = kwargs['AFFINE_CORREC_THRESHOLD']

    # workflow.connect(datagrabber, "img1", preconform, 'img')  # External connection

    # normalize intensities between 0 and 1 for Tensorflow initial (rough) brain mask:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    # First prediction node for rough mask creation
    descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    if kwargs['CONTAINERIZE_NODES']:
        pre_brain_mask = Node(PredictSingularity(), name="pre_brain_mask")
        pre_brain_mask.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
        if kwargs['MODELS_PATH'] not in descriptor:
            descriptor_dir = os.path.dirname(descriptor)
            pre_brain_mask.inputs.snglrt_bind.append((descriptor_dir, descriptor_dir, 'ro'))
        pre_brain_mask.inputs.out_filename = '/mnt/data/pre_brain_mask.nii.gz'
        pre_brain_mask.inputs.snglrt_enable_nvidia = True
        pre_brain_mask.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        pre_brain_mask = Node(Predict(), "pre_brain_mask")
        pre_brain_mask.inputs.out_filename = 'pre_brain_mask.nii.gz'
    pre_brain_mask.inputs.model = kwargs['MODELS_PATH']
    if isinstance(kwargs['GPU'], int):
        pre_brain_mask.inputs.gpu_number = kwargs['GPU']

    plugin_args = kwargs['PRED_PLUGIN_ARGS']

    if not kwargs['BRAIN_SEG'] == 'shiva_gpu':
        pre_brain_mask.inputs.use_cpu = kwargs['AI_THREADS']
        if '--gpus' in plugin_args:
            # Remove the '--gpus' from the plugin args if necessary
            list_args = plugin_args.split(' ')
            gpu_arg_ind1 = list_args.index('--gpus')
            gpu_arg_ind2 = gpu_arg_ind1 + 1
            list_args_noGPU = [arg for i, arg in enumerate(list_args) if i not in [gpu_arg_ind1, gpu_arg_ind2]]
            plugin_args = ' '.join(list_args_noGPU)

    pre_brain_mask.plugin_args = plugin_args
    pre_brain_mask.inputs.descriptor = descriptor

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')

    # conform premask to 256 256 256, same as anatomical conformed image (works even on cropped input like shiva mask)
    conform_premask = Node(Conform(),
                           name="conform_premask")
    conform_premask.inputs.dimensions = (256, 256, 256)
    conform_premask.inputs.voxel_size = kwargs['RESOLUTION']
    conform_premask.inputs.voxels_tolerance = kwargs['TOLERANCE']
    conform_premask.inputs.orientation = kwargs['ORIENTATION']
    conform_premask.inputs.order = 0
    conform_premask.inputs.ignore_bad_affine = True  # The previous conform breaks the affine, but we don't care here

    workflow.connect(pre_brain_mask, 'segmentation', conform_premask, 'img')

    # binarize rough brain mask
    binarize_premask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="binarize_premask")
    binarize_premask.inputs.binarize = True
    binarize_premask.inputs.open = 3  # morphological opening of clusters using a ball of radius 3
    binarize_premask.inputs.minVol = 30000  # Get rif of potential small clusters
    binarize_premask.inputs.clusterCheck = 'size'  # Select biggest cluster
    workflow.connect(conform_premask, 'resampled', binarize_premask, 'img')

    # normalize intensities between 0 and 1 for Tensorflow using the rough mask
    intensity_norm_with_premask = Node(Normalization(percentile=kwargs['PERCENTILE']), name="intensity_norm_with_premask")
    workflow.connect(binarize_premask, 'thresholded',
                     intensity_norm_with_premask, 'brain_mask')
    # Requires external connection from conformed img1 to intensity_norm_with_premask.input_image

    # crop img1 centered on premask
    crop_with_premask = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                             name="crop_with_premask")
    workflow.connect(intensity_norm_with_premask, 'intensity_normalized',
                     crop_with_premask, 'apply_to')
    workflow.connect(binarize_premask, 'thresholded',
                     crop_with_premask, 'roi_mask')

    # New better mask creation
    if kwargs['CONTAINERIZE_NODES']:
        proper_brain_mask = Node(PredictSingularity(), name="proper_brain_mask")
        proper_brain_mask.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
        proper_brain_mask.inputs.out_filename = '/mnt/data/brain_mask.nii.gz'
        proper_brain_mask.inputs.snglrt_enable_nvidia = True
        proper_brain_mask.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        proper_brain_mask = Node(Predict(),  name="proper_brain_mask")
        proper_brain_mask.inputs.out_filename = 'brain_mask.nii.gz'
    proper_brain_mask.inputs.model = kwargs['MODELS_PATH']

    if not kwargs['BRAIN_SEG'] == 'shiva_gpu':
        proper_brain_mask.inputs.use_cpu = kwargs['AI_THREADS']

    if isinstance(kwargs['GPU'], int):
        proper_brain_mask.inputs.gpu_number = kwargs['GPU']

    proper_brain_mask.plugin_args = plugin_args
    proper_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    workflow.connect(crop_with_premask, 'cropped',
                     proper_brain_mask, 't1')

    # WF ENDPOINT: proper_brain_mask.segmentation
    return workflow
