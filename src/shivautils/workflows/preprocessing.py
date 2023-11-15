#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing, with conformation and preparation before AI segmentation.
    Defacing of native and final images. This also handles back-registration from
    conformed-crop to T1 or SWI ('img1').

    Its datagrabber requires to be connected to an external 'subject_id' from an iterable
"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber

from shivautils.interfaces.image import (Threshold, Normalization,
                                         Conform, Crop, Resample_from_to)
from shivautils.interfaces.shiva import Predict, PredictSingularity


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def as_list(input):
    return [input]


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    "conformed" = (256, 256, 256) 1x1x1mm3
    "preconformed" = (160, 214, 176) = pred model dim (no specific resolution in mm3)
    "unpreconformed" = "preconformed" sent in "conformed" space
    Returns:
        workflow
    """
    wf_name = 'shiva_preprocessing'
    if 'wf_name' in kwargs.keys():
        wf_name = kwargs['wf_name']
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on

    # file selection
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1', 'img2', 'img3', 'mask']),
        name='datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'

    # conform img1 to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.orientation = kwargs['ORIENTATION']

    workflow.connect(datagrabber, "img1", conform, 'img')

    # preconform img1 to resampled image in model-imposed dim while keeping fov (for 1st brain mask)
    preconform = Node(Conform(),
                      name="preconform")
    preconform.inputs.dimensions = kwargs['IMAGE_SIZE']  # = final dim = 160x214x176
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "img1", preconform, 'img')

    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    if kwargs['CONTAINERIZE_NODES']:
        pre_brain_mask = Node(PredictSingularity(), name="pre_brain_mask")
        pre_brain_mask.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
        pre_brain_mask.inputs.out_filename = '/mnt/data/pre_brain_mask.nii.gz'
        pre_brain_mask.inputs.snglrt_enable_nvidia = True
        pre_brain_mask.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        pre_brain_mask = Node(Predict(), "pre_brain_mask")
        pre_brain_mask.inputs.out_filename = 'pre_brain_mask.nii.gz'
    pre_brain_mask.inputs.model = kwargs['MODELS_PATH']

    if kwargs['MASK_ON_GPU']:
        if kwargs['GPU'] is not None:
            pre_brain_mask.inputs.gpu_number = kwargs['GPU']
    else:
        pre_brain_mask.inputs.gpu_number = -1

    plugin_args = kwargs['PRED_PLUGIN_ARGS']['sbatch_args']
    if '--gpus' in plugin_args and not kwargs['MASK_ON_GPU']:
        list_args = plugin_args.split(' ')
        gpu_arg_ind1 = list_args.index('--gpus')
        gpu_arg_ind2 = gpu_arg_ind1 + 1
        list_args_noGPU = [arg for i, arg in enumerate(list_args) if i not in [gpu_arg_ind1, gpu_arg_ind2]]
        plugin_args = ' '.join(list_args_noGPU)

    pre_brain_mask.plugin_args = plugin_args
    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')

    # send mask from preconformed space to
    # conformed space 256 256 256, same as anatomical conformed image
    unpreconform = Node(Conform(),
                        name="unpreconform")
    unpreconform.inputs.dimensions = (256, 256, 256)
    unpreconform.inputs.voxel_size = kwargs['RESOLUTION']
    unpreconform.inputs.orientation = 'RAS'

    workflow.connect(pre_brain_mask, 'segmentation', unpreconform, 'img')

    # binarize unpreconformed brain mask
    hard_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="hard_brain_mask")
    hard_brain_mask.inputs.binarize = True
    hard_brain_mask.inputs.open = 3  # morphological opening of clusters using a ball of radius 3
    hard_brain_mask.inputs.minVol = 30000  # Get rif of potential small clusters
    hard_brain_mask.inputs.clusterCheck = 'size'  # Select biggest cluster
    workflow.connect(unpreconform, 'resampled', hard_brain_mask, 'img')

    # normalize intensities between 0 and 1 for Tensorflow
    post_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="post_intensity_normalization")
    workflow.connect(conform, 'resampled',
                     post_normalization, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
                     post_normalization, 'brain_mask')

    # crop img1 centered on mask origin
    crop_normalized = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                           name="crop_normalized")
    workflow.connect(post_normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop_normalized, 'roi_mask')

    # crop raw
    # crop img1 centered on mask
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop, 'roi_mask')

    if kwargs['CONTAINERIZE_NODES']:
        post_brain_mask = Node(PredictSingularity(), name="post_brain_mask")
        post_brain_mask.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
        post_brain_mask.inputs.out_filename = '/mnt/data/post_brain_mask.nii.gz'
        post_brain_mask.inputs.snglrt_enable_nvidia = True
        post_brain_mask.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        post_brain_mask = Node(Predict(),  name="post_brain_mask")
        post_brain_mask.inputs.out_filename = 'post_brain_mask.nii.gz'
    post_brain_mask.inputs.model = kwargs['MODELS_PATH']

    if kwargs['MASK_ON_GPU']:
        if kwargs['GPU'] is not None:
            post_brain_mask.inputs.gpu_number = kwargs['GPU']
    else:
        post_brain_mask.inputs.gpu_number = -1

    post_brain_mask.plugin_args = plugin_args
    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 't1')

    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="hard_post_brain_mask")
    hard_post_brain_mask.inputs.binarize = True
    hard_post_brain_mask.inputs.open = 3
    hard_post_brain_mask.inputs.minVol = 30000
    hard_post_brain_mask.inputs.clusterCheck = 'size'
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    # brain seg from img1 back to native space
    mask_to_img1 = Node(Resample_from_to(), name="mask_to_img1")
    mask_to_img1.inputs.spline_order = 0  # should be equivalent to NearestNeighbor(?)
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_img1, 'moving_image')
    workflow.connect(datagrabber, "img1", mask_to_img1, 'fixed_image')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 1)
    img1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img1_final_intensity_normalization")
    workflow.connect(crop, 'cropped',
                     img1_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     img1_norm, 'brain_mask')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
