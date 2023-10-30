#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing, with conformation and preparation before AI segmentation.
    Defacing of native and final images. This also handles back-registration from
    conformed-crop to T1 or SWI ('img1').

    Its datagrabber requires to be connected to an outsite 'subject_id' from an iterable
"""
import os

from nipype.pipeline.engine import Node, JoinNode, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.image import (Threshold, Normalization,
                                         Conform, Crop)
from shivautils.interfaces.shiva import PredictSingularity, Predict


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
    "unpreconformed" = "preconfored" sent in "conformed" space
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
        outfields=['img1', 'img2', 'img3']),  # TODO: swi = img3
        name='datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'
    datagrabber.inputs.template_args = {'img1': [['subject_id', 'img1']]}

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

    if kwargs['CONTAINER'] == True:  # TODO: Check with PY
        pre_brain_mask = Node(Predict(), "pre_brain_mask")
        pre_brain_mask.inputs.model = kwargs['MODELS_PATH']

    if kwargs['GPU'] is not None:
        pre_brain_mask.inputs.gpu_number = kwargs['GPU']

    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.out_filename = 'pre_brain_mask.nii.gz'

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

    if kwargs['CONTAINER'] == True:  # TODO: Check with PY
        post_brain_mask = Node(Predict(), "post_brain_mask")
        post_brain_mask.inputs.model = kwargs['MODELS_PATH']

    if kwargs['GPU'] is not None:
        post_brain_mask.inputs.gpu_number = kwargs['GPU']

    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    post_brain_mask.inputs.out_filename = 'post_brain_mask.nii.gz'
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 't1')

    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="hard_post_brain_mask")
    hard_post_brain_mask.inputs.binarize = True
    hard_post_brain_mask.inputs.open = 3
    hard_post_brain_mask.inputs.minVol = 30000
    hard_post_brain_mask.inputs.clusterCheck = 'size'
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    # compute 3-dof (translations) coregistration parameters of cropped to native img1
    crop_to_img1 = Node(ants.Registration(),
                        name='crop_to_img1')
    crop_to_img1.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}  # TODO: Check why it's used
    crop_to_img1.inputs.transforms = ['Rigid']
    crop_to_img1.inputs.restrict_deformation = [[1, 0, 0,], [1, 0, 0,], [1, 0, 0]]
    crop_to_img1.inputs.transform_parameters = [(0.1,)]
    crop_to_img1.inputs.metric = ['MI']
    crop_to_img1.inputs.radius_or_number_of_bins = [64]
    crop_to_img1.inputs.shrink_factors = [[8, 4, 2, 1]]
    crop_to_img1.inputs.output_warped_image = False
    crop_to_img1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    crop_to_img1.inputs.num_threads = 8
    crop_to_img1.inputs.number_of_iterations = [[1000, 500, 250, 125]]
    crop_to_img1.inputs.sampling_strategy = ['Regular']
    crop_to_img1.inputs.sampling_percentage = [0.25]
    crop_to_img1.inputs.output_transform_prefix = "cropped_to_source_"
    crop_to_img1.inputs.verbose = True
    crop_to_img1.inputs.winsorize_lower_quantile = 0.0
    crop_to_img1.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(datagrabber, "img1",
                     crop_to_img1, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_img1, 'moving_image')

    # write brain seg on img1 in native space
    mask_to_img1 = Node(ants.ApplyTransforms(), name="mask_to_img1")
    mask_to_img1.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_img1, 'forward_transforms', mask_to_img1, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_img1, 'input_image')
    workflow.connect(datagrabber, "img1", mask_to_img1, 'reference_image')

    # write original image into img1 crop space
    img1_to_mask = Node(ants.ApplyTransforms(), name="img1_to_mask")
    img1_to_mask.inputs.invert_transform_flags = [True]
    img1_to_mask.inputs.interpolation = kwargs['INTERPOLATION']

    workflow.connect(crop_to_img1, 'forward_transforms', img1_to_mask, 'transforms')
    workflow.connect(datagrabber, "img1", img1_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', img1_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    img1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img1_final_intensity_normalization")
    workflow.connect(img1_to_mask, 'output_image',
                     img1_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     img1_norm, 'brain_mask')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
