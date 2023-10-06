#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing, with conformation and preparation before AI segmentation.
    Defacing of native and final images. This also handles back-registration from
    conformed-crop to T1.
"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

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

    Returns:
        workflow
    """
    wf_name = kwargs['WF_DIRS']['preproc']
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # Define if dual (FLAIR + T1) or T1 only
    if kwargs['PREDICTION'] == ['PVS']:
        dual = False
    elif 'PVS2' in kwargs['PREDICTION'] or 'WMH' in kwargs['PREDICTION']:
        dual = True

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    if dual:
        datagrabber = Node(DataGrabber(
            infields=['subject_id'],
            outfields=['t1', 'flair']),
            name='dataGrabber')
    else:
        datagrabber = Node(DataGrabber(
            infields=['subject_id'],
            outfields=['t1']),
            name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii.gz'
    if kwargs['INPUT_TYPE'] in ['standard', 'json']:
        if dual:
            datagrabber.inputs.field_template = {'t1': '%s/%s/*_raw.nii.gz',
                                                 'flair': '%s/%s/*_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 't1']],
                                                'flair': [['subject_id', 'flair']]}
        else:
            datagrabber.inputs.field_template = {'t1': '%s/%s/*_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 't1']]}

    if kwargs['INPUT_TYPE'] == 'BIDS':
        if dual:
            datagrabber.inputs.field_template = {'t1': '%s/anat/%s_T1_raw.nii.gz',
                                                 'flair': '%s/anat/%s_FLAIR_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']],
                                                'flair': [['subject_id', 'subject_id']]}
        else:
            datagrabber.inputs.field_template = {'t1': '%s/anat/%s_T1_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    # conform t1 to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.orientation = kwargs['ORIENTATION']

    workflow.connect(datagrabber, "t1", conform, 'img')

    # preconform t1 to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(),
                      name="preconform")
    preconform.inputs.dimensions = kwargs['IMAGE_SIZE']
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "t1", preconform, 'img')

    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile=kwargs['PERCENTILE']), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled', preconf_normalization, 'input_image')

    pre_brain_mask = Node(Predict(), "pre_brain_mask")
    pre_brain_mask.inputs.model = kwargs['MODELS_PATH']
    if kwargs['GPU'] is not None:
        pre_brain_mask.inputs.gpu_number = kwargs['GPU']

    pre_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    pre_brain_mask.inputs.out_filename = 'pre_brain_mask.nii.gz'

    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')

    # send mask from preconformed space to
    # conformed space 256 256 256 , same as anatomical conformed image
    unconform = Node(Conform(),
                     name="unpreconform")
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

    # crop t1 centered on mask origin
    crop_normalized = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                           name="crop_normalized")
    workflow.connect(post_normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop_normalized, 'roi_mask')

    # crop raw
    # crop t1 centered on mask
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop, 'roi_mask')

    post_brain_mask = Node(Predict(), "post_brain_mask")
    post_brain_mask.inputs.model = kwargs['MODELS_PATH']
    if kwargs['GPU'] is not None:
        post_brain_mask.inputs.gpu_number = kwargs['GPU'] 


    post_brain_mask.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    post_brain_mask.inputs.out_filename = 'post_brain_mask.nii.gz'
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 't1')


    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD'], binarize=True), name="hard_post_brain_mask")
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    if dual:
        # compute 6-dof coregistration parameters of accessory scan
        # to cropped t1 image
        coreg = Node(ants.Registration(),
                     name='coregister')
        coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
        coreg.inputs.transforms = ['Rigid']
        coreg.inputs.transform_parameters = [(0.1,)]
        coreg.inputs.metric = ['MI']
        coreg.inputs.radius_or_number_of_bins = [64]
        coreg.inputs.interpolation = 'WelchWindowedSinc'
        coreg.inputs.shrink_factors = [[8, 4, 2, 1]]
        coreg.inputs.output_warped_image = True
        coreg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
        coreg.inputs.num_threads = 8
        coreg.inputs.number_of_iterations = [[1000, 500, 250, 125]]
        coreg.inputs.sampling_strategy = ['Regular']
        coreg.inputs.sampling_percentage = [0.25]
        coreg.inputs.output_transform_prefix = "t1_to_flair_"
        coreg.inputs.verbose = True
        coreg.inputs.winsorize_lower_quantile = 0.005
        coreg.inputs.winsorize_upper_quantile = 0.995

        workflow.connect(datagrabber, "flair",
                         coreg, 'moving_image')
        workflow.connect(crop, 'cropped',
                         coreg, 'fixed_image')
        workflow.connect(hard_post_brain_mask, ('thresholded', as_list),
                         coreg, 'fixed_image_masks')

    # compute 3-dof (translations) coregistration parameters of cropped to native t1
    crop_to_t1 = Node(ants.Registration(),
                      name='crop_to_t1')
    crop_to_t1.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}  # TODO: Check why it's used
    crop_to_t1.inputs.transforms = ['Rigid']
    crop_to_t1.inputs.restrict_deformation = [[1, 0, 0,], [1, 0, 0,], [1, 0, 0]]
    crop_to_t1.inputs.transform_parameters = [(0.1,)]
    crop_to_t1.inputs.metric = ['MI']
    crop_to_t1.inputs.radius_or_number_of_bins = [64]
    crop_to_t1.inputs.shrink_factors = [[8, 4, 2, 1]]
    crop_to_t1.inputs.output_warped_image = False
    crop_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    crop_to_t1.inputs.num_threads = 8
    crop_to_t1.inputs.number_of_iterations = [[1000, 500, 250, 125]]
    crop_to_t1.inputs.sampling_strategy = ['Regular']
    crop_to_t1.inputs.sampling_percentage = [0.25]
    crop_to_t1.inputs.output_transform_prefix = "cropped_to_source_"
    crop_to_t1.inputs.verbose = True
    crop_to_t1.inputs.winsorize_lower_quantile = 0.0
    crop_to_t1.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(datagrabber, "t1",
                     crop_to_t1, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_t1, 'moving_image')

    # write brain seg on t1 in native space
    mask_to_t1 = Node(ants.ApplyTransforms(), name="mask_to_t1")
    mask_to_t1.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_t1, 'forward_transforms', mask_to_t1, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_t1, 'input_image')
    workflow.connect(datagrabber, "t1", mask_to_t1, 'reference_image')

    if dual:
        # write mask to flair in native space
        mask_to_flair = Node(ants.ApplyTransforms(), name="mask_to_flair")
        mask_to_flair.inputs.interpolation = 'NearestNeighbor'
        mask_to_flair.inputs.invert_transform_flags = [True]

        workflow.connect(coreg, 'forward_transforms',
                         mask_to_flair, 'transforms')
        workflow.connect(hard_post_brain_mask, 'thresholded',
                         mask_to_flair, 'input_image')
        workflow.connect(datagrabber, 'flair',
                         mask_to_flair, 'reference_image')

    # write original image into t1 crop space
    t1_to_mask = Node(ants.ApplyTransforms(), name="t1_to_mask")
    t1_to_mask.inputs.invert_transform_flags = [True]
    t1_to_mask.inputs.interpolation = kwargs['INTERPOLATION']

    workflow.connect(crop_to_t1, 'forward_transforms', t1_to_mask, 'transforms')
    workflow.connect(datagrabber, "t1", t1_to_mask, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded', t1_to_mask, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    t1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="t1_final_intensity_normalization")
    workflow.connect(t1_to_mask, 'output_image',
                     t1_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     t1_norm, 'brain_mask')

    if dual:
        # Intensity normalize coregistered image for tensorflow (ENDPOINT 2)
        flair_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="flair_final_intensity_normalization")
        workflow.connect(coreg, 'warped_image',
                         flair_norm, 'input_image')
        workflow.connect(hard_post_brain_mask, 'thresholded',
                         flair_norm, 'brain_mask')

    workflow.write_graph(graph2use='orig', dotfilename='graph.svg', format='svg')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
