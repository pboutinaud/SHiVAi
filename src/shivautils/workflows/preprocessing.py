#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing, with conformation and preparation before AI segmentation.
    Defacing of native and final images. This also handles back-registration from
    conformed-crop to T1.
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


def make_output_dict(sub_list,
                     t1_preproc_list,
                     brainmask_list,
                     pre_brainmask_list,
                     T1_cropped_list,
                     T1_conform_list,
                     BBOX1_list,
                     BBOX2_list,
                     CDG_IJK_list,
                     sum_preproc_wf_list,
                     flair_preproc_list=None,
                     swi_preproc_list=None
                     ):
    """Takes the list participants, all needed lists of preproc data
    and put them in a dict to pass to the next workflow. 
    """
    if flair_preproc_list is None:
        flair_preproc_list = [None]*len(sub_list)
    if swi_preproc_list is None:
        swi_preproc_list = [None]*len(sub_list)
    out_dict = {sub: {'t1': t1,
                      'brainmask': brainmask,
                      'pre_brainmask': pre_brainmask,
                      'T1_cropped': T1_cropped,
                      'T1_conform': T1_conform,
                      'BBOX1': BBOX1,
                      'BBOX2': BBOX2,
                      'CDG_IJK': CDG_IJK,
                      'sum_preproc_wf': sum_preproc_wf,
                      'flair': flair,
                      'swi': swi
                      } for (sub,
                             t1,
                             brainmask,
                             pre_brainmask,
                             T1_cropped,
                             T1_conform,
                             BBOX1,
                             BBOX2,
                             CDG_IJK,
                             sum_preproc_wf,
                             flair,
                             swi
                             ) in zip(sub_list,
                                      t1_preproc_list,
                                      brainmask_list,
                                      pre_brainmask_list,
                                      T1_cropped_list,
                                      T1_conform_list,
                                      BBOX1_list,
                                      BBOX2_list,
                                      CDG_IJK_list,
                                      sum_preproc_wf_list,
                                      flair_preproc_list,
                                      swi_preproc_list)}
    return out_dict


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    "conformed" = (256, 256, 256) 1x1x1mm3
    "preconformed" = (160, 214, 176) = pred model dim (no specific resolution in mm3)
    "unpreconformed" = "preconfored" sent in "conformed" space
    Returns:
        workflow
    """
    wf_name = 'shiva_mono_preprocessing'
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['t1']),
        name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'
    datagrabber.inputs.template_args = {'main': [['subject_id', 'main']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    # conform t1 to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.orientation = kwargs['ORIENTATION']

    workflow.connect(datagrabber, "t1", conform, 'img')

    # preconform t1 to resampled image in model-imposed dim while keeping fov (for 1st brain mask)
    preconform = Node(Conform(),
                      name="preconform")
    preconform.inputs.dimensions = kwargs['IMAGE_SIZE']  # = final dim = 160x214x176
    preconform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, "t1", preconform, 'img')

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

    # Prepare output for connection with next workflow
    preproc_out_node = JoinNode(
        Function(
            input_names=['sub_list', 't1_preproc_list'],
            output_names='preproc_out_dict',
            function=make_output_dict),
        name='preproc_out_node',
        joinsource=subject_list,
        joinfield=['sub_list', 't1_preproc_list']
    )
    workflow.connect(subject_list, 'subject_id', preproc_out_node, 'sub_list')
    workflow.connect(t1_norm, 'intensity_normalized', preproc_out_node, 't1_preproc_list')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
