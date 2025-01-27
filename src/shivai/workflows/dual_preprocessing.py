#!/usr/bin/env python
"""Add accessory image (flair) co-registration to cropped space (through ANTS),
   and defacing of native and final images.
   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.quickshear import Quickshear

from shivai.utils.misc import as_list

from shivai.interfaces.image import Normalization, Conform
from shivai.interfaces.shiva import (AntsRegistration_Singularity,
                                     AntsApplyTransforms_Singularity,
                                     Quickshear_Singularity)
from shivai.workflows.qc_preproc import qc_wf_add_flair


def graft_img2_preproc(workflow: Workflow, **kwargs):
    """Add FLAIR preprocessing to a T1-only workflow
    Doing it this way allows to do all choices relative to brain segmentation
    in the T1 preproc wf, and just add the FLAIR preproc afterward as needed 
    (mutate the workflow)

    Returns:
        workflow
    """

    # file selection
    datagrabber = workflow.get_node('datagrabber')

    # Conform img2, should not be necessary but allows for the centering
    # of the origin of the nifti image (if far out of the brain)
    conform_flair = Node(Conform(),
                         name='conform_flair')
    conform_flair.inputs.dimensions = (256, 256, 256)
    conform_flair.inputs.voxel_size = kwargs['RESOLUTION']
    conform_flair.inputs.orientation = kwargs['ORIENTATION']

    crop = workflow.get_node('crop')
    img1_norm = workflow.get_node('img1_final_intensity_normalization')
    mask_to_crop = workflow.get_node('mask_to_crop')

    workflow.connect(datagrabber, 'img2',
                     conform_flair, 'img')

    # # write mask to flair in conformed space  # TODO: add it back maybe
    # if kwargs['CONTAINERIZE_NODES']:
    #     mask_to_img2 = Node(AntsApplyTransforms_Singularity(), name='mask_to_img2')
    #     mask_to_img2.inputs.snglrt_bind = [
    #         (kwargs['DATA_DIR'], kwargs['DATA_DIR'], 'ro'),
    #         (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
    #         ('`pwd`', '`pwd`', 'rw'),]
    #     mask_to_img2.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    # else:
    #     mask_to_img2 = Node(ants.ApplyTransforms(), name="mask_to_img2")
    # mask_to_img2.inputs.float = True
    # mask_to_img2.inputs.out_postfix = '_flair-space'
    # mask_to_img2.inputs.interpolation = 'NearestNeighbor'
    # mask_to_img2.inputs.invert_transform_flags = [True]

    # workflow.connect(flair_to_t1, 'forward_transforms',
    #                  mask_to_img2, 'transforms')
    # workflow.connect(mask_to_crop, 'resampled_image',
    #                  mask_to_img2, 'input_image')
    # workflow.connect(conform_flair, 'resampled',
    #                  mask_to_img2, 'reference_image')

    # Defacing the conformed image (uses the conformed mask from the 'unpreconform' node)
    if kwargs['CONTAINERIZE_NODES']:
        defacing_flair = Node(Quickshear_Singularity(), name="defacing_flair")
        defacing_flair.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        defacing_flair.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '`pwd`', 'rw')]  # TODO: See if this works
    else:
        defacing_flair = Node(Quickshear(),
                              name='defacing_flair')

    if kwargs['PREP_SETTINGS']['prereg_flair']:  # FLAIR already registered
        workflow.connect(conform_flair, 'resampled', defacing_flair, 'in_file')
    else:
        # compute 6-dof coregistration parameters of accessory scan
        # to cropped t1 image
        if kwargs['CONTAINERIZE_NODES']:
            flair_to_t1 = Node(AntsRegistration_Singularity(), name="flair_to_t1")
            flair_to_t1.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '`pwd`', 'rw'),]
            flair_to_t1.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            flair_to_t1 = Node(ants.Registration(),
                               name='flair_to_t1')
        flair_to_t1.inputs.float = True
        flair_to_t1.inputs.output_transform_prefix = "flair_to_t1_"
        flair_to_t1.plugin_args = kwargs['REG_PLUGIN_ARGS']
        flair_to_t1.inputs.transforms = ['Rigid']
        flair_to_t1.inputs.transform_parameters = [(0.1,)]
        flair_to_t1.inputs.metric = ['MI']
        flair_to_t1.inputs.radius_or_number_of_bins = [64]
        flair_to_t1.inputs.interpolation = kwargs['INTERPOLATION']
        flair_to_t1.inputs.shrink_factors = [[8, 4, 2, 1]]
        flair_to_t1.inputs.output_warped_image = True
        flair_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
        flair_to_t1.inputs.num_threads = 8
        flair_to_t1.inputs.number_of_iterations = [[1000, 500, 250, 125]]
        flair_to_t1.inputs.sampling_strategy = ['Regular']
        flair_to_t1.inputs.sampling_percentage = [0.25]
        flair_to_t1.inputs.verbose = True
        flair_to_t1.inputs.winsorize_lower_quantile = 0.005
        flair_to_t1.inputs.winsorize_upper_quantile = 0.995

        workflow.connect(conform_flair, 'resampled',
                         flair_to_t1, 'moving_image')

        workflow.connect(crop, 'cropped',
                         flair_to_t1, 'fixed_image')

        workflow.connect(mask_to_crop, ('resampled_image', as_list),
                         flair_to_t1, 'fixed_image_masks')

        workflow.connect(flair_to_t1, 'warped_image', defacing_flair, 'in_file')

    workflow.connect(mask_to_crop, 'resampled_image', defacing_flair, 'mask_file')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 2)
    img2_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img2_final_intensity_normalization")
    workflow.connect(defacing_flair, 'out_file',
                     img2_norm, 'input_image')
    workflow.connect(mask_to_crop, 'resampled_image',
                     img2_norm, 'brain_mask')

    # QC
    qc_wf = workflow.get_node('preproc_qc_workflow')
    workflow.connect(img2_norm, 'mode', qc_wf, 'qc_metrics.flair_norm_peak')
    if not kwargs['PREP_SETTINGS']['prereg_flair']:
        qc_wf = qc_wf_add_flair(qc_wf)
        workflow.connect(img2_norm, 'intensity_normalized', qc_wf, 'qc_coreg_FLAIR_T1.path_image')
        workflow.connect(img1_norm, 'intensity_normalized', qc_wf, 'qc_coreg_FLAIR_T1.path_ref_image')
        workflow.connect(mask_to_crop, 'resampled_image', qc_wf, 'qc_coreg_FLAIR_T1.path_brainmask')
        workflow.connect(flair_to_t1, 'forward_transforms', qc_wf, 'qc_metrics.flair_reg_mat')
