#!/usr/bin/env python
"""
Plugin workflow to add coregistration steps from SWI to T1 when doing CMB
segmentation while a T1 is available from another segmentation
"""
from nipype.interfaces import ants
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.quickshear import Quickshear

from shivai.interfaces.image import Normalization, Conform, Crop, Resample_from_to
from shivai.workflows.qc_preproc import qc_wf_add_swi
from shivai.interfaces.shiva import (AntsRegistration_Singularity,
                                     AntsApplyTransforms_Singularity,
                                     Quickshear_Singularity)


def graft_workflow_swi(preproc_wf: Workflow, **kwargs):
    """
    Workflow for SWI preprocessing when doing CMB
    segmentation using the T1-defined mask from another segmentation preproc.
    Graft this subworkflow to the preprocessing workflown, uses the data grabber from the other workflow.
    It's basically a plugin of the T1 workflow
    (mutate the workflow)


    Required external input connections: 
        cmb_preprocessing.swi_intensity_normalisation.intensity_normalized
        cmb_preprocessing.swi_to_t1.forward_transforms
        cmb_preprocessing.mask_to_crop_swi.resampled_image
        cmb_preprocessing.crop_swi.bbox1_file
        cmb_preprocessing.crop_swi.bbox2_file
        cmb_preprocessing.crop_swi.cdg_ijk_file
    """

    wf_name = 'cmb_preprocessing'
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # Conforms the SWI image, must be connected to the datagrabber from the other workflow
    conform_swi = Node(Conform(),
                       name="conform_swi")
    conform_swi.inputs.dimensions = (256, 256, 256)
    conform_swi.inputs.voxel_size = kwargs['RESOLUTION']
    conform_swi.inputs.orientation = kwargs['ORIENTATION']

    # compute 6-dof coregistration parameters of conformed swi
    # to t1 cropped image
    if kwargs['CONTAINERIZE_NODES']:
        swi_to_t1 = Node(AntsRegistration_Singularity(), name='swi_to_t1')
        swi_to_t1.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '`pwd`', 'rw'),]
        swi_to_t1.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        swi_to_t1 = Node(ants.Registration(),
                         name='swi_to_t1')
    swi_to_t1.inputs.float = True
    swi_to_t1.plugin_args = kwargs['REG_PLUGIN_ARGS']
    swi_to_t1.inputs.transforms = ['Rigid']
    swi_to_t1.inputs.transform_parameters = [(0.1,)]
    swi_to_t1.inputs.metric = ['MI']
    swi_to_t1.inputs.radius_or_number_of_bins = [64]
    swi_to_t1.inputs.interpolation = 'WelchWindowedSinc'
    swi_to_t1.inputs.shrink_factors = [[8, 4, 2, 1]]
    swi_to_t1.inputs.output_warped_image = True
    swi_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    swi_to_t1.inputs.num_threads = 8
    swi_to_t1.inputs.number_of_iterations = [[1000, 500, 250, 125]]
    swi_to_t1.inputs.sampling_strategy = ['Regular']
    swi_to_t1.inputs.sampling_percentage = [0.25]
    swi_to_t1.inputs.output_transform_prefix = "swi_to_t1_"
    swi_to_t1.inputs.verbose = True
    swi_to_t1.inputs.winsorize_lower_quantile = 0.005
    swi_to_t1.inputs.winsorize_upper_quantile = 0.995

    workflow.connect(conform_swi, 'resampled', swi_to_t1, 'moving_image')

    # Application of the t1 to swi transformation to the t1 mask
    if kwargs['CONTAINERIZE_NODES']:
        mask_to_swi = Node(AntsApplyTransforms_Singularity(), name="mask_to_swi")
        mask_to_swi.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '`pwd`', 'rw'),]
        mask_to_swi.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
    else:
        mask_to_swi = Node(ants.ApplyTransforms(), name="mask_to_swi")
    mask_to_swi.inputs.float = True
    mask_to_swi.inputs.out_postfix = '_conformed_swi-space'
    mask_to_swi.inputs.interpolation = 'NearestNeighbor'
    mask_to_swi.inputs.invert_transform_flags = [True]
    workflow.connect(swi_to_t1, 'forward_transforms', mask_to_swi, 'transforms')
    workflow.connect(conform_swi, 'resampled', mask_to_swi, 'reference_image')

    # Defacing the conformed image
    if kwargs['CONTAINERIZE_NODES']:
        defacing_swi = Node(Quickshear_Singularity(), name="defacing_swi")
        defacing_swi.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        defacing_swi.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '`pwd`', 'rw')]  # TODO: See if this works
    else:
        defacing_swi = Node(Quickshear(),
                            name='defacing_swi')

    workflow.connect(conform_swi, 'resampled', defacing_swi, 'in_file')
    workflow.connect(mask_to_swi, 'output_image', defacing_swi, 'mask_file')

    # Crop SWI image
    crop_swi = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                    name="crop_swi")
    workflow.connect(defacing_swi, 'out_file', crop_swi, 'apply_to')
    workflow.connect(mask_to_swi, 'output_image', crop_swi, 'roi_mask')

    # Conformed mask (256x256x256) to cropped space
    mask_to_crop_swi = Node(Resample_from_to(),
                            name='mask_to_crop_swi')
    mask_to_crop_swi.inputs.spline_order = 0
    mask_to_crop_swi.inputs.out_name = 'brainmask_cropped_swi-space.nii.gz'
    workflow.connect(mask_to_swi, 'output_image', mask_to_crop_swi, 'moving_image')
    workflow.connect(crop_swi, 'cropped', mask_to_crop_swi, 'fixed_image')

    # Intensity normalization of the cropped image for the segmentation (ENDPOINT)
    swi_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="swi_intensity_normalisation")
    workflow.connect(crop_swi, 'cropped', swi_norm, 'input_image')
    workflow.connect(mask_to_crop_swi, 'resampled_image', swi_norm, 'brain_mask')

    # Adding the subworkflow to the main preprocessing workflow and connecting the nodes
    preproc_wf.add_nodes([workflow])
    datagrabber = preproc_wf.get_node('datagrabber')
    crop = preproc_wf.get_node('crop')
    mask_to_crop = preproc_wf.get_node('mask_to_crop')
    preproc_wf.connect(datagrabber, 'img3', workflow, 'conform_swi.img')
    preproc_wf.connect(crop, 'cropped', workflow, 'swi_to_t1.fixed_image')
    preproc_wf.connect(mask_to_crop, 'resampled_image', mask_to_swi, 'input_image')  # using "workflow, 'mask_to_swi.input_image'" does not work for some reason...

    # Adding SWI/CMB nodes to the QC sub-workflow and connecting the nodes
    qc_wf = preproc_wf.get_node('preproc_qc_workflow')
    qc_wf = qc_wf_add_swi(qc_wf)
    qc_overlay_brainmask_swi = qc_wf.get_node('qc_overlay_brainmask_swi')
    qc_metrics = qc_wf.get_node('qc_metrics')

    qc_wf.connect(mask_to_crop_swi, 'resampled_image', qc_overlay_brainmask_swi, 'brainmask')
    qc_wf.connect(swi_norm, 'intensity_normalized', qc_overlay_brainmask_swi, 'img_ref')
    qc_wf.connect(swi_norm, 'mode', qc_metrics, 'swi_norm_peak')
    qc_wf.connect(swi_to_t1, 'forward_transforms', qc_metrics, 'swi_reg_mat')
