#!/usr/bin/env python
"""
Plugin workflow to add coregistration steps from SWI to T1 when doing CMB
segmentation while a T1 is available from another segmentation
"""
from nipype.interfaces import ants
from nipype.pipeline.engine import Node, Workflow
from shivautils.interfaces.image import Normalization, Conform, Crop, Resample_from_to


def gen_workflow_swi(**kwargs) -> Workflow:
    """
    Workflow for SWI preprocessing when doing CMB
    segmentation using the T1-defined mask from another segmentation preproc.
    Also uses the data grabber from the other workflow.
    It's basically a plugin of the T1 workflow


    external connections required: 
        full_wf.connect(wf1, 'datagrabber.img3', wf_swi, 'conform.img')
        full_wf.connect(wf1, 'crop.cropped', wf_swi, 'swi_to_t1.moving_image')
        full_wf.connect(wf1, ('hard_post_brain_mask.thresholded', lambda input: [input]), wf_swi, 'swi_to_t1.fixed_image_masks')
        full_wf.connect(wf1, 'hard_post_brain_mask.thresholded', wf_swi, 'mask_to_swi.input_image')
    Returns:
        workflow
    """

    wf_name = 'shiva_cmb_preprocessing'
    if 'wf_name' in kwargs.keys():
        wf_name = kwargs['wf_name']
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # Conforms the SWI image, must be connected to the datagrabber from the other workflow
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.orientation = kwargs['ORIENTATION']

    # compute 6-dof coregistration parameters of conformed swi
    # to t1 cropped image
    swi_to_t1 = Node(ants.Registration(),
                     name='swi_to_t1')
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

    workflow.connect(conform, 'resampled', swi_to_t1, 'moving_image')

    # Application of the t1 to swi transformation to the t1 mask
    mask_to_swi = Node(ants.ApplyTransforms(), name="mask_to_swi")
    mask_to_swi.inputs.interpolation = 'NearestNeighbor'
    mask_to_swi.inputs.invert_transform_flags = [True]
    workflow.connect(swi_to_t1, 'forward_transforms', mask_to_swi, 'transforms')
    workflow.connect(conform, 'resampled', mask_to_swi, 'reference_image')

    # Crop SWI image
    crop_swi = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                    name="crop_swi")
    workflow.connect(conform, 'resampled', crop_swi, 'apply_to')
    workflow.connect(mask_to_swi, 'output_image', crop_swi, 'roi_mask')

    # Conformed mask (256x256x256) to cropped space
    mask_to_crop = Node(Resample_from_to(),
                        name='mask_to_crop')
    mask_to_crop.inputs.spline_order = 0
    workflow.connect(mask_to_swi, 'output_image', mask_to_crop, 'moving_image')
    workflow.connect(crop_swi, 'cropped', mask_to_crop, 'fixed_image')

    # Intensity normalization of the cropped image for the segmentation (ENDPOINT)
    swi_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="swi_intensity_normalisation")
    workflow.connect(crop_swi, 'cropped', swi_norm, 'input_image')
    workflow.connect(mask_to_crop, 'resampled_image', swi_norm, 'brain_mask')

    return workflow
