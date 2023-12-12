#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing for nifti file on which the brain has already been masked (brain-extracted images),
    with conformation and preparation before AI segmentation.

    Its datagrabber requires to be connected to an external 'subject_id' from an iterable
"""

import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber

from shivautils.interfaces.image import (Normalization, Threshold,
                                         Conform, Crop, Resample_from_to)


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    "conformed" = (256, 256, 256) 1x1x1mm3

    Returns:
        workflow
    """
    wf_name = 'shiva_preprocessing_premasked'
    if 'wf_name' in kwargs.keys():
        wf_name = kwargs['wf_name']
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on

    # file selection
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1', 'img2', 'img3', 'brainmask']),
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

    # conform brainmask to 1 mm isotropic, freesurfer-style
    conform_mask = Node(Conform(),
                        name="conform_mask")
    conform_mask.inputs.dimensions = (256, 256, 256)
    conform_mask.inputs.voxel_size = kwargs['RESOLUTION']
    conform_mask.inputs.orientation = kwargs['ORIENTATION']
    conform_mask.inputs.order = 0

    workflow.connect(datagrabber, "brainmask", conform_mask, 'img')

    # crop img1 centered on mask
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(conform_mask, 'resampled',
                     crop, 'roi_mask')

    mask_to_crop = Node(Resample_from_to(),
                        name='mask_to_crop')
    mask_to_crop.inputs.spline_order = 0  # should be equivalent to NearestNeighbor(?)
    workflow.connect(conform_mask, 'resampled', mask_to_crop, 'moving_image')
    workflow.connect(crop, "cropped", mask_to_crop, 'fixed_image')

    # binarize brain mask, not strictly necessary, but makes the workflow compatible with others
    hard_post_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="hard_post_brain_mask")
    hard_post_brain_mask.inputs.binarize = True
    hard_post_brain_mask.inputs.clusterCheck = 'all'
    hard_post_brain_mask.inputs.outname = 'brainmask.nii.gz'
    workflow.connect(mask_to_crop, 'resampled_image', hard_post_brain_mask, 'img')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 1)
    img1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']),
                     name="img1_final_intensity_normalization")
    workflow.connect(crop, 'cropped',
                     img1_norm, 'input_image')
    workflow.connect(mask_to_crop, 'resampled_image',
                     img1_norm, 'brain_mask')

    return workflow
