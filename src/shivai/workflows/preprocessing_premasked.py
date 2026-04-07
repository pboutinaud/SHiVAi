#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing for nifti file on which the brain has already been masked (brain-extracted images),
    with conformation and preparation before AI segmentation.

    Its datagrabber requires to be connected to an external 'subject_id' from an iterable
"""

from nipype.pipeline.engine import Workflow

from shivai.workflows.preprocessing import genWorkflow as gen_preproc_wf


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for shiva preprocessing with nifti file on which the brain has already been masked

    Returns:
        workflow
    """
    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_premasked'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_premasked'

    # Initializing the wf
    workflow = gen_preproc_wf(**kwargs)

    # Changing the connection, using img1 as mask
    mask_to_conform = workflow.get_node('mask_to_conform')
    corrected_affine_img1 = workflow.get_node('correct_affine_img1')
    correct_affine_seg = workflow.get_node('correct_affine_seg')
    workflow.remove_nodes([correct_affine_seg])
    workflow.connect(corrected_affine_img1, 'corrected_img', mask_to_conform, 'moving_image')

    return workflow
