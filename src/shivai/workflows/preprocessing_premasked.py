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

    # Initilazing the wf
    workflow = gen_preproc_wf(**kwargs)

    # Chaing the connection, using img1 as mask
    datagrabber = workflow.get_node('datagrabber')
    mask_to_conform = workflow.get_node('mask_to_conform')
    workflow.disconnect(datagrabber, 'seg', mask_to_conform, 'moving_image')
    workflow.connect(datagrabber, 'img1', mask_to_conform, 'moving_image')

    return workflow
