#!/usr/bin/env python
"""
    Nipype workflow derived from preprocessing_synthseg.py, swapping the 
    synthseg node by a datagrabber that get the precomputed synthseg parc
"""
import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from shivai.workflows.preprocessing_synthseg import genWorkflow as gen_synthseg_wf


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg precomputed segmentation
    It is initialized with gen_synthseg_wf

    Required external input connections:
        synthseg_grabber.subject_id

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_fs'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_fs'

    kwargs['SYNTHSEG_IMAGE'] = kwargs['CONTAINER_IMAGE']  # Dummy image because unused here

    # Initilazing the wf with the synthseg_wf (to get the segmentation preproc)
    workflow = gen_synthseg_wf(**kwargs)

    datagrabber = workflow.get_node('datagrabber')

    # Rewiring the workflow with the datagrabber
    synthseg = workflow.get_node('synthseg')
    seg_cleaning = workflow.get_node('seg_cleaning')
    workflow.disconnect(synthseg, 'segmentation', seg_cleaning, 'input_seg')
    workflow.remove_nodes([synthseg])
    workflow.connect(datagrabber, 'seg', seg_cleaning, 'input_seg')

    return workflow
