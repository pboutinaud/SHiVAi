#!/usr/bin/env python
"""
    Nipype workflow derived from preprocessing_synthseg.py, swapping the 
    synthseg node by a datagrabber that get the precomputed synthseg parc
"""
import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from shivautils.workflows.preprocessing_synthseg import genWorkflow as gen_synthseg_wf


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg precomputed segmentation
    Used when the synthseg parcellation is directly given as a path, like when using SWOMed
    It is initialized with gen_synthseg_wf
    Removes unused parts of the wf
    Needs outside connection from the parcellation path to (seg_cleaning, 'input_seg')

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_synthseg_precomp'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_precomp'

    # Initilazing the wf
    workflow = gen_synthseg_wf(**kwargs)

    # Rewiring the workflow with the new nodes
    synthseg = workflow.get_node('synthseg')
    seg_cleaning = workflow.get_node('seg_cleaning')
    workflow.disconnect(synthseg, 'segmentation', seg_cleaning, 'input_seg')
    workflow.remove_nodes([synthseg])

    return workflow
