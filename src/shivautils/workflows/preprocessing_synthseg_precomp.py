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
    It is initialized with gen_synthseg_wf

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_synthseg_precomp'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_precomp'

    # Initilazing the wf
    workflow = gen_synthseg_wf(**kwargs)

    # Create the datagrabber node that will replace the synthseg node
    synthseg_grabber = Node(DataGrabber(infields=['subject_id'],
                                        outfields=['segmentation', 'qc', 'volumes']),
                            name='synthseg_grabber')
    synthseg_grabber.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results', 'shiva_preproc', 'synthseg')
    synthseg_grabber.inputs.raise_on_empty = True
    synthseg_grabber.inputs.sort_filelist = True
    synthseg_grabber.inputs.template = '%s/%s/*.nii*'
    synthseg_grabber.inputs.field_template = {'segmentation': '%s/synthseg_parc.nii*'}  # add 'qc' and 'volumes' here if needed
    synthseg_grabber.inputs.template_args = {'segmentation': [['subject_id']]}

    # Rewiring the workflow with the new nodes
    synthseg = workflow.get_node('synthseg')
    seg_cleaning = workflow.get_node('seg_cleaning')
    workflow.disconnect(synthseg, 'segmentation', seg_cleaning, 'input_seg')
    workflow.remove_nodes([synthseg])
    workflow.connect(synthseg_grabber, 'segmentation', seg_cleaning, 'input_seg')

    return workflow
