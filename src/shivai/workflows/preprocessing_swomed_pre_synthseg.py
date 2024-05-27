#!/usr/bin/env python
"""
    Nipype workflow derived from preprocessing_synthseg.py, swapping the
    synthseg node by a datagrabber that get the precomputed synthseg parc
"""
import os
from nipype.pipeline.engine import Node, Workflow
from shivautils.interfaces.shiva import Direct_File_Provider
from shivautils.workflows.preprocessing_synthseg import genWorkflow as gen_synthseg_wf


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg precomputed segmentation
    Used when the synthseg parcellation is directly given as a path, like when using SWOMed
    It is initialized with gen_synthseg_wf
    Removes unused parts of the wf
    Replaces the datagrabber with and identity interface (but keeps the name 'datagrabber')
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

    # Original file selector and synthseg
    datagrabber = workflow.get_node('datagrabber')
    synthseg = workflow.get_node('synthseg')

    # Replacement

    # List storing the reconnections to avoid doing it while iterating on the graph edges
    reconnections = []
    for _, connected_node, connection_dict in workflow._graph.out_edges(datagrabber, data=True):
        out_and_in = connection_dict['connect']
        for grab_out, node_in in out_and_in:
            reconnections.append((grab_out, connected_node, node_in))

    # Removing unused nodes
    workflow.remove_nodes([datagrabber, synthseg])

    # Datagrabber replacement with swomed input
    files_plug = Node(Direct_File_Provider(), name='datagrabber')

    # Rewiring the workflow with the new nodes
    for grabber_out, connected_node, node_in in reconnections:
        if connected_node is synthseg:
            continue
        workflow.connect(files_plug, grabber_out,
                         connected_node, node_in)

    # Connecting the precomputed Synthseg parc to the wf
    seg_cleaning = workflow.get_node('seg_cleaning')
    workflow.connect(files_plug, 'seg', seg_cleaning, 'input_seg')

    return workflow
