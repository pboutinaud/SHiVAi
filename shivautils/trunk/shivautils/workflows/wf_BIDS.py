"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow, MapNode
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces import utility
from pathlib import Path

from shivautils.interfaces.shiva import PredictDirect

dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("test_workflow_BIDS")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    bg = Node(DataGrabber(infields=['subject_id'],
                          outfields=['T1', 'FLAIR']),
              name='BIDSdataGrabber')
    bg.inputs.raise_on_empty = True
    bg.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', bg, 'subject_id')

    # Map node for T1 image
    mapnode_t1 = MapNode(utility.IdentityInterface(fields=['T1']),
                         iterfield=['T1'], name='mapnode_T1')

    # Map node for FLAIR image
    mapnode_flair = MapNode(utility.IdentityInterface(fields=['FLAIR']),
                            iterfield=['FLAIR'], name='mapnode_FLAIR')

    # Connections
    workflow.connect(bg, 'T1', mapnode_t1, 'T1')
    workflow.connect(bg, 'FLAIR', mapnode_flair, 'FLAIR')

    return workflow
