"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.shiva import Predict


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def get_input_files(subject_id, in_dict):
    t1 = in_dict[subject_id]['t1']
    flair = in_dict[subject_id]['flair']
    return t1, flair,


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow
    Most arguments are part of the wfargs in shiva.py
    But there is also "PRED" which may contains the type of prediction (PVS/PVS2/WMH/CMB)
    Returns:
        workflow
    """
    workflow = Workflow('predictor_workflow')
    workflow.base_dir = kwargs['BASE_DIR']

    subject_list = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # Data input possibilities: from WF or with Datagrabber:
    if 'SUB_WF' in kwargs.keys() and kwargs['SUB_WF']:  # From previous workflow
        input_node = Node(
            Function(
                input_names=['subject_id', 'in_dict'],
                output_names=['t1', 'flair'],  # flair unsued here
                function=get_input_files
            ),
            name='input_parser'
        )

    else:  # Using a datagrabber (requires additional settings outside of the wf)
        # get a list of subjects to iterate on

        # file selection
        input_node = Node(
            DataGrabber(
                infields=['subject_id'],
                outfields=['t1', 'flair']),  # flair unsued here
            name='dataGrabber')

        input_node.inputs.template = '%s/%s/*.nii*'
        input_node.inputs.raise_on_empty = True
        input_node.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', input_node, 'subject_id')

    if 'PRED' in kwargs.keys():  # Else need to be set outside of the wf
        PRED = kwargs['PRED']
        pred = PRED[:3].lower()  # biomarkers should have 3 letters
        predictor_node = Node(Predict(), name=f"predict_{pred}")
        predictor_node.inputs.descriptor = kwargs[f'{PRED}_DESCRIPTOR']
        predictor_node.inputs.out_filename = f'{pred}_map.nii.gz'
    else:
        predictor_node = Node(Predict(), name="predict_seg")
    predictor_node.inputs.model = kwargs['MODELS_PATH']

    workflow.connect(input_node, "t1", predictor_node, "t1")
    return workflow
