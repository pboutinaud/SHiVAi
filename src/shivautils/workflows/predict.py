"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.shiva import PredictSingularity, Predict


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def get_input_lists(in_dict):
    subject_list = in_dict['subject_id']
    t1_list = in_dict['t1_preproc']
    flair_list = in_dict['flair_preproc']
    return subject_list, t1_list, flair_list


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow('predictor_workflow')
    workflow.base_dir = kwargs['BASE_DIR']

    # Data input possibilities: from WF or with Datagrabber:
    if 'SUB_WF' in kwargs.keys() and kwargs['SUB_WF']:  # From previous workflow
        input_node = Node(
            Function(
                input_names='in_dict',
                output_names=['subject_list', 't1_list', 'flair_list'],  # flair unsued here
                function=get_input_lists
            ),
            name='input_parser'
        )
        input_node.synchronize = True
        input_node.iterables([('subject_id', 't1'),
                              zip(
                                  input_node.outputs.subject_list,
                                  input_node.outputs.t1_list
        )])

    else:  # Using a datagrabber (requires additional settings outside of the wf)
        # get a list of subjects to iterate on
        subject_list = Node(
            IdentityInterface(
                fields=['subject_id'],
                mandatory_inputs=True),
            name="subject_list")
        subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

        # file selection
        inputNode = Node(
            DataGrabber(
                infields=['subject_id'],
                outfields=['t1']),
            name='dataGrabber')

        inputNode.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'shiva_mono_preprocessing')
        inputNode.inputs.template = '%s/%s/*.nii*'
        inputNode.inputs.raise_on_empty = True
        inputNode.inputs.sort_filelist = True

        workflow.connect(subject_list, 'subject_id', inputNode, 'subject_id')

    predict_pvs = Node(Predict(), name=f"predict_{pred}")
    predict_pvs.inputs.model = kwargs['MODELS_PATH']
    if 'PREDICTION' in kwargs.keys():  # Else need to be set outside of the wf
        PRED = kwargs['PREDICTION']
        pred = PRED[:3].lower()  # biomarkers should have 3 letters
        predict_pvs.inputs.descriptor = kwargs[f'{PRED}_DESCRIPTOR']
        predict_pvs.inputs.out_filename = f'{pred}_map.nii.gz'
    workflow.connect(inputNode, "t1", predict_pvs, "t1")
    return workflow
