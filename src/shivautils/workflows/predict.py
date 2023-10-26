"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.shiva import Predict


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def get_input_files(subject_id, in_dict):
    t1 = in_dict[subject_id]['T1_cropped']
    flair = in_dict[subject_id]['FLAIR_cropped']
    return t1, flair


def make_output_dict(sub_list, pred_map_list):
    return {sub: pred_map for sub, pred_map in zip(sub_list, pred_map_list)}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow
    Most arguments are part of the wfargs in shiva.py
    But there is also "PRED" which may contains the type of prediction (PVS/PVS2/WMH/CMB)
    Returns:
        workflow
    """

    if 'PRED' in kwargs.keys():  # When called from a bigger wf (shiva.py)
        PRED = kwargs['PRED']
        pred = PRED.lower()
        if pred == 'pvs2':
            pred = 'pvs'
    else:  # placeholders
        pred = 'seg'
        PRED = 'SEG'

    wf_name = f'{pred}_predictor_workflow'
    workflow = Workflow(wf_name)
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

    predictor_node = Node(Predict(), name=f"predict_{pred}")
    predictor_node.inputs.descriptor = kwargs[f'{PRED}_DESCRIPTOR']
    predictor_node.inputs.out_filename = f'{pred}_map.nii.gz'
    predictor_node.inputs.model = kwargs['MODELS_PATH']

    workflow.connect(input_node, "t1", predictor_node, "t1")

    predict_out_node = JoinNode(
        Function(
            input_names=['sub_list', 'pred_map_list'],
            output_names='predict_out_dict',
            function=make_output_dict),
        name='predict_out_node',
        joinsource=subject_list,
        joinfield=['sub_list', 'pred_map_list']
    )

    workflow.connect(subject_list, 'subject_id', predict_out_node, 'sub_list')
    workflow.connect(predictor_node, "segmentation", predict_out_node, "pred_map_list")

    return workflow
