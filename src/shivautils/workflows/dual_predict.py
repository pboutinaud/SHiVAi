"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Workflow
from shivautils.workflows.predict import genWorkflow as genWorkflowPredict


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """

    workflow = genWorkflowPredict(**kwargs)
    workflow.name = 'dual_predictor_workflow'

    if 'SUB_WF' in kwargs.keys() and kwargs['SUB_WF']:
        input_node = workflow.get_node('input_parser')
    else:
        input_node = workflow.get_node('dataGrabber')

    if 'PRED' in kwargs.keys():
        PRED = kwargs['PRED']
        pred = PRED[:3].lower()
        pred_name = f"predict_{pred}"
    else:
        pred_name = "predict_seg"

    predictor_node = workflow.get_node(pred_name)
    workflow.connect(input_node, "flair", predictor_node, "flair")  # T1 already set in imported wf

    return workflow
