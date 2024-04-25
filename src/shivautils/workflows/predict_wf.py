"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow

from shivautils.interfaces.shiva import Predict, PredictSingularity


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow with all the predict nodes needed
    Requires external connections for inputs
    Returns:
        workflow
    """
    segmentation_wf = Workflow('Segmentation')
    for pred in kwargs['PREDICTION']:
        if pred == 'PVS2':
            pred = 'PVS'
            descriptor = kwargs['PVS2_DESCRIPTOR']
        else:
            descriptor = kwargs[f'{pred}_DESCRIPTOR']
        lpred = pred.lower()
        # Prediction Node set-up
        if kwargs['CONTAINERIZE_NODES']:
            predict_node = Node(PredictSingularity(), name=f'predict_{lpred}')
            predict_node.inputs.out_filename = f'/mnt/data/{lpred}_map.nii.gz'
            predict_node.inputs.snglrt_enable_nvidia = True
            predict_node.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
            predict_node.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            preproc_dir = kwargs['PREP_SETTINGS']['preproc_res']
            if preproc_dir and kwargs['BASE_DIR'] not in preproc_dir:  # Preprocessed data not in BASE_DIR
                predict_node.inputs.snglrt_bind.append(
                    (preproc_dir, preproc_dir, 'ro')
                )
        else:
            predict_node = Node(Predict(), name=f'predict_{lpred}')
            predict_node.inputs.out_filename = f'{lpred}_map.nii.gz'
        predict_node.inputs.model = kwargs['MODELS_PATH']
        predict_node.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_node.inputs.descriptor = descriptor

        segmentation_wf.add_nodes([predict_node])

    return segmentation_wf
