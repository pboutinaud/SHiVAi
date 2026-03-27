"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow

from shivai.interfaces.shiva import Predict_Multi, Predict_Multi_Contained
from shivai.utils.container_config import configure_container_node


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
        container_runtime = kwargs.get('CONTAINER_RUNTIME')
        if container_runtime:
            predict_node = Node(Predict_Multi_Contained(), name=f'predict_{lpred}')
            predict_node.inputs.out_dir = '/mnt/data'
            bind_list = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            if kwargs['MODELS_PATH'] not in descriptor:
                descriptor_dir = os.path.dirname(descriptor)
                bind_list.append((descriptor_dir, descriptor_dir, 'ro'))
            preproc_dir = kwargs['PREP_SETTINGS']['preproc_res']
            if preproc_dir and kwargs['BASE_DIR'] not in preproc_dir:  # Preprocessed data not in BASE_DIR
                bind_list.append(
                    (preproc_dir, preproc_dir, 'ro')
                )
            configure_container_node(predict_node, container_runtime, kwargs['CONTAINER_IMAGE'], bind_list)
        else:
            predict_node = Node(Predict_Multi(), name=f'predict_{lpred}')
        predict_node.inputs.foutname = f'{{sub}}_{lpred}_map.nii.gz'
        predict_node.inputs.model_dir = kwargs['MODELS_PATH']
        predict_node.inputs.descriptor = descriptor
        predict_node.inputs.input_size = kwargs['IMAGE_SIZE']
        predict_node.inputs.batch_size = kwargs['BATCH_SIZE']

        plugin_args = kwargs['PRED_PLUGIN_ARGS']

        if kwargs['GPU'] is not None and kwargs['GPU'] < 0:
            predict_node.inputs.use_cpu = kwargs['AI_THREADS']
            if '--gpus' in plugin_args:
                list_args = plugin_args.split(' ')
                gpu_arg_ind1 = list_args.index('--gpus')
                gpu_arg_ind2 = gpu_arg_ind1 + 1
                list_args_noGPU = [arg for i, arg in enumerate(list_args) if i not in [gpu_arg_ind1, gpu_arg_ind2]]
                plugin_args = ' '.join(list_args_noGPU)

        predict_node.plugin_args = plugin_args

        segmentation_wf.add_nodes([predict_node])

    return segmentation_wf
