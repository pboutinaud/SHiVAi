"""Nipype workflow for prediction and return segmentation images"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.shiva import PredictSingularity, Predict



dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("SWI_predictor_workflow")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['SWI']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    if kwargs['CONTAINER'] == True:
        predict_cmb = Node(Predict(), name="predict_cmb")
        predict_cmb.inputs.model = kwargs['MODELS_PATH']
    else:
        predict_cmb = Node(PredictSingularity(), name="predict_cmb")
        predict_cmb.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
        predict_cmb.inputs.snglrt_bind =  [
            (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
            ('`pwd`','/mnt/data','rw'),
            ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
            ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
            (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
        predict_cmb.inputs.model = '/mnt/model'
        predict_cmb.inputs.snglrt_enable_nvidia = True
        predict_cmb.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'

    predict_cmb.inputs.descriptor = kwargs['CMB_DESCRIPTOR']
    predict_cmb.inputs.out_filename = 'cmb_map.nii.gz'

    workflow.connect(datagrabber, "SWI", predict_cmb, "swi")

    return workflow