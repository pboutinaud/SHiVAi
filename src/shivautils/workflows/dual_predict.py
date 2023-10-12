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
    workflow = Workflow(kwargs['WF_DIRS']['pred'])
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['t1', 'flair']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], kwargs['WF_DIRS']['preproc'])
    datagrabber.inputs.template = '%s/%s/*.nii*'
    datagrabber.inputs.field_template = {
        't1': '_subject_id_%s/t1_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz',
        'flair': '_subject_id_%s/flair_final_intensity_normalization/t1_to_flair__Warped_img_normalized.nii.gz'}
    datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']],
                                        'flair': [['subject_id']]}
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    if kwargs['CONTAINER'] == True:  # BUG? Shouldn't it be False? (because PredictSingularity is used in the 'else')
        predict_pvs = Node(Predict(), name="predict_pvs")
        predict_pvs.inputs.model = kwargs['MODELS_PATH']
    else:
        predict_pvs = Node(PredictSingularity(), name="predict_pvs")
        predict_pvs.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
        predict_pvs.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
        predict_pvs.inputs.model = '/mnt/model'
        predict_pvs.inputs.snglrt_enable_nvidia = True
        predict_pvs.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'

    predict_pvs.inputs.descriptor = kwargs['PVS_DESCRIPTOR']
    predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'

    workflow.connect(datagrabber, "t1", predict_pvs, "t1")
    workflow.connect(datagrabber, "flair", predict_pvs, "flair")

    if kwargs['CONTAINER'] == True:  # BUG? Shouldn't it be False? (because PredictSingularity is used in the 'else')
        predict_wmh = Node(Predict(), name="predict_wmh")
        predict_wmh.inputs.model = kwargs['MODELS_PATH']

    else:
        predict_wmh = Node(PredictSingularity(), name="predict_wmh")
        predict_wmh.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
        predict_wmh.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            ('/bigdata/resources/cudas/cuda-11.2', '/mnt/cuda', 'ro'),
            ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
            (kwargs['MODELS_PATH'], '/mnt/model', 'ro')]
        predict_wmh.inputs.model = '/mnt/model'
        predict_wmh.inputs.snglrt_enable_nvidia = True
        predict_wmh.inputs.snglrt_image = '/bigdata/yrio/singularity/predict_2.sif'

    predict_wmh.inputs.descriptor = kwargs['WMH_DESCRIPTOR']
    predict_wmh.inputs.out_filename = 'wmh_map.nii.gz'

    workflow.connect(datagrabber, "t1", predict_wmh, "t1")
    workflow.connect(datagrabber, "flair", predict_wmh, "flair")

    return workflow
