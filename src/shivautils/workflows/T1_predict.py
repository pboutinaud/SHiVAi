"""Nipype workflow for prediction and return segmentation images with only T1 scans as input"""
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
                                   outfields=['t1']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], kwargs['WF_DIRS']['preproc'])
    datagrabber.inputs.template = '%s/%s/*.nii.gz'
    datagrabber.inputs.field_template = {
        't1': '_subject_id_%s/t1_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz'}
    datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']],
                                        }
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    predict_pvs = Node(Predict(), name="predict_pvs")
    predict_pvs.inputs.model = kwargs['MODELS_PATH']

    predict_pvs.inputs.descriptor = kwargs['PVS_DESCRIPTOR']
    predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'

    workflow.connect(datagrabber, "t1", predict_pvs, "t1")

    return workflow
