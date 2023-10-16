"""SynthSeg nipype workflow"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.shiva import SynthSeg

dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}



def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow(kwargs['WF_SWI_DIRS']['pred'])
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['main']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = os.path.join(kwargs['BASE_DIR'])
    datagrabber.inputs.template = '%s/%s/*.nii.gz'
    datagrabber.inputs.field_template = {'main': ''}
    datagrabber.inputs.template_args = {'main': [['subject_id']]}
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    
    synth_seg = Node(SynthSeg(), name="synth_seg")
    synth_seg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    synth_seg.inputs.cpu = False
    synth_seg.inputs.robust = True
    synth_seg.inputs.parc=True
    synth_seg.inputs.out_filename = 'seg.nii.gz'

    workflow.connect(datagrabber, "main", synth_seg, "input")

    return workflow
  