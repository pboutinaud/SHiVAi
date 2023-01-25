import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.image import Normalization, Conform


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genParamWf(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("wf_parametre")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subjectList = Node(IdentityInterface(
        fields=['subject_id'], mandatory_inputs=True), name="subjectList")
    subjectList.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['main']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s'
    datagrabber.inputs.template_args = {'main': [['subject_id']]}

    workflow.connect(subjectList, 'subject_id', datagrabber, 'subject_id')

    conform = Node(Conform(dimensions=(256, 256, 256),
                           voxel_size=(1.0, 1.0, 1.0),
                           orientation='RAS'),
                   name="conform")

    workflow.connect(datagrabber, 'main', conform, 'img')

    normalization = Node(Normalization(), name="normalization")
    normalization.iterables = ("percentile", kwargs['percentiles'])
    workflow.connect(conform, 'resampled', normalization, 'input_image')

    return workflow