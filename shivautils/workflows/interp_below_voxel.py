#!/usr/bin/env python
"""Nipype workflow for conformation and preparation before deep
   learning, with accessory image coregistation to cropped space (through ANTS). 
   This also handles back-registration from conformed-crop to main.
   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.ants import ApplyTransforms

from shivautils.interfaces.shiva import PredictDirect
from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def as_list(input):
    return [input]
    

def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_registration_processing")
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
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'t1': [['subject_id', 't1']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
                
    # write brain seg on main in native space
    below_voxel = Node(ants.ApplyTransforms(), name="below_voxel")
    below_voxel.inputs.interpolation = 'Linear'
    below_voxel.inputs.transforms = kwargs['MATRIX_TRANSFORMS']
    workflow.connect(datagrabber, 't1', below_voxel, 'input_image')
    workflow.connect(datagrabber, "t1", below_voxel, 'reference_image')

    predict = Node(PredictDirect(), name="predict")
    predict.inputs.model = kwargs['MODELS_PATH']
    predict.inputs.descriptor = kwargs['DESCRIPTOR']
    predict.inputs.out_filename = 'map.nii.gz'

    workflow.connect(below_voxel, 'output_image', predict, 't1')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
