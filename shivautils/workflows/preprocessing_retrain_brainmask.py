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
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.fsl import BET

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
    workflow = Workflow("preprocessing_retrain_brainmask")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['img']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template_args = {'img': [['subject_id']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    field_correction = Node(N4BiasFieldCorrection(), name='field_correction')
    workflow.connect(datagrabber, 'img', field_correction, 'input_image')
    
    brain_mask_raw = Node(BET(), name='brain_mask_raw')
    brain_mask_raw.inputs.mask = True
    workflow.connect(field_correction, 'output_image', brain_mask_raw, 'in_file')

    # crop main centered on mask
    crop = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop")
    workflow.connect(datagrabber, 'img',
                     crop, 'apply_to')
    workflow.connect(brain_mask_raw, 'mask_file',
                     crop, 'roi_mask')
    
    crop_brainmask = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop_brainmask")
    workflow.connect(brain_mask_raw, 'mask_file',
                     crop_brainmask, 'apply_to')
    workflow.connect(brain_mask_raw, 'mask_file',
                     crop_brainmask, 'roi_mask')
    
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (160, 214, 176)
    conform.inputs.orientation = 'RAS'

    workflow.connect(datagrabber, 'img', conform, 'img')

    conform_brainmask = Node(Conform(),
                   name="conform_brainmask")
    conform_brainmask.inputs.dimensions = (160, 214, 176)
    conform_brainmask.inputs.orientation = 'RAS'

    workflow.connect(brain_mask_raw, "mask_file", conform_brainmask, 'img')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
