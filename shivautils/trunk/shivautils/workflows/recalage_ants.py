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
                                   outfields=['main_fealinx', 'main_gin']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'main_fealinx': [['subject_id', 'main_fealinx']],
                                        'main_gin': [['subject_id', 'main_gin']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
    

    # compute 3-dof (translations) coregistration parameters of cropped to native main
    gin_to_fealinx = Node(ants.Registration(),
                 name='gin_to_fealinx')
    gin_to_fealinx.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    gin_to_fealinx.inputs.transforms = ['Rigid']
    gin_to_fealinx.inputs.restrict_deformation=[[1,0,0,],[1,0,0,],[1,0,0]]
    gin_to_fealinx.inputs.transform_parameters = [(0.1,)]
    gin_to_fealinx.inputs.metric = ['MI']
    gin_to_fealinx.inputs.radius_or_number_of_bins = [64]
    gin_to_fealinx.inputs.shrink_factors = [[8,4,2,1]]
    gin_to_fealinx.inputs.output_warped_image = False
    gin_to_fealinx.inputs.smoothing_sigmas = [[3,2,1,0]]
    gin_to_fealinx.inputs.num_threads = 8
    gin_to_fealinx.inputs.number_of_iterations = [[1000,500,250,125]]
    gin_to_fealinx.inputs.sampling_strategy = ['Regular']
    gin_to_fealinx.inputs.sampling_percentage = [0.25]
    gin_to_fealinx.inputs.output_transform_prefix = "cropped_to_source_"
    gin_to_fealinx.inputs.verbose = True
    gin_to_fealinx.inputs.winsorize_lower_quantile = 0.0
    gin_to_fealinx.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(datagrabber, "main_fealinx",
                     gin_to_fealinx, 'moving_image')
    workflow.connect(datagrabber, 'main_gin',
                     gin_to_fealinx, 'fixed_image')
                
    # write brain seg on main in native space
    fealinx_to_gin = Node(ants.ApplyTransforms(), name="fealinx_to_gin")
    fealinx_to_gin.inputs.interpolation = 'Linear'
    workflow.connect(gin_to_fealinx, 'forward_transforms', fealinx_to_gin, 'transforms' )
    workflow.connect(datagrabber, 'main_fealinx', fealinx_to_gin, 'input_image')
    workflow.connect(datagrabber, "main_gin", fealinx_to_gin, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    main_norm =  Node(Normalization(percentile = 99), name="main_final_intensity_normalization")
    workflow.connect(fealinx_to_gin, "output_image",
    		         main_norm, 'input_image')

    predict = Node(PredictDirect(), name="predict")
    predict.inputs.model = kwargs['MODELS_PATH']
    predict.inputs.descriptor = kwargs['DESCRIPTOR']
    predict.inputs.out_filename = 'map.nii.gz'

    workflow.connect(main_norm, "intensity_normalized", predict, 'swi')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
