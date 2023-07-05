#!/usr/bin/env python
# Script workflow in containeur singularity
import os
import argparse
import json
import nibabel as nb
import numpy as np
import scipy.io

from shivautils.workflows.test_metrics_prediction import genWorkflow

DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                preprocess a lot of images at the same time."""
                 
def existing_file(file):
    """Checking if file exist

    Args:
        f (_type_): _description_

    Raises:
        argparse.ArgumentTypeError: _description_

    Returns:
        _type_: _description_
    """
    if not os.path.isfile(file):
        raise argparse.ArgumentTypeError(file + " not found.")
    else:
        return file


parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument('--in', dest='input',
                    help='JSON formatted extract of the Slicer plugin',
                    metavar='path/to/existing/slicer_extension_database.json',
                    required=True)

parser.add_argument('--out', dest='output',
                    type=str,
                    help='Output folder path (nipype working directory)',
                    metavar='path/to/nipype_work_dir',
                    required=True)

parser.add_argument('--grab', dest='grab_pattern',
                    type=str,
                    help='data grabber pattern, between quotes',
                    metavar='%s/%s/*nii',
                    required=True)
                    
parser.add_argument('--model',
                    default=None,
                    help='path to model descriptor')

parser.add_argument('--simg',
                    type=existing_file,
                    default=None,
                    help='Predictor Singularity image to use')

parser.add_argument('--gpu',
                    type=int,
                    help='GPU to use.')



args = parser.parse_args()
# the path from which the images to be preprocessed come
# the preprocessed image destination path
subject_dict = args.input

out_dir = args.output
wfargs = {'SUBJECT_LIST': os.listdir(subject_dict),
          'BASE_DIR': out_dir,
          # CMB_descriptor ou PVS_descriptor
          'DESCRIPTOR': os.path.join(args.model, 'brainmask/V0/model_info.json'),
          'MODELS_PATH': args.model,
          'SYNTHSEG': True}

if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
    os.makedirs(out_dir)
print(f'Working directory set to: {out_dir}')

wf = genWorkflow(**wfargs)
wf.base_dir = out_dir
wf.get_node('dataGrabber').inputs.base_directory = subject_dict
wf.get_node('dataGrabber').inputs.template = args.grab_pattern
wf.get_node('dataGrabber').inputs.template_args = {'t1_preproc': [['subject_id', 't1_preproc']],
                                                   'prediction': [['subject_id', 'prediction']]}

wf.run(plugin='Linear')
