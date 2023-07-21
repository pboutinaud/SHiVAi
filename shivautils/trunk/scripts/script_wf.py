#!/usr/bin/env python
# Script workflow in containeur singularity
import os
import argparse
import json

from shivautils.workflows.dual_full_processing import genWorkflow
from shivautils.workflows.SWI_preprocessing import swiWorkflow


DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                preprocess a lot of images at the same time. In the last step the file segmentations with the wmh and 
                pvs models are processed"""


parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument('--in', dest='input',
                    help='Folder path with files, BIDS structure folder path or JSON formatted extract of the Slicer plugin',
                    metavar='path/to/existing/slicer_extension_database.json',
                    required=True)

parser.add_argument('--out', dest='output',
                    type=str,
                    help='Output folder path (nipype working directory)',
                    metavar='path/to/nipype_work_dir',
                    required=True)

parser.add_argument('--input_type',
                    type=str,
                    help='Way to grab and manage nifti files : standard, BIDS or json',
                    default='standard')

parser.add_argument('--grab', dest='grab_pattern',
                    type=str,
                    help='data grabber pattern, between quotes',
                    metavar='%s/*nii',
                    default='%s/*nii',
                    required=True)

parser.add_argument('--percentile',
                    type=float,
                    default=99,
                    help='Threshold value expressed as percentile')

parser.add_argument('--final_dimensions',
                    nargs='+', type=int,
                    default=(160, 214, 176),
                    help='Final image array size in i, j, k.')

parser.add_argument('--voxels_size', nargs='+',
                    type=float,
                    default=(1.0, 1.0, 1.0),
                    help='Voxel size of final image')
                    
parser.add_argument('--model',
                    default=None,
                    help='path to model descriptor')

parser.add_argument('--synthseg',
                    default=False,
                    help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

parser.add_argument('--SWI',
                    default='False',
                    help='If a second workflow for CMB is required')

parser.add_argument('--gpu',
                    type=int,
                    help='GPU to use.')



args = parser.parse_args()

SWI = str(args.SWI)
synthseg = args.synthseg

if args.input_type == 'json':
    with open(args.input, 'r') as json_in:
        subject_dict = json.load(json_in)
    
    out_dir = subject_dict['parameters']['out_dir']
    subject_directory = subject_dict["files_dir"]
    subject_list = os.listdir(subject_directory)
    brainmask_descriptor = subject_dict['parameters']['brainmask_descriptor']
    if subject_dict['parameters']['WMH_descriptor']:
        wmh_descriptor = subject_dict['parameters']['WMH_descriptor']
    else:
        wmh_descriptor = None
    if subject_dict['parameters']['PVS_descriptor']:
        pvs_descriptor = subject_dict['parameters']['PVS_descriptor']
    else:
        pvs_descriptor = None
    if subject_dict['parameters']['CMB_descriptor']:
        cmb_descriptor = subject_dict['parameters']['CMB_descriptor']
    else:
        cmb_descriptor = None

if args.input_type == 'standard' or args.input_type == 'BIDS':
    subject_directory = args.input
    out_dir = args.output
    subject_list = os.listdir(subject_directory)
    brainmask_descriptor = os.path.join(args.model, 'brainmask/V0/model_info.json')
    wmh_descriptor = os.path.join(args.model, 'T1.FLAIR-WMH/V1/model_info.json')
    pvs_descriptor = os.path.join(args.model, 'T1.FLAIR-PVS/V0/model_info.json')
    if SWI == 'True':
        cmb_descriptor = os.path.join(args.model, 'SWI-CMB/V0/model_info.json')
    else:
        cmb_descriptor = None
    

wfargs = {'SUBJECT_LIST': subject_list,
          'DATA_DIR': subject_directory,
          'BASE_DIR': out_dir,
          'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
          'WMH_DESCRIPTOR': wmh_descriptor,
          'PVS_DESCRIPTOR': pvs_descriptor,
          'CMB_DESCRIPTOR': cmb_descriptor,
          'MODELS_PATH': args.model,
          'SYNTHSEG': synthseg}

if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
    os.makedirs(out_dir)
print(f'Working directory set to: {out_dir}')

wf = genWorkflow(**wfargs)
wf.base_dir = out_dir

if SWI == 'True':
    swi_wf = swiWorkflow(**wfargs)
    swi_wf.base_dir = out_dir

    

if args.input_type == 'standard' or args.input_type == 'json':    
    wf.get_node('dataGrabber').inputs.base_directory = subject_directory
    wf.get_node('dataGrabber').inputs.template = args.grab_pattern
    wf.get_node('dataGrabber').inputs.template_args = {'T1': [['subject_id', 'T1']],
                                                       'FLAIR': [['subject_id', 'FLAIR']]}
    if SWI == 'True':
        swi_wf.get_node('dataGrabber').inputs.base_directory = subject_directory
        swi_wf.get_node('dataGrabber').inputs.template = args.grab_pattern
        swi_wf.get_node('dataGrabber').inputs.template_args = {'SWI': [['subject_id', 'SWI']]}

if args.input_type == 'BIDS':
    wf.get_node('dataGrabber').inputs.base_directory = args.input
    wf.get_node('dataGrabber').inputs.template = args.grab_pattern
    wf.get_node('dataGrabber').inputs.template_args = {'T1': [['subject_id', 'subject_id']],
                                                       'FLAIR': [['subject_id', 'subject_id']]}
    wf.get_node('dataGrabber').inputs.field_template = {'T1': '%s/anat/%s_T1_raw.nii.gz',
                                                        'FLAIR': '%s/anat/%s_FLAIR_raw.nii.gz'}
    if SWI == 'True':
        swi_wf.get_node('dataGrabber').inputs.base_directory = subject_directory
        swi_wf.get_node('dataGrabber').inputs.template = args.grab_pattern
        swi_wf.get_node('dataGrabber').inputs.template_args = {'SWI': '%s/anat/%s_T1_raw.nii.gz'}

wf.get_node('conform').inputs.dimensions = (256, 256, 256)
wf.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
wf.get_node('conform').inputs.orientation = 'RAS'
wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

if args.gpu:
    wf.get_node('pre_brain_mask').inputs.gpu_number = args.gpu
    wf.get_node('post_brain_mask').inputs.gpu_number = args.gpu

wf.run(plugin='Linear')

if SWI == 'True':
    swi_wf.run(plugin='Linear')
