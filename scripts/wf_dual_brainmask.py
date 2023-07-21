#!/usr/bin/env python
# Script workflow in containeur singularity
import os
import argparse
import json



from shivautils.workflows.direct_dual_processing import genWorkflow

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
                    help='path folder dataset',
                    metavar='path input',
                    required=True)

parser.add_argument('--out', dest='output',
                    type=str,
                    help='Output folder path (nipype working directory)',
                    metavar='path/to/nipype_work_dir',
                    required=True)

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

parser.add_argument('--voxel_size', nargs='+',
                    type=float,
                    default=(1.0, 1.0, 1.0),
                    help='Voxel size of final image')
                    
parser.add_argument('--model',
                    default=None,
                    help='path to model descriptor')

parser.add_argument('--synthseg',
                    default=False,
                    help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

parser.add_argument('--gpu',
                    type=int,
                    help='GPU to use.')



args = parser.parse_args()

subject_directory = args.input
subject_list = os.listdir(subject_directory)

out_dir = args.output
wfargs = {'SUBJECT_LIST': subject_list,
          'DATA_DIR': subject_directory,
          'BASE_DIR': out_dir,
          'BRAINMASK_DESCRIPTOR': os.path.join(args.model, 'brainmask/V0/model_info.json'),
          'WMH_DESCRIPTOR': os.path.join(args.model, 'T1.FLAIR-WMH/V1/model_info.json'),
          'PVS_DESCRIPTOR': os.path.join(args.model, 'T1.FLAIR-PVS/V0/model_info.json'),
          'MODELS_PATH': args.model,
          'ANONYMIZED': False,
          'INTERPOLATION': 'WelchWindowedSinc',
          'SYNTHSEG': args.synthseg,
          'PERCENTILE': args.percentile,
          'THRESHOLD': 0.5,
          'IMAGE_SIZE': args.final_dimensions,
          'RESOLUTION': args.voxel_size}

if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
    os.makedirs(out_dir)
print(f'Working directory set to: {out_dir}')

wf = genWorkflow(**wfargs)
wf.base_dir = out_dir
wf.get_node('dataGrabber').inputs.base_directory = subject_directory
wf.get_node('dataGrabber').inputs.template = args.grab_pattern
wf.get_node('dataGrabber').inputs.template_args = {'T1': [['subject_id', 'T1']],
                                                   'FLAIR': [['subject_id', 'FLAIR']]}
wf.get_node('conform').inputs.dimensions = (256, 256, 256)
wf.get_node('conform').inputs.voxel_size = tuple(args.voxel_size)
wf.get_node('conform').inputs.orientation = 'RAS'
wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

if args.gpu:
    wf.get_node('pre_brain_mask').inputs.gpu_number = args.gpu
    wf.get_node('post_brain_mask').inputs.gpu_number = args.gpu

wf.run(plugin='Linear')
