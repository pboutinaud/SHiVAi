#!/usr/bin/env python
"""Script workflow"""
import os
import argparse
import json



from shivautils.workflows.test_single_interpolation import genWorkflow

DESCRIPTION = """SHIVA deep learning predictors.
                 A nipype workflow is used to preprocess a 
                 lot of image in same time"""
                 
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


def build_args_parser():
    """Create a command line to specify arguments with argparse

    Returns:
        arguments
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='input',
                        help='path input',
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

    parser.add_argument('--simg',
                        type=existing_file,
                        default=None,
                        help='Predictor Singularity image to use')
    
    parser.add_argument('--gpu',
                        type=int,
                        help='GPU to use.')

    return parser


def main():
    """Parameterize and run the nipype preprocessing workflow."""
    parser = build_args_parser()
    args = parser.parse_args()
    # the path from which the images to be preprocessed come
    # the preprocessed image destination path
    subject_dict = args.input
    
    out_dir = args.output
    wfargs = {'SUBJECT_LIST': os.listdir(subject_dict),
              'BASE_DIR': out_dir,
              'MODELS_PATH': args.model,
              'BRAINMASK_DESCRIPTOR': os.path.join(args.model, 'brainmask/V0/model_info.json'),
              'PVS_DESCRIPTOR': os.path.join(args.model, 'T1-PVS/V0/model_info.json'),
              'PERCENTILE_VALUE': args.percentile}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    wf = genWorkflow(**wfargs)
    wf.base_dir = out_dir
    wf.get_node('dataGrabber').inputs.base_directory = subject_dict
    wf.get_node('dataGrabber').inputs.template = args.grab_pattern
    wf.get_node('dataGrabber').inputs.template_args = {'T1': [['subject_id', 'T1']],
                                                       'GIN': [['subject_id', 'GIN']]}
    
    wf.run(plugin='SLURM')


if __name__ == "__main__":
    main()
