#!/usr/bin/env python
"""Script workflow"""
import os
import argparse
import json



from shivautils.workflows.wf_BIDS import genWorkflow

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
                        help='path',
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
    data_dir = os.path.normpath(args.input)
    subject_list = os.listdir(data_dir)

    out_dir = args.output
    wfargs = {'SUBJECT_LIST': subject_list,
              'BASE_DIR': out_dir,
              'BIDS_DIR': data_dir,
              'MODELS_PATH': args.model}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    wf = genWorkflow(**wfargs)
    wf.base_dir = out_dir
    wf.get_node('BIDSdataGrabber').inputs.base_directory = data_dir
    wf.get_node('BIDSdataGrabber').inputs.template = args.grab_pattern
    wf.get_node('BIDSdataGrabber').inputs.template_args = {'T1': [['subject_id', 'subject_id']],
                                                           'FLAIR': [['subject_id', 'subject_id']]}
    wf.get_node('BIDSdataGrabber').inputs.field_template = {'T1': '%s/anat/%s_T1_raw.nii.gz',
                                                            'FLAIR': '%s/anat/%s_FLAIR_raw.nii.gz'}


    wf.run(plugin='Linear')



if __name__ == "__main__":
    main()
