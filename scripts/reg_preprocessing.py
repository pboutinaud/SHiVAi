"""Script workflow"""
import os
import argparse
import json

from shivautils.workflows.slicer_preprocessing import genWorkflow

DESCRIPTION = """SHIVA preprocessing for deep learning predictors.
                 Perform calculating and registration of an main image on
                 an accessory image to had the same shape, size of voxels and
                 cropping followed by intensity normalization. 
                 A nipype workflow is used to preprocess a lot of
                 image in same time"""
                 
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
                        help='JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/slicer_extension_database.json',
                        required=True)

    '''
    parser.add_argument('--main', dest='main',
                        type=str,
                        help='reference image for registration',
                        required=True)
    parser.add_argument('--apply_to', dest='apply_to',
                        type=str,
                        help='images to change on the main conformation',
                        required=True)

    parser.add_argument('--brainmask',
                        type=str,
                        help='brain_mask calculating with tensorflow model on main image')
    parser.add_argument('--out', dest='output',
                        type=str,
                        help='Output folder path (nipype working directory)',
                        metavar='path/to/nipype_work_dir',
                        required=True)
    '''

    return parser


def main():
    """Parameterize and run the nipype preprocessing workflow."""
    parser = build_args_parser()
    args = parser.parse_args()
    # the path from which the images to be preprocessed come
    # the preprocessed image destination path
    with open(args.input, 'r') as json_in:
        slicerdb = json.load(json_in)
    
    out_dir = os.path.abspath(slicerdb['parameters']['out_dir'])
    wfargs = {'FILES_LIST': [(os.path.join(slicerdb['files_dir'], elt['raw']), 
                              os.path.join(slicerdb['files_dir'], elt['ref']),
                              os.path.join(slicerdb['files_dir'], elt['brainmask']))
                             for elt in slicerdb['all_files'].values()],
              'BASE_DIR': out_dir}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    wf = genWorkflow(**wfargs)
    wf.base_dir = out_dir

    wf.run(plugin='SLURM')


if __name__ == "__main__":
    main()
