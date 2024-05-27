#!/usr/bin/env python
"""SHIVA preprocessing workflow for 3D Slicer extension"""
import os
import argparse
import json

from shivai.old_workflows.slicer_preprocessing import genWorkflow

DESCRIPTION = """SHIVA preprocessing for deep learning predictors, for the 3D SLicer extension.
                 Perform resampling of a structural NIfTI head image,
                 followed by intensity normalization, and cropping centered on
                 the brain. A nipype workflow is used to preprocess a lot of
                 image in same time.
                 
                 Inputs are taken from a JSON file generated from 3D slicer."""


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

    parser.add_argument('--in', dest='in_dir',
                        help='JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/slicer_extension_database.json',
                        required=True)

    parser.add_argument('--plugin', dest='plugin', type=str,
                        help='Nipype plugin for batch processing (Linear or SLURM)',
                        required=False)
    return parser


def main():
    """Parameterize and run the nipype preprocessing workflow."""
    parser = build_args_parser()
    args = parser.parse_args()
    # the path from which the images to be preprocessed come
    # the preprocessed image destination path
    with open(args.in_dir, 'r') as json_in:
        slicerdb = json.load(json_in)

    subject_list = list(slicerdb['all_files'].keys())

    out_dir = os.path.abspath(slicerdb['parameters']['out_dir'])
    wfargs = {'FILES_LIST': subject_list,
              'ARGS': slicerdb,
              'BASE_DIR': slicerdb['files_dir']}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    wf = genWorkflow(**wfargs)
    wf.base_dir = out_dir

    wf.get_node('conform').inputs.dimensions = (256, 256, 256)
    wf.get_node('conform').inputs.voxel_size = tuple(slicerdb['parameters']['voxels_size'])
    wf.get_node('conform').inputs.orientation = 'RAS'
    wf.get_node('conform_seg').inputs.dimensions = (256, 256, 256)
    wf.get_node('conform_seg').inputs.voxel_size = tuple(slicerdb['parameters']['voxels_size'])
    wf.get_node('conform_seg').inputs.orientation = 'RAS'
    wf.get_node('crop').inputs.final_dimensions = tuple(slicerdb['parameters']['final_dimensions'])
    wf.get_node('crop_seg').inputs.final_dimensions = tuple(slicerdb['parameters']['final_dimensions'])
    wf.get_node('crop_2').inputs.final_dimensions = tuple(slicerdb['parameters']['final_dimensions'])
    wf.get_node('intensity_normalization').inputs.percentile = slicerdb['parameters']['percentile']
    wf.get_node('intensity_normalization_2').inputs.percentile = slicerdb['parameters']['percentile']
    # Predictor model descriptor file (JSON)
    wf.get_node('brain_mask_2').plugin_args = {'sbatch_args': '--gpus 1'}

    wf.get_node('brain_mask_2').inputs.descriptor = slicerdb['parameters']['brainmask_model']

    # singularity container bind mounts (so that folders of
    # the host appear inside the container)
    # gcc, cuda libs, model directory and source data dir on the host
    wf.get_node('brain_mask_2').inputs.snglrt_bind = [
        (out_dir, out_dir, 'rw'),
        ('`pwd`', '/mnt/data', 'rw'),
        ('/bigdata/resources/cudas/cuda-11.2', '/mnt/cuda', 'ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        ('/homes_unix/boutinaud/ReferenceModels', '/mnt/model', 'ro'),
        ('/homes_unix/yrio/Documents/data/BORCMB15/test2img/model_info', '/homes_unix/yrio/mnt/descriptor', 'ro')]
    wf.get_node('brain_mask_2').inputs.descriptor = '/homes_unix/yrio/mnt/descriptor/model_info.json'
    wf.get_node('brain_mask_2').inputs.model = '/mnt/model'
    wf.get_node('brain_mask_2').inputs.snglrt_image = '/homes_unix/yrio/singularity/predict.sif'
    wf.get_node('brain_mask_2').inputs.out_filename = '/mnt/data/brain_mask.nii.gz'

    wf.run(plugin=args.plugin)


if __name__ == "__main__":
    main()
