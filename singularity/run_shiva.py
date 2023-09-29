#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import yaml
import os


def main():
    DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                    followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                    preprocess a lot of images at the same time."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("-i", "--in", dest='input',
                        type=Path, action='store',
                        help="path of folder images data",
                        required=True)
    parser.add_argument("-o", "--out", dest='output',
                        type=Path, action='store',
                        help="path for output of processing",
                        required=True)
    parser.add_argument("-it", "--input_type",
                        default='standard',
                        choices=['standard', 'BIDS', 'json'],
                        help="File staging convention: 'standard', 'BIDS' or 'json'")
    parser.add_argument("-c", "--config",
                        help='yaml file for configuration of workflow',
                        required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_content = yaml.safe_load(file)

    parameters = yaml_content['parameters']

    bind_model = f"{yaml_content['model_path']}:/mnt/model:ro"
    bind_input = f"{args.input}:/mnt/data/input:rw"
    if not (os.path.exists(args.output) and os.path.isdir(args.output)):
        os.makedirs(args.output)
    bind_output = f"{args.output}:/mnt/data/output:rw"
    singularity_image = f"{yaml_content['singularity_image']}"
    input = f"--in /mnt/data/input"
    output = f"--out /mnt/data/output"
    input_type = f"--input_type {args.input_type}"
    percentile = f"--percentile {parameters['percentile']}"
    threshold = f"--threshold {parameters['threshold']}"
    threshold_clusters = f"--threshold_clusters {parameters['threshold_clusters']}"
    final_dimensions = f"--final_dimensions {parameters['final_dimensions']}"
    voxels_size = f"--voxels_size {parameters['voxels_size']}"
    interpolation = f"--interpolation {parameters['interpolation']}"
    model = f"--model /mnt/model"
    swi = f"--SWI {parameters['SWI']}"
    brainmask_descriptor = f"--brainmask_descriptor {parameters['brainmask_descriptor']}"
    pvs_descriptor = f"--pvs_descriptor {parameters['PVS_descriptor']}"
    wmh_descriptor = f"--wmh_descriptor {parameters['WMH_descriptor']}"
    cmb_descriptor = f"--cmb_descriptor {parameters['CMB_descriptor']}"

    bind_list = [bind_model, bind_input, bind_output]
    bind = ','.join(bind_list)

    command_list = ["singularity exec --nv --bind", bind, singularity_image,
                    "shiva.py", input, output, input_type, percentile,
                    threshold, threshold_clusters, final_dimensions,
                    voxels_size, model, interpolation, swi, brainmask_descriptor,
                    pvs_descriptor, wmh_descriptor, cmb_descriptor]

    command = ' '.join(command_list)

    print(command)

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
