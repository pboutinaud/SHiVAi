#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import yaml
import os


def singParser():
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
    parser.add_argument("-p", '--prediction',
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'all'],
                        nargs='?',
                        help=("Choice of the type of prediction (i.e. segmentation) you want to compute.\n"
                              "A combination of multiple predictions (separated by a white space) can be given.\n"
                              "- 'PVS' for the segmentation of perivascular spaces using only T1 scans\n"
                              "- 'PVS2' for the segmentation of perivascular spaces using both T1 and FLAIR scans\n"
                              "- 'WMH' for the segmentation of white matter hyperintensities (requires both T1 and FLAIR scans)\n"
                              "- 'CMB' for the segmentation of cerebral microbleeds (requires SWI scans)\n"
                              "- 'all' for doing 'PVS2', 'WMH', and 'CMB' segmentation (requires T1, FLAIR, and SWI scans)"),
                        default=['PVS'])
    return parser


def main():

    args = singParser.parse_args()

    with open(args.config, 'r') as file:
        yaml_content = yaml.safe_load(file)

    bind_model = f"{yaml_content['model_path']}:/mnt/model:ro"
    bind_input = f"{args.input}:/mnt/data/input:rw"
    if not (os.path.exists(args.output) and os.path.isdir(args.output)):
        os.makedirs(args.output)
    bind_output = f"{args.output}:/mnt/data/output:rw"
    singularity_image = f"{yaml_content['singularity_image']}"
    input = f"--in /mnt/data/input"
    output = f"--out /mnt/data/output"
    input_type = f"--input_type {args.input_type}"
    config = f"--model_config {args.config}"
    pred = f"--prediction {' '.join(args.prediction)}"

    bind_list = [bind_model, bind_input, bind_output]
    bind = ','.join(bind_list)

    command_list = ["singularity exec --nv --bind", bind, singularity_image,
                    "shiva.py --container", input, output, input_type, pred, config]

    command = ' '.join(command_list)

    print(command)

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
