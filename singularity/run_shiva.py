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

    bind_list = [bind_model, bind_input, bind_output]
    bind = ','.join(bind_list)

    command_list = ["singularity exec --nv --bind", bind, singularity_image,
                    "shiva.py --container", input, output, input_type, config]

    command = ' '.join(command_list)

    print(command)

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
