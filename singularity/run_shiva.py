#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import yaml


DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                preprocess a lot of images at the same time."""

parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument("--in", dest='input', type=Path, action='store', help="path of folder images data", required=True)
parser.add_argument("--out", dest='output', type=Path, action='store', help="path for output of processing", required=True)
parser.add_argument("--model", default='/homes_unix/yrio/Documents/modele/ReferenceModels', required=False, type=Path, action='store', help="path to model folder")
parser.add_argument("--input_type", default='standard', type=str, help="File staging convention: 'standard', 'BIDS' or 'json'")
parser.add_argument("--config", help='yaml file for configuration of workflow')

args = parser.parse_args()

with open(args.config, 'r') as file:
    
    yaml_content = yaml.safe_load(file)

parameters = yaml_content['parameters']

bind_cuda = f"{yaml_content['cuda']}:/mnt/cuda:ro"
bind_gcc = f"{yaml_content['gcc']}:/mnt/gcc:ro"
bind_model = f"{args.model}:/mnt/model:rw"
bind_input = f"{args.input}:/mnt/data/input:rw"
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


bind_list = [bind_cuda, bind_gcc, bind_model, bind_input, bind_output]
bind = ','.join(bind_list)

command_list = ["singularity exec --nv --bind", bind, singularity_image,
                "script_wf.py", input, output, input_type, percentile,
                threshold, threshold_clusters, final_dimensions, 
                voxels_size, model, interpolation, swi, brainmask_descriptor,
                pvs_descriptor, wmh_descriptor, cmb_descriptor]

command = ' '.join(command_list)

print(command)

subprocess.run(command, shell=True)

