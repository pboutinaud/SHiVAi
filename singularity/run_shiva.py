#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import yaml
import os
import os.path as op


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

    parser.add_argument("-pr", '--preproc_results',
                        type=str,
                        help=(
                            'Path to the results folder of a previous shiva run, containing all the preprocessed data.\n'
                            'Requires that all the subjects from the current subject list (as per the content of --in or --sub_list) '
                            'are available in the results folder. If you have subjects with missing preprocessed data, you will '
                            'need to run their processing separatly.'
                        ))

    parser.add_argument("-c", "--config",
                        help='yaml file for configuration of workflow',
                        required=True)

    parser.add_argument("-p", '--prediction',
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'LAC', 'all'],
                        nargs='+',
                        help=("Choice of the type of prediction (i.e. segmentation) you want to compute.\n"
                              "A combination of multiple predictions (separated by a white space) can be given.\n"
                              "- 'PVS' for the segmentation of perivascular spaces using only T1 scans\n"
                              "- 'PVS2' for the segmentation of perivascular spaces using both T1 and FLAIR scans\n"
                              "- 'WMH' for the segmentation of white matter hyperintensities (requires both T1 and FLAIR scans)\n"
                              "- 'CMB' for the segmentation of cerebral microbleeds (requires SWI scans)\n"
                              "- 'LAC' for the segmentation of lacunas (requires both T1 and FLAIR scans)\n"
                              "- 'all' for doing 'PVS2', 'WMH', 'LAC', and 'CMB' segmentation (requires T1, FLAIR, and SWI scans)"),
                        default=['PVS'])

    sub_lists_args = parser.add_mutually_exclusive_group()
    sub_lists_args.add_argument('--sub_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to be processed. The IDs must be '
                                      'the same as the ones given in the input folder. In the file, the IDs can be separated '
                                      'by a whitespace, a new line, or any of the following characters [ "," ";" "|" ] '
                                      '(or a combination of those). If this argument is not given, all the participants '
                                      'in the input folder will be processed'))

    sub_lists_args.add_argument('--exclusion_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to NOT be processed. This option can be '
                                      'used when processing all the data in the input folder except for a few (because they have '
                                      'faulty data for exemple).\n'
                                      'In the file, the syntax is the same as for --sub_list.'))

    parser.add_argument('--masked',
                        action='store_true',
                        help='Select this if the input images are masked (i.e. with the brain extracted)')

    # parser.add_argument('-r', '--retry',
    #                     action='store_true',
    #                     help='Relaunch the pipeline from where it stopped')

    parser.add_argument('--run_plugin',
                        default='Linear',
                        help=('Type of plugin used by Nipype to run the workflow.\n'
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))
    parser.add_argument('--run_plugin_args',
                        type=str,
                        help=('Configuration file (.yml) for the plugin used by Nipype to run the workflow.\n'
                              'It will be imported as a dictionary and given plugin_args '
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))
    return parser


def main():
    parser = singParser()
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_content = yaml.safe_load(file)

    bind_model = f"{yaml_content['model_path']}:/mnt/model:ro"
    bind_input = f"{args.input}:/mnt/data/input:rw"
    if not (op.exists(args.output) and op.isdir(args.output)):
        os.makedirs(args.output)
    bind_output = f"{args.output}:/mnt/data/output:rw"
    bind_config = f"{op.dirname(op.abspath(args.config))}:/mnt/config:rw"
    singularity_image = f"{yaml_content['singularity_image']}"

    # Minimal shiva input
    input = "--in /mnt/data/input"
    output = "--out /mnt/data/output"
    pred = f"--prediction {' '.join(args.prediction)}"
    config = f"--model_config /mnt/config/{op.basename(args.config)}"

    # Optional shiva input
    opt_args = []
    opt_args.append(f"--input_type {args.input_type}")
    opt_args.append(f'--run_plugin {args.run_plugin}')
    if args.sub_list:
        opt_args.append(f"--sub_list {args.sub_list}")
    if args.exclusion_list:
        opt_args.append(f"--exclusion_list {args.exclusion_list}")
    if args.run_plugin_args:
        opt_args.append(f'--run_plugin_args /mnt/plugin/{op.basename(args.run_plugin_args)}')
    if args.masked:
        opt_args.append("--masked")
    if args.preproc_results:
        opt_args.append(f"--preproc_results {args.preproc_results}")

    bind_list = [bind_model, bind_input, bind_output, bind_config]
    if args.run_plugin_args:
        bind_plugin = f"{op.dirname(op.abspath(args.run_plugin_args))}:/mnt/plugin:rw"
        bind_list.append(bind_plugin)
    bind = ','.join(bind_list)

    command_list = ["singularity exec --nv --bind", bind, singularity_image,
                    "shiva --containerized_all", input, output, pred, config
                    ] + opt_args

    # if args.retry:
    #     command_list.append('--retry')

    if args.keep_all:
        command_list.append('--keep_all')

    command = ' '.join(command_list)

    print(command)

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
