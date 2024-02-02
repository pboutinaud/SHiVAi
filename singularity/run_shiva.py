#!/usr/bin/env python
import subprocess
import argparse
from pathlib import Path
import yaml
import json
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

    parser.add_argument("-db", "--db_name",
                        help='Name of the data-base the input scans originate from. It is only to add this detail in the report')

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
                                      '(or a combination of those). If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--sub_names',
                                nargs='+',
                                required=False,
                                help=('List of participant IDs to be processed. With this option, the IDs are given directly '
                                      'in the command line, separated by a white-space, and must be the same as the ones given '
                                      'in the input folder. If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--exclusion_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to NOT be processed. This option can be '
                                      'used when processing all the data in the input folder except for a few (because they have '
                                      'faulty data for exemple).\n'
                                      'In the file, the syntax is the same as for --sub_list\n.'
                                      'If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    parser.add_argument('--replace_t1',
                        type=str,
                        metavar='img_type',
                        help=('Image type to be used instead of T1w for PVS (and PVS2), WMH, and Lacuna segmentations.\n'
                              '(Note that part of the labels may keep the "t1" notation instead of the image type you '
                              'specified)'))

    parser.add_argument('--replace_flair',
                        type=str,
                        metavar='img_type',
                        help=('Image type to be used instead of FLAIR for PVS2, WMH, and Lacuna segmentations.\n'
                              '(Note that part of the labels may keep the "flair" notation instead of the image type you '
                              'specified)'))

    parser.add_argument('--replace_swi',
                        type=str,
                        metavar='img_type',
                        help=('Image type to be used instead of SWI for CMB segmentations.\n'
                              '(Note that part of the labels may keep the "swi" notation instead of the image type you '
                              'specified)'))

    parser.add_argument('--masked',
                        action='store_true',
                        help='Select this if the input images are masked (i.e. with the brain extracted)')

    parser.add_argument('--gpu',
                        type=int,
                        help='ID of the GPU to use (default is taken from "CUDA_VISIBLE_DEVICES").')

    parser.add_argument('--synthseg',
                        action='store_true',
                        help=('Use SynthSeg to create a parcellation of the brain. Is will be used as a brain mask '
                              'and will allow the classification of segmented biomarkers by region.'))

    parser.add_argument('--synthseg_cpu',
                        action='store_true',
                        help='If selected, will run Synthseg using CPUs instead of GPUs')

    parser.add_argument('--synthseg_threads',
                        default=8,
                        type=int,
                        help='Number of threads to create for parallel computation when using --synthseg_cpu (default is 8).')

    parser.add_argument('--run_plugin',
                        default='Linear',
                        help=('Type of plugin used by Nipype to run the workflow.\n'
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))
    parser.add_argument('--run_plugin_args',  # hidden feature: you can also give a json string '{"arg1": val1, ...}'
                        type=str,
                        help=('Configuration file (.yml) for the plugin used by Nipype to run the workflow.\n'
                              'It will be imported as a dictionary and given plugin_args '
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))

    parser.add_argument('--keep_all',
                        action='store_true',
                        help='Keep all intermediary file, which is usually necessary for debugging.')

    parser.add_argument('--cpus',
                        default=10,
                        type=float,
                        help="Number of CPUs (can be fractions of CPUs) that the Apptainer image will use at most.")
    return parser


def main():
    parser = singParser()
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Minimal input
    input = "--in /mnt/data/input"
    output = "--out /mnt/data/output"
    pred = f"--prediction {' '.join(args.prediction)}"
    config = f"--model_config /mnt/config/{op.basename(args.config)}"  # Only for Shiva

    if not (op.exists(args.output) and op.isdir(args.output)):
        os.makedirs(args.output)

    # Convert the run_plugin_args yaml file to a json string to avoid mounting too many folders
    if args.run_plugin_args:
        if os.path.isfile(args.run_plugin_args):
            with open(args.run_plugin_args, 'r') as file:
                args.run_plugin_args = json.dumps(yaml.safe_load(file))
        # else: already json sring or empty, which is fine for both

    # Optional inputs (common Shiva and Synthseg)
    opt_args1_names = ['replace_t1',
                       'replace_swi',
                       'input_type',
                       'run_plugin',
                       'run_plugin_args',
                       'gpu']
    opt_args1 = [f'--{arg_name} {getattr(args, arg_name)}' for arg_name in opt_args1_names if getattr(args, arg_name)]
    bind_sublist = None
    if args.sub_list:
        bind_sublist = f"{op.dirname(op.abspath(args.sub_list))}:/mnt/sublist:rw"
        opt_args1.append(f"--sub_list /mnt/sublist/{op.basename(args.sub_list)}")
    elif args.exclusion_list:
        bind_sublist = f"{op.dirname(op.abspath(args.exclusion_list))}:/mnt/sublist:rw"
        opt_args1.append(f"--exclusion_list /mnt/sublist/{op.basename(args.exclusion_list)}")
    elif args.sub_names:
        opt_args1.append(f"--sub_names {' '.join(args.sub_names)}")

    # Synthseg precomputation
    if args.synthseg:
        args_ss = []
        if args.synthseg_cpu:
            args_ss.append("--synthseg_cpu")
            nv = ''  # nvidia support for GPU usage with Singularity
        else:
            nv = '--nv'
        args_ss.append(f"--threads {args.synthseg_threads}")
        sing_image_ss = f"{yaml_content['synthseg_image']}"
        bind_list_ss = [f"{args.input}:/mnt/data/input:rw", f"{args.output}:/mnt/data/output:rw"]
        if args.run_plugin_args:
            bind_list_ss.append(f"{op.dirname(op.abspath(args.run_plugin_args))}:/mnt/plugin:rw")
        if bind_sublist:
            bind_list_ss.append(bind_sublist)
        bind_ss = ','.join(bind_list_ss)
        command_list_ss = [
            f"singularity exec", nv,
            "--bind", bind_ss,
            sing_image_ss,
            "precomp_synthseg.py", input, output, pred
        ] + args_ss + opt_args1

        command_ss = ' '.join(command_list_ss)
        print(command_ss)
        # singularity exec --bind /scratch/nozais/test_shiva/MRI_anat_moi:/mnt/data/input:rw,/scratch/nozais/test_shiva/test_synthprecomp:/mnt/data/output:rw /scratch/nozais/test_shiva/synthseg_test.sif precomp_synthseg.py --in /mnt/data/input --out /mnt/data/output --synthseg_cpu

        res_proc = subprocess.run(command_ss, shell=True)
        if res_proc.returncode != 0:
            raise RuntimeError('The Synthseg process failed. Interrupting the Shiva process.')

    bind_model = f"{yaml_content['model_path']}:/mnt/model:ro"
    bind_input = f"{args.input}:/mnt/data/input:rw"
    bind_output = f"{args.output}:/mnt/data/output:rw"
    bind_config = f"{op.dirname(op.abspath(args.config))}:/mnt/config:rw"

    singularity_image = f"{yaml_content['apptainer_image']}"

    # Optional input only for Shiva
    opt_args2_names = ['db_name',
                       'replace_flair',
                       'masked',]
    opt_args2 = [f'--{arg_name} {getattr(args, arg_name)}' for arg_name in opt_args2_names if getattr(args, arg_name)]
    preproc = None
    if args.preproc_results:
        preproc = f"{args.preproc_results}:/mnt/preproc:rw"
        opt_args2.append("--preproc_results /mnt/preproc")

    bind_list = [bind_model, bind_input, bind_output, bind_config]
    if bind_sublist:
        bind_list.append(bind_sublist)
    if preproc:
        bind_list.append(preproc)
    if args.run_plugin_args:
        bind_plugin = f"{op.dirname(op.abspath(args.run_plugin_args))}:/mnt/plugin:rw"
        bind_list.append(bind_plugin)
    bind = ','.join(bind_list)

    command_list = [f"singularity exec --nv --bind", bind,
                    singularity_image,
                    "shiva --containerized_all", input, output, pred, config
                    ] + opt_args1 + opt_args2

    if args.keep_all:
        command_list.append('--keep_all')

    command = ' '.join(command_list)

    print(command)

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
