#!/usr/bin/env python
import os
import argparse
import yaml


def shivaParser():
    DESCRIPTION = """SHIVA pipeline for deep-learning imaging biomarkers computation. Performs resampling and coregistration 
                of a set of structural NIfTI head image, followed by intensity normalization, and cropping centered on the brain.
                A nipype workflow is used to preprocess a lot of images at the same time.
                The segmentation from the wmh, cmb and pvs models are generated depending on the inputs. A Report is generated.
                
                Input data can be staged in BIDS or a simplified file arborescence, or described with a JSON file (for the 3D Slicer extension)."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='input',
                        help='Folder path with files, BIDS structure folder path or JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/folder/structure',
                        required=True)

    parser.add_argument('--out', dest='output',
                        type=str,
                        help='Output folder path (nipype working directory)',
                        metavar='path/to/nipype_work_dir',
                        required=True)

    parser.add_argument('--input_type',
                        choices=['standard', 'BIDS', 'json'],
                        help="Way to grab and manage nifti files : 'standard', 'BIDS' or 'json'",
                        default='standard')

    parser.add_argument('--sub_list',
                        type=str,
                        required=False,
                        help=('Text file containing the list of participant IDs to be processed. The IDs must be '
                              'the same as the ones given in the input folder. In the file, the IDs can be separated '
                              'by a whitespace, a comma, or a new line (or a combination of those). If this argument '
                              'is not given, all the participants in the input folder will be processed'))

    parser.add_argument('--prediction',
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'LAC', 'all'],
                        nargs='+',
                        help=("Choice of the type of prediction (i.e. segmentation) you want to compute.\n"
                              "A combination of multiple predictions (separated by a white space) can be given.\n"
                              "- 'PVS' for the segmentation of perivascular spaces using only T1 scans\n"
                              "- 'PVS2' for the segmentation of perivascular spaces using both T1 and FLAIR scans\n"
                              "- 'WMH' for the segmentation of white matter hyperintensities (requires both T1 and FLAIR scans)\n"
                              "- 'CMB' for the segmentation of cerebral microbleeds (requires SWI scans)\n"
                              "- 'LAC' for the segmentation of cerebral lacunas (requires both T1 and FLAIR scans)\n"
                              "- 'all' for doing 'PVS2', 'WMH', and 'CMB' segmentation (requires T1, FLAIR, and SWI scans)"),
                        default=['PVS'])

    parser.add_argument('--synthseg',
                        action='store_true',
                        help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

    parser.add_argument('--masked',
                        action='store_true',
                        help='Select this if the input images are masked (i.e. with the brain extracted)')

    parser.add_argument('--gpu',
                        type=int,
                        help='ID of the GPU to use (default is taken from "CUDA_VISIBLE_DEVICES").')

    parser.add_argument('--mask_on_gpu',
                        action='store_true',
                        help='Use GPU to compute the brain mask.')

    container_args = parser.add_mutually_exclusive_group()

    container_args.add_argument('--containerized_all',
                                help='Used when the whole process is launched from inside a container',
                                action='store_true')

    container_args.add_argument('--containerized_nodes',
                                help='Used when the process uses the container to run specific nodes (prediction and registration)',
                                action='store_true')

    parser.add_argument('--retry',
                        action='store_true',
                        help='Relaunch the pipeline from where it stopped')

    parser.add_argument('--anonymize',
                        action='store_true',
                        help='Anonymize the report')

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

    parser.add_argument('--node_plugin_args',
                        type=str,
                        help=('Configuration file (.yml) for the plugin used by Nipype to run some specific nodes. '
                              'It will be imported as a dictionary of dictionaries. The root keys refer to the nodes '
                              '("pred" and "reg" available now, for the predictions and registration nodes) and the '
                              'dictionary they are referring to are "plugin_args" so they must follow the Nipype '
                              'syntax for this argument (see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))

    parser.add_argument('--max_jobs',
                        type=int,
                        default=50,
                        help='Number of jobs to run and queue simultaneously when using an HPC plugin (like SLURM). Default is 50.'
                        )

    parser.add_argument('--prev_qc',
                        type=str,
                        default=None,
                        help=('CSV file from a previous QC with the metrics computed on other participants '
                              'preprocessing. This data will be used to estimate outliers and thus help detect '
                              'participants that may have a faulty preprocessing'))

    parser.add_argument('--keep_all',
                        action='store_true',
                        help='Keep all intermediary file, which is usually necessary for debugging.')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Like --retry plus stop on first crash')

    parser.add_argument('--model_config',
                        type=str,
                        help=('Configuration file (.yml) containing the information and parameters for the '
                              'AI model (as well as the path to the AppTainer container when used).\n'
                              'Using a configuration file is incompatible with the arguments listed below '
                              '(i.e. --model --percentile --threshold --threshold_clusters --final_dimensions '
                              '--voxels_size --interpolation --brainmask_descriptor --pvs_descriptor '
                              '--pvs2_descriptor --wmh_descriptor --cmb_descriptor, --lac_descriptor).'),
                        default=None)

    # Manual input
    parser.add_argument('--container_image',
                        default=None,
                        help='path to the SHIV-AI apptainer image (.sif file)')

    parser.add_argument('--synthseg_image',
                        default=None,
                        help='path to the synthseg apptainer image (.sif file)')

    parser.add_argument('--model',
                        default=None,
                        help='path to the AI model weights and descriptors')

    parser.add_argument('--percentile',
                        type=float,
                        default=99,
                        help='Percentile of the data to keep when doing image normalisation (to remove hotspots)')

    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Threshold to binarise estimated brain mask')

    parser.add_argument('--threshold_clusters',
                        type=float,
                        default=0.2,
                        help='Threshold to compute clusters metrics')

    parser.add_argument('--min_pvs_size',
                        type=int,
                        default=5,
                        help='Size (in voxels) below which segmented PVSs are discarded')

    parser.add_argument('--min_wmh_size',
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented WMHs are discarded')

    parser.add_argument('--min_cmb_size',
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented CMBs are discarded')

    parser.add_argument('--min_lac_size',
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented lacunas are discarded')

    parser.add_argument('--final_dimensions',
                        nargs=3, type=int,
                        default=(160, 214, 176),
                        help='Final image array size in i, j, k.')

    parser.add_argument('--voxels_size', nargs=3,
                        type=float,
                        default=(1.0, 1.0, 1.0),
                        help='Voxel size of final image')

    parser.add_argument('--interpolation',
                        type=str,
                        default='WelchWindowedSinc',
                        help='final interpolation apply to the t1 image')

    parser.add_argument('--brainmask_descriptor',
                        type=str,
                        default='brainmask/V0/model_info.json',
                        help='brainmask descriptor file path')

    parser.add_argument('--pvs_descriptor',
                        type=str,
                        default='T1-PVS/V1/model_info.json',
                        help='pvs descriptor file path')

    parser.add_argument('--pvs2_descriptor',
                        type=str,
                        default='T1.FLAIR-PVS/V0/model_info.json',
                        help='pvs dual descriptor file path')

    parser.add_argument('--wmh_descriptor',
                        type=str,
                        default='T1.FLAIR-WMH/V1/model_info.json',
                        help='wmh descriptor file path')

    parser.add_argument('--cmb_descriptor',
                        type=str,
                        default='SWI-CMB/V1/model_info.json',
                        help='cmb descriptor file path')

    parser.add_argument('--lac_descriptor',
                        type=str,
                        default='T1.FLAIR-LAC/model_info.json',
                        help='Lacuna descriptor file path')

    return parser


def set_args_and_check(inParser):

    args = inParser.parse_args()
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    if args.debug:
        args.retry = True
    if args.retry:
        args.keep_all = True

    if (os.path.isdir(args.output)
        and bool(os.listdir(args.output))
            and not args.retry):
        inParser.error(
            'The output directory already exists and is not empty.'
        )

    subject_list = os.listdir(args.input)
    if args.sub_list is None:
        args.sub_list = subject_list
    else:
        list_path = os.path.abspath(args.sub_list)
        args.sub_list = []
        if not os.path.exists(list_path):
            raise FileNotFoundError(f'The participant list file was not found at the given location: {list_path}')
        with open(list_path) as f:
            lines = f.readlines()
        for line in lines:
            line_s = line.strip('\n')
            subs = line_s.split(',')
            subs = [s.strip() for s in subs]
            for sub in subs:
                if ' ' in sub:
                    subs2 = sub.split()  # if sep is whitespace
                    for sub2 in subs2:
                        if len(sub):
                            args.sub_list.append(sub2)
                else:
                    if len(sub):
                        args.sub_list.append(sub)
        subs_not_in_dir = set(args.sub_list) - set(subject_list)
        if len(subs_not_in_dir) == len(args.sub_list):
            raise ValueError('None of the participant IDs given in the sub_list file was found in the input directory.\n'
                             f'Participant IDs given: {args.sub_list}\n'
                             f'Participant available: {subject_list}')
        elif len(subs_not_in_dir) > 0:
            raise ValueError(f'Some participants where not found in the input directory: {sorted(list(subs_not_in_dir))}')

    if args.model_config:  # Parse the config file
        args.model_config = os.path.abspath(args.model_config)
        with open(args.model_config, 'r') as file:
            yaml_content = yaml.safe_load(file)
        if args.containerized_all or args.containerized_nodes:
            args.container_image = yaml_content['apptainer_image']
        if args.synthseg:
            args.synthseg_image = yaml_content['synthseg_image']
        parameters = yaml_content['parameters']
        args.model = yaml_content['model_path']  # only used when not with container
        args.percentile = parameters['percentile']
        args.threshold = parameters['threshold']
        args.threshold_clusters = parameters['threshold_clusters']
        if 'min_pvs_size' in parameters.keys():
            args.min_pvs_size = parameters['min_pvs_size']
        if 'min_wmh_size' in parameters.keys():
            args.min_wmh_size = parameters['min_wmh_size']
        if 'min_cmb_size' in parameters.keys():
            args.min_cmb_size = parameters['min_cmb_size']
        if 'min_lac_size' in parameters.keys():
            args.min_lac_size = parameters['min_lac_size']
        args.final_dimensions = tuple(parameters['final_dimensions'])
        args.voxels_size = tuple(parameters['voxels_size'])
        args.interpolation = parameters['interpolation']
        args.brainmask_descriptor = parameters['brainmask_descriptor']
        args.pvs_descriptor = parameters['PVS_descriptor']
        args.pvs2_descriptor = parameters['PVS2_descriptor']
        args.wmh_descriptor = parameters['WMH_descriptor']
        args.cmb_descriptor = parameters['CMB_descriptor']
        args.lac_descriptor = parameters['LAC_descriptor']
    args.model = os.path.abspath(args.model)

    if (args.containerized_all or args.containerized_nodes) and not args.container_image:
        inParser.error(
            'Using a container (with the "--containerized_all" or "containerized_nodes" arguments) '
            'requires a container image (.sif file) but none was given. Add its path --container_image '
            'or in the configuration file (.yaml file).')

    if args.run_plugin_args:  # Parse the plugin arguments
        with open(args.run_plugin_args, 'r') as file:
            yaml_content = yaml.safe_load(file)
        args.run_plugin_args = yaml_content
    else:
        args.run_plugin_args = {}
    args.run_plugin_args['max_jobs'] = args.max_jobs

    if args.node_plugin_args:
        with open(args.node_plugin_args, 'r') as file:
            yaml_content = yaml.safe_load(file)
        args.node_plugin_args = yaml_content
    else:
        args.node_plugin_args = {}

    if args.containerized_all:
        args.model = '/mnt/model'

    if 'all' in args.prediction:
        args.prediction = ['PVS2', 'WMH', 'CMB', 'LAC']
    if not isinstance(args.prediction, list):  # When only one input
        args.prediction = [args.prediction]
    return args
