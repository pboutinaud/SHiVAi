#!/usr/bin/env python
import os
import argparse
import yaml
import json


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

    parser.add_argument('--db_name',
                        help='Name of the data-base the input scans originate from. It is only to add this detail in the report')

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

    parser.add_argument('--synthseg',
                        action='store_true',
                        help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

    parser.add_argument('--synthseg_cpu',
                        action='store_true',
                        help='If selected, will run Synthseg using CPUs instead of GPUs')

    parser.add_argument('--synthseg_threads',
                        default=8,
                        type=int,
                        help='Number of threads to create for parallel computation when using --synthseg_cpu (default is 8).')

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

    # parser.add_argument('--retry',
    #                     action='store_true',
    #                     help='Relaunch the pipeline from where it stopped')

    parser.add_argument('--anonymize',
                        action='store_true',
                        help='Anonymize the report')

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

    # parser.add_argument('--preproc_results',
    #                     type=str,
    #                     help=(
    #                         'Path to the results folder of a previous shiva run, containing all the preprocessed data.\n'
    #                         'Requires that all the subjects from the current subject list (as per the content of --in or --sub_list) '
    #                         'are available in the results folder. If you have subjects with missing preprocessed data, you will '
    #                         'need to run their processing separatly.'
    #                     ))

    parser.add_argument('--synthseg_precomp',
                        action='store_true',
                        help=(
                            "Option used when the Synthseg parcellation has already been computed AND is stored "
                            "in the process's result folder. This is specifically designed to work with the "
                            "'precomp_synthseg.py script called when using 'run_shiva' and using --containerized_all "
                            "here."))

    file_management = parser.add_mutually_exclusive_group()

    file_management.add_argument('--keep_all',
                                 action='store_true',
                                 help='Keep all intermediary file, which is usually necessary for debugging.')

    file_management.add_argument('--debug',
                                 action='store_true',
                                 help='Like --keep_all plus stop on first crash')

    file_management.add_argument('--remove_intermediates',
                                 action='store_true',
                                 help=('Remove the folder containing all the intermediary steps, keeping only the "results" folder.\n'
                                       'Obvioulsy not compatible with debugging or re-running the workflow.'))

    # Config file where lots of arguments are already written
    parser.add_argument('--model_config',
                        type=str,
                        help=('Configuration file (.yml) containing the information and parameters for the '
                              'AI model (as well as the path to the AppTainer container when used).\n'
                              'Using a configuration file is incompatible with the arguments listed below '
                              '(i.e. --model --percentile --threshold --threshold_clusters --final_dimensions '
                              '--voxels_size --interpolation --brainmask_descriptor --pvs_descriptor '
                              '--pvs2_descriptor --wmh_descriptor --cmb_descriptor, --lac_descriptor).'),
                        default=None)

    # Manual input if no config file
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

    parser.add_argument('--threshold_pvs',
                        type=float,
                        default=0.5,
                        help='Threshold to compute PVS clusters metrics')

    parser.add_argument('--threshold_wmh',
                        type=float,
                        default=0.2,
                        help='Threshold to compute WMH clusters metrics')

    parser.add_argument('--threshold_cmb',
                        type=float,
                        default=0.5,
                        help='Threshold to compute CMB clusters metrics')

    parser.add_argument('--threshold_lac',
                        type=float,
                        default=0.2,
                        help='Threshold to compute lacuna clusters metrics')

    parser.add_argument('--threshold_clusters',
                        type=float,
                        help=('Unique threshold to compute clusters metrics '
                              'for all biomarkers, override biomarker-specific '
                              'thresholds'))

    parser.add_argument('--min_pvs_size',
                        type=int,
                        default=5,
                        help='Size (in voxels at "voxels_size") below which segmented PVSs are discarded')

    parser.add_argument('--min_wmh_size',
                        type=int,
                        default=1,
                        help='Size (in voxels at "voxels_size") below which segmented WMHs are discarded')

    parser.add_argument('--min_cmb_size',
                        type=int,
                        default=1,
                        help='Size (in voxels at "voxels_size") below which segmented CMBs are discarded')

    parser.add_argument('--min_lac_size',
                        type=int,
                        default=3,
                        help='Size (in voxels at "voxels_size") below which segmented lacunas are discarded')

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

    def parse_sub_list_file(filename):
        list_path = os.path.abspath(filename)
        sub_list = []
        sep_chars = [' ', ';', '|']
        if not os.path.exists(list_path):
            raise FileNotFoundError(f'The participant list file was not found at the given location: {list_path}')
        with open(list_path) as f:
            lines = f.readlines()
        for line in lines:
            line_s = line.strip('\n')
            # replacing potential separators with commas
            for sep in sep_chars:
                if sep in line_s:
                    line_s = line_s.replace(sep, ',')
            subs = line_s.split(',')
            sub_list += [s.strip() for s in subs if s]
        return sub_list

    args = inParser.parse_args()
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    if args.debug:
        args.keep_all = True

    # Checks and parsing of subjects
    subject_list = os.listdir(args.input)
    if args.sub_list is None and args.sub_names is None:
        if args.exclusion_list:
            args.exclusion_list = parse_sub_list_file(args.exclusion_list)
            subject_list = sorted(list(set(subject_list) - set(args.exclusion_list)))
        args.sub_list = subject_list
    else:
        if args.sub_list:
            args.sub_list = parse_sub_list_file(args.sub_list)
        elif args.sub_names:
            args.sub_list = args.sub_names
        subs_not_in_dir = set(args.sub_list) - set(subject_list)
        if len(subs_not_in_dir) == len(args.sub_list):
            raise inParser.error('None of the participant IDs given in the sub_list file was found in the input directory.\n'
                                 f'Participant IDs given: {args.sub_list}\n'
                                 f'Participant available: {subject_list}')
        elif len(subs_not_in_dir) > 0:
            raise inParser.error(f'Some participants where not found in the input directory: {sorted(list(subs_not_in_dir))}')

    # Parse the thresholds
    if args.threshold_clusters is not None:
        if args.model_config:
            raise inParser.error('Both "--threshold_clusters" and "--model_config" were given as argument, '
                                 'but only one of them can be used at the same time.')
        # threshold_clusters override the others
        args.threshold_pvs = args.threshold_clusters
        args.threshold_wmh = args.threshold_clusters
        args.threshold_cmb = args.threshold_clusters
        args.threshold_lac = args.threshold_clusters

    # Parse the config file
    if args.model_config:
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
        args.threshold_pvs = parameters['threshold_pvs']
        args.threshold_wmh = parameters['threshold_wmh']
        args.threshold_cmb = parameters['threshold_cmb']
        args.threshold_lac = parameters['threshold_lac']
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

    if args.synthseg_precomp:
        args.synthseg = True

    # Check containerizing options
    if (args.containerized_all or args.containerized_nodes) and not args.container_image:
        inParser.error(
            'Using a container (with the "--containerized_all" or "containerized_nodes" arguments) '
            'requires a container image (.sif file) but none was given. Add its path --container_image '
            'or in the configuration file (.yaml file).')
    if args.containerized_nodes and (args.synthseg and not args.synthseg_precomp) and not args.synthseg_image:
        inParser.error(
            'Using the "containerized_nodes" option with synthseg, but no synthseg apptainer image was provided')

    # Parse the plugin arguments
    if args.run_plugin_args:
        if os.path.isfile(args.run_plugin_args):
            with open(args.run_plugin_args, 'r') as file:
                yaml_content = yaml.safe_load(file)
            args.run_plugin_args = yaml_content
        else:
            args.run_plugin_args = json.loads(args.run_plugin_args)
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

    # Parse the prediction
    if 'all' in args.prediction:
        if 'PVS' in args.prediction:
            args.prediction = ['PVS', 'WMH', 'CMB', 'LAC']
        else:
            args.prediction = ['PVS2', 'WMH', 'CMB', 'LAC']
    if not isinstance(args.prediction, list):  # When only one input
        args.prediction = [args.prediction]

    # Check the preprocessing files input when given
    setattr(args, 'preproc_results', None)  # TODO: remove when preproc_results updated
    if args.preproc_results is not None:
        args.preproc_results = os.path.abspath(args.preproc_results)
        if not os.path.exists(args.preproc_results):
            raise FileNotFoundError(
                f'The folder containing the results from the previous processing was not found: {args.preproc_results}'
            )
        dir_list = os.listdir(args.preproc_results)
        dir_name = os.path.basename(args.preproc_results)
        err_msg = (
            'The folder containing the results  from the previous processing should either be the "shiva_preproc" '
            f'folder or the folder containing the "shiva_preproc" folder, but it is not the case: {args.preproc_results}'
        )
        if not dir_name == 'shiva_preproc':
            if 'shiva_preproc' in dir_list:
                args.preproc_results = os.path.join(args.preproc_results, 'shiva_preproc')
            elif 'results' in dir_list:
                args.preproc_results = os.path.join(args.preproc_results, 'results')
                dir_list2 = os.listdir(args.preproc_results)
                if 'shiva_preproc' in dir_list2:
                    args.preproc_results = os.path.join(args.preproc_results, 'shiva_preproc')
                else:
                    raise FileNotFoundError(err_msg)
            else:
                raise FileNotFoundError(err_msg)
    return args
