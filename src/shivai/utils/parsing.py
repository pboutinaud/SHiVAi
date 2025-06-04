#!/usr/bin/env python
import os
import argparse
import yaml
import json
import xml.etree.ElementTree as ET
import pandas as pd


def shivaParser():
    DESCRIPTION = """SHIVA pipeline for deep-learning imaging biomarkers computation. Performs resampling and coregistration
                of a set of structural NIfTI head image, followed by intensity normalization, and cropping centered on the brain.
                A nipype workflow is used to preprocess a lot of images at the same time.
                The segmentation from the wmh, cmb and pvs models are generated depending on the inputs. A Report is generated.

                Input data can be staged in BIDS or a simplified file arborescence, or described with a JSON file (for the 3D Slicer extension)."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='in_dir',
                        help='Folder path with files, BIDS structure folder path or JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/folder/structure',
                        required=True)

    parser.add_argument('--out', dest='out_dir',
                        type=str,
                        help='Output folder path (nipype working directory)',
                        metavar='path/to/nipype_work_dir',
                        required=True)

    parser.add_argument('--input_type',
                        choices=['standard', 'BIDS', 'swomed'],  # , 'json'
                        help="Way to grab and manage nifti files : 'standard' (default) or 'BIDS'",
                        default='standard')

    parser.add_argument('--db_name',
                        help='Name of the data-base the input scans originate from. It is only to add this detail in the report')

    parser.add_argument('--file_type',
                        choices=['nifti', 'dicom'],  # , 'json'
                        help="The type of files input: 'nifti' (default) or 'dicom'",
                        default='nifti')

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

    parser.add_argument('--use_t1',
                        action='store_true',
                        help=('Can be used when predicting CMBs only (so only expecting SWI acquisitions) while T1 acquisitions '
                              'are available. This enable the CMB preprocessing steps using t1 for the brain parcelization. '
                              'This option can also be used with "replace_t1" to use another type of acquisition.'))

    parser.add_argument('--brain_seg',
                        choices=['shiva', 'shiva_gpu', 'synthseg', 'synthseg_cpu', 'synthseg_precomp', 'fs_precomp', 'premasked', 'custom'],
                        help=('Type of brain segmentation used in the pipeline\n'
                              '- "shiva" uses an inhouse AI model to create a simple brain mask. By default it runs on CPUs.\n'
                              '- "shiva_gpu" is the same as "shiva" but runs on a GPU\n'
                              '- "synthseg" uses the Synthseg, AI-based, parcellation scheme from FreeSurfer to give a full '
                              'parcellation of the brain and adapted region-wise metrics of the segmented biomarkers. It uses '
                              'a GPU by default.\n'
                              '- "synthseg_cpu" is the same as "synthseg" but running on CPUs (which number can be controlled '
                              'with "--ai_threads"), and is thus much slower\n'
                              '- "synthseg_precomp" is used when the synthseg parcellisation was precomputed and is already '
                              'stored in the results (typically used by the run_shiva.py script)\n'
                              '- "premasked" is to be used if the input images are already masked/brain-extracted\n'
                              '- "custom" considers that you provide a custom brain segmentation. If the said segmentation is '
                              'a full brain parcellation for region-wise analysis, a LUT must be provided with the "--custom_LUT"'
                              'argument. Otherwise, the segmentation is considered simply as a brain mask.\n'
                              '- "fs_precomp" considers that, in the input folder, you provide brain segmentations created by FreeSurfer. '
                              'It can be .nii(.gz) or .mgz files, but the filename must be "aparc+aseg" (e.g. "aparc+aseg.mgz")'),
                        default='shiva')

    parser.add_argument('--ss_qc',
                        action='store_true',
                        help='Runs the SynthSeg QC if --brain_seg synthseg* is selected too.')

    parser.add_argument('--ss_vol',
                        action='store_true',
                        help='Runs the SynthSeg volumes if --brain_seg synthseg* is selected too.')

    parser.add_argument('--ai_threads',
                        default=8,
                        type=int,
                        help=('Number of threads (default is 8) to create for parallel computation when using cpu to compute AI-based parcellation. '
                              'This involve the following options "--brain_seg shiva", "--brain_seg synthseg_cpu", and "--use_cpu".')
                        )

    parser.add_argument('--custom_LUT',
                        type=str,
                        help=('Look-up table (LUT) for the association between values and region name in a custom brain segmentation '
                              '(when using "custom" in "--brain_seg").\n'
                              'If no LUT is provided, the custom segmentation is assumed to be a simple brain mask.\n'
                              'The different accepted LUT styles are:\n'
                              '- .json file with paired brain region names and integer labels (keys and values can be either, but '
                              'stay consistent in the file)\n'
                              '- BIDS style .tsv file\n'
                              '- FSL style .lut file\n'
                              '- FSL style .xml file\n'
                              '- FreeSurfer style .txt file'))

    parser.add_argument('--preproc_only',
                        action='store_true',
                        help=('If used, only the preprocessing steps will be run (usefull for training new data for example).\n'
                              'This option still needs the "--prediction" argument to know what type of input will be given\n'
                              'for the preprocessing.'))

    parser.add_argument('--use_cpu',
                        action='store_true',
                        help='If selected, will ignore available GPU(s) and run the segmentations on CPUs. Be aware that some models may not be compatible with this option.')

    parser.add_argument('--swomed_parc',  # Hidden option overriding 'brain_seg', used in SWOMed to give the path to the synthseg parcelation
                        required=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--swomed_ssvol',  # Hidden option used in SWOMed to give the path to the synthseg vol.csv file
                        required=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--swomed_ssqc',  # Hidden option used in SWOMed to give the path to the synthseg qc.csv file
                        required=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--swomed_t1',  # Hidden option overriding 't1', used in SWOMed to give the path to the t1 (or equivalent)
                        required=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--swomed_flair',  # Hidden option overriding 'brain_seg', used in SWOMed to give the path to the flair (or equivalent)
                        required=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--swomed_swi',  # Hidden option overriding 'brain_seg', used in SWOMed to give the path to the swi (or equivalent)
                        required=False,
                        help=argparse.SUPPRESS)

    container_args = parser.add_mutually_exclusive_group()

    container_args.add_argument('--containerized_all',
                                help='Used when the whole process is launched from inside a container',
                                action='store_true')

    container_args.add_argument('--containerized_nodes',
                                help='Used when the process uses the container to run specific nodes (prediction and registration)',
                                action='store_true')

    parser.add_argument('--local_synthseg',
                        action='store_true',
                        help='If selected, overrides the --containerized_nodes option for synthseg, using the local installation instead')

    parser.add_argument('--prereg_flair',
                        action='store_true',
                        help='If selected, the FLAIR images are concidered pre-registered to the the T1 images (no additional registration will be done).')

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

    parser.add_argument('--preproc_results',
                        type=str,
                        help=(
                            'Path to the results folder of a previous shiva run, containing all the preprocessed data.\n'
                            'Requires that all the subjects from the current subject list (as per the content of --in or --sub_list) '
                            'are available in the results folder. If you have subjects with missing preprocessed data, you will '
                            'need to run their processing separatly.'
                        ))

    file_management = parser.add_mutually_exclusive_group()

    file_management.add_argument('--keep_all',
                                 action='store_true',
                                 help='Keep all intermediary file')

    file_management.add_argument('--debug',
                                 action='store_true',
                                 help='Like --keep_all plus stop on first crash')

    file_management.add_argument('--remove_intermediates',
                                 action='store_true',
                                 help=('Remove the folder containing all the intermediary steps, keeping only the "results" folder.\n'
                                       'Obvioulsy not compatible with debugging or re-running the workflow.'))

    # Config file where lots of arguments are already written
    parser.add_argument('--config',
                        type=str,
                        help=('Configuration file (.yml) containing the information and parameters for the '
                              'AI model (as well as the path to the AppTainer container when used).\n'
                              'Using a configuration file is incompatible with the arguments listed below '
                              '(i.e. --model --swi_file_num --percentile --threshold --threshold_clusters '
                              '--final_dimensions --voxels_size --voxels_tolerance --interpolation --brainmask_descriptor --pvs_descriptor '
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

    parser.add_argument('--swi_file_num',
                        type=int,
                        help='Index (starting at 0) of the SWI file to select after DICOM to NIfTI conversion (i.e. which echo to chose)')

    parser.add_argument('--percentile',
                        type=float,
                        # default=99,
                        help='Percentile of the data to keep when doing image normalisation (to remove hotspots)')

    parser.add_argument('--threshold',
                        type=float,
                        # default=0.5,
                        help='Threshold to binarise estimated brain mask')

    parser.add_argument('--threshold_pvs',
                        type=float,
                        # default=0.5,
                        help='Threshold to compute PVS clusters metrics')

    parser.add_argument('--threshold_wmh',
                        type=float,
                        # default=0.2,
                        help='Threshold to compute WMH clusters metrics')

    parser.add_argument('--threshold_cmb',
                        type=float,
                        # default=0.5,
                        help='Threshold to compute CMB clusters metrics')

    parser.add_argument('--threshold_lac',
                        type=float,
                        # default=0.2,
                        help='Threshold to compute lacuna clusters metrics')

    parser.add_argument('--threshold_clusters',
                        type=float,
                        help=('Unique threshold to compute clusters metrics '
                              'for all biomarkers, override biomarker-specific '
                              'thresholds'))

    parser.add_argument('--min_pvs_size',
                        type=int,
                        # default=5,
                        help='Size (in voxels at "voxels_size") below which segmented PVSs are discarded')

    parser.add_argument('--min_wmh_size',
                        type=int,
                        # default=1,
                        help='Size (in voxels at "voxels_size") below which segmented WMHs are discarded')

    parser.add_argument('--min_cmb_size',
                        type=int,
                        # default=1,
                        help='Size (in voxels at "voxels_size") below which segmented CMBs are discarded')

    parser.add_argument('--min_lac_size',
                        type=int,
                        # default=3,
                        help='Size (in voxels at "voxels_size") below which segmented lacunas are discarded')

    parser.add_argument('--final_dimensions',
                        nargs=3, type=int,
                        # default=(160, 214, 176),
                        help='Final image array size in i, j, k.')

    parser.add_argument('--voxels_size', nargs=3,
                        type=float,
                        # default=(1.0, 1.0, 1.0),
                        help='Voxel size of final image')

    parser.add_argument('--voxels_tolerance', nargs=3,
                        type=float,
                        # default=(0.0, 0.0, 0.0),
                        help=('Tolerance on the original voxel size: if the difference between the original '
                              'voxel size and the final voxel size (give by voxels_size) is less than the '
                              'tolerance, then the original voxel size is kept.'))

    parser.add_argument('--interpolation',
                        type=str,
                        # default='WelchWindowedSinc',
                        help='final interpolation apply to the t1 image')

    parser.add_argument('--brainmask_descriptor',
                        type=str,
                        # default='brainmask/V0/model_info.json',
                        help='brainmask descriptor file path')

    parser.add_argument('--pvs_descriptor',
                        type=str,
                        # default='T1-PVS/V1/model_info.json',
                        help='pvs descriptor file path')

    parser.add_argument('--pvs2_descriptor',
                        type=str,
                        # default='T1.FLAIR-PVS/V0/model_info.json',
                        help='pvs dual descriptor file path')

    parser.add_argument('--wmh_descriptor',
                        type=str,
                        # default='T1.FLAIR-WMH/V1/model_info.json',
                        help='wmh descriptor file path')

    parser.add_argument('--cmb_descriptor',
                        type=str,
                        # default='SWI-CMB/V1/model_info.json',
                        help='cmb descriptor file path')

    parser.add_argument('--lac_descriptor',
                        type=str,
                        # default='T1.FLAIR-LAC/model_info.json',
                        help='Lacuna descriptor file path')

    return parser


def parse_LUT(inLUT):  # TODO: tester avec de vraies LUT
    '''
    Parse the input LUT file into a dict
    '''
    # when inLUT is a json file
    if os.path.splitext(inLUT)[-1] == '.json':
        with open(inLUT, 'r') as jsonLUT:
            dictLUT = json.load(jsonLUT)
        if all(isinstance(k, str) for k in dictLUT) and all(isinstance(val, int) for val in dictLUT.values()):
            pass  # All is well and good
        elif all(isinstance(k, int) for k in dictLUT) and all(isinstance(val, str) for val in dictLUT.values()):
            # keys and values must be swapped
            dictLUT = {val: k for k, val in dictLUT.items()}
        else:
            raise ValueError('Unrecognized key:value structure in the LUT. One must be str and the other int')

    # when inLUT is a tsv file (see )
    if os.path.splitext(inLUT)[-1] == '.tsv':
        dictLUT = {}
        dfLUT = pd.read_csv(inLUT, sep='\t+', engine='python')
        if 'abbreviation' in dfLUT:
            regcol = 'abbreviation'
        else:
            regcol = 'name'
        for _, row in dfLUT.iterrows():
            reg = row[regcol]
            val = int(row['index'])
            dictLUT[reg] = val

    # when inLUT is a FSL style .lut LUT
    if os.path.splitext(inLUT)[-1] == '.lut':
        dictLUT = {}
        with open(inLUT, 'r') as lutLUT:
            for line in lutLUT.readlines():
                line = line.strip()
                line = line.split(' ')
                line = [val for val in line if val != '']  # filtering out additional whitespace spots
                if line:
                    if line[0].isdigit():
                        val = int(line[0])
                        reg = ' '.join(line[4:])  # Region name can have spaces and starts in 5th position (after label and RGB)
                        dictLUT[reg] = val

    # when inLUT is a FSL style .xml LUT
    if os.path.splitext(inLUT)[-1] == '.xml':
        dictLUT = {}
        # import xml.etree.ElementTree as ET
        tree = ET.parse(inLUT)
        root = tree.getroot()
        for label in root.iter('label'):
            val = int(label.get('index'))
            reg = label.text
            dictLUT[reg] = val

    # when inLUT is a FreeSurfer style .txt LUT (e.g. https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT)
    if os.path.splitext(inLUT)[-1] == '.txt':
        dictLUT = {}
        with open(inLUT, 'r') as lutLUT:
            for line in lutLUT.readlines():
                line = line.strip()
                line = line.split(' ')
                line = [val for val in line if val != '']
                if line:
                    if line[0].isdigit():
                        val = int(line[0])
                        reg = line[1]
                        dictLUT[reg] = val

    return dictLUT


def set_args_and_check(inParser):

    def parse_sub_list_file(filename):
        list_path = os.path.abspath(filename)
        sub_list = []
        sep_chars = [' ', ';', '|']
        if not os.path.exists(list_path):
            raise ValueError(f'The participant list file was not found at the given location: {list_path}')
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
    args.in_dir = os.path.abspath(args.in_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    if args.debug:
        args.keep_all = True

    # Check if there is a LUT with the custom seg and parse it
    if args.brain_seg == 'custom' and args.custom_LUT:
        args.custom_LUT = os.path.abspath(args.custom_LUT)
        if not os.path.exists(args.custom_LUT):
            raise inParser.error(f'Using the "custom" segmentation with a LUT but the file given with '
                                 f'--custom_LUT was not found: {args.custom_LUT}')
        args.custom_LUT = parse_LUT(args.custom_LUT)

    if args.file_type == 'dicom' and args.input_type == 'BIDS':
        raise inParser.error('BIDS data structure not compatible with DICOM input')

    # Checks and parsing of subjects
    subject_list = os.listdir(args.in_dir)
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

    # Check the SWOMed direct input paths
    if args.swomed_parc is not None and not os.path.exists(args.swomed_parc):
        inParser.error(f'The parcellation given to "--swomed_parc" was not found: {args.swomed_parc}')
    if args.swomed_t1 is not None and not os.path.exists(args.swomed_t1):
        inParser.error(f'The t1 given to "--swomed_t1" was not found: {args.swomed_t1}')
    if args.swomed_flair is not None and not os.path.exists(args.swomed_flair):
        inParser.error(f'The flair given to "--swomed_flair" was not found: {args.swomed_flair}')
    if args.swomed_swi is not None and not os.path.exists(args.swomed_swi):
        inParser.error(f'The swi given to "--swomed_swi" was not found: {args.swomed_swi}')

    # Parse the thresholds
    if args.threshold_clusters is not None:
        if args.config:
            raise inParser.error('Both "--threshold_clusters" and "--config" were given as argument, '
                                 'but only one of them can be used at the same time.')
        # threshold_clusters override the others
        args.threshold_pvs = args.threshold_clusters
        args.threshold_wmh = args.threshold_clusters
        args.threshold_cmb = args.threshold_clusters
        args.threshold_lac = args.threshold_clusters

    # Parse the config file
    if args.config:
        config_params = ['percentile', 'threshold', 'threshold_pvs', 'threshold_wmh',
                         'threshold_cmb', 'threshold_lac', 'min_pvs_size', 'min_wmh_size',
                         'min_cmb_size', 'min_lac_size', 'interpolation']
        args.config = os.path.abspath(args.config)
        with open(args.config, 'r') as file:
            yaml_content = yaml.safe_load(file)
        if args.containerized_all or args.containerized_nodes:
            args.container_image = yaml_content['apptainer_image']
            if 'synthseg' in args.brain_seg and not args.brain_seg == 'synthseg_precomp':
                args.synthseg_image = yaml_content['synthseg_image']
        parameters = yaml_content['parameters']
        for param in config_params:
            if getattr(args, param) is None:  # Giving param as argument to the command line overrides the config.yml params
                setattr(args, param, parameters[param])
        if args.model is None and 'model_path' in yaml_content.keys():
            args.model = yaml_content['model_path']  # only used when not with container, otherwise set to '/mnt/model' (see below)

        if args.final_dimensions is None:
            args.final_dimensions = tuple(parameters['final_dimensions'])
        if args.voxels_size is None:
            args.voxels_size = tuple(parameters['voxels_size'])
        if args.voxels_tolerance is None:
            if 'voxels_tolerance' in parameters:
                args.voxels_tolerance = tuple(parameters['voxels_tolerance'])
            else:
                args.voxels_tolerance = (0.0, 0.0, 0.0)

        # Checking and setting the model descriptors (not checking md5 yet though)
        if args.brainmask_descriptor is None:  # otherwise override the config file when manually inputing the descriptor file
            if 'brainmask_descriptor' in parameters:
                args.brainmask_descriptor = parameters['brainmask_descriptor']
            elif 'shiva' in args.brain_seg:
                inParser.error('The model descriptor (json file) for the shiva brain masking model has not been specified')
        if args.pvs_descriptor is None:
            if 'PVS_descriptor' in parameters:
                args.pvs_descriptor = parameters['PVS_descriptor']
            elif 'PVS' in args.prediction:
                inParser.error('The model descriptor (json file) for the monomodal PVS model has not been specified')
        if args.pvs2_descriptor is None:
            if 'PVS2_descriptor' in parameters:
                args.pvs2_descriptor = parameters['PVS2_descriptor']
            elif 'PVS2' in args.prediction or ('all' in args.prediction and 'PVS' not in args.prediction):
                inParser.error('The model descriptor (json file) for the bimodal PVS model has not been specified')
        if args.wmh_descriptor is None:
            if 'WMH_descriptor' in parameters:
                args.wmh_descriptor = parameters['WMH_descriptor']
            elif 'WMH' in args.prediction or 'all' in args.prediction:
                inParser.error('The model descriptor (json file) for the WMH model has not been specified')
        if args.cmb_descriptor is None:
            if 'CMB_descriptor' in parameters:
                args.cmb_descriptor = parameters['CMB_descriptor']
            elif 'CMB' in args.prediction or 'all' in args.prediction:
                inParser.error('The model descriptor (json file) for the CMB model has not been specified')
        if args.lac_descriptor is None:
            if 'LAC_descriptor' in parameters:
                args.lac_descriptor = parameters['LAC_descriptor']
            elif 'LAC' in args.prediction or 'all' in args.prediction:
                inParser.error('The model descriptor (json file) for the Lacuna model has not been specified')
        if ('CMB' in args.prediction or 'all' in args.prediction) and args.file_type == 'dicom':
            if args.swi_file_num is None:
                if 'swi_echo' in parameters:
                    args.swi_file_num = parameters['swi_echo']
                else:
                    inParser.error('The model descriptor (json file) for the Lacuna model has not been specified')

    args.model = os.path.abspath(args.model)

    # Check containerizing options
    if (args.containerized_all or args.containerized_nodes) and not args.container_image:
        inParser.error(
            'Using a container (with the "--containerized_all" or "containerized_nodes" arguments) '
            'requires a container image (.sif file) but none was given. Add its path --container_image '
            'or in the configuration file (.yaml file).')
    if args.containerized_nodes and args.brain_seg in ['synthseg', 'synthseg_cpu'] and not args.synthseg_image:
        inParser.error(
            'Using the "containerized_nodes" option with synthseg, but no synthseg apptainer image was provided')

    if args.brain_seg == 'synthseg_precomp':
        args.synthseg_image = args.container_image  # This is just a dummy here to avoid problems

    # Parse the plugin arguments
    if args.run_plugin_args:
        if os.path.isfile(args.run_plugin_args):
            with open(args.run_plugin_args, 'r') as file:
                yaml_content = yaml.safe_load(file)
            args.run_plugin_args = yaml_content
        else:
            try:
                args.run_plugin_args = json.loads(args.run_plugin_args)
            except json.JSONDecodeError:
                raise ValueError('The "--run_plugin_args" argument was not recognised as a file path (file not existing) '
                                 f'nor a json string (bad formatting possibly). Input string: {args.run_plugin_args}')
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
    # setattr(args, 'preproc_results', None)  # TODO: remove when preproc_results updated
    if args.preproc_results is not None:
        args.preproc_results = os.path.abspath(args.preproc_results)
        if not os.path.exists(args.preproc_results):
            raise ValueError(
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
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
    return args
