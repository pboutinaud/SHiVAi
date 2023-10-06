#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.workflows.SWI_postprocessing import genWorkflow as genWorkflowPostSWI
from shivautils.workflows.SWI_predict import genWorkflow as genWorkflowPredictSWI
from shivautils.workflows.SWI_preprocessing import genWorkflow as genWorkflowSWI
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.predict import genWorkflow as genWorkflowPredict
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from nipype import config
import os
import argparse
import json
import sys
import yaml

sys.path.append('/mnt/devt')

config.enable_provenance()


def shivaParser():
    DESCRIPTION = """SHIVA pipeline for deep-learning imaging biomarkers computation. Performs resampling and coregistration 
                of a set of structural NIfTI head image, followed by intensity normalization, and cropping centered on the brain.
                A nipype workflow is used to preprocess a lot of images at the same time.
                The segmentations from the wmh, cmb and pvs models are generated depending on the inputs. A Report is generated.
                
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

    parser.add_argument('--prediction',
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'all'],
                        nargs='+',
                        help=("Choice of the type of prediction (i.e. segmentation) you want to compute.\n"
                              "A combination of multiple predictions (separated by a white space) can be given.\n"
                              "- 'PVS' for the segmentation of perivascular spaces using only T1 scans\n"
                              "- 'PVS2' for the segmentation of perivascular spaces using both T1 and FLAIR scans\n"
                              "- 'WMH' for the segmentation of white matter hyperintensities (requires both T1 and FLAIR scans)\n"
                              "- 'CMB' for the segmentation of cerebral microbleeds (requires SWI scans)\n"
                              "- 'all' for doing 'PVS2', 'WMH', and 'CMB' segmentation (requires T1, FLAIR, and SWI scans)"),
                        default=['PVS'])

    parser.add_argument('--synthseg',
                        action='store_true',
                        help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

    parser.add_argument('--gpu',
                        type=int,
                        help='Force GPU to use (default is taken from "CUDA_VISIBLE_DEVICES").')

    parser.add_argument('--container',
                        action='store_true',
                        help='Wether or not the script is being called from a container or not.')

    parser.add_argument('--model_config',
                        type=str,
                        help=('Configuration file (.yml) containing the information and parameters for the '
                              'AI model (as well as the path to the AppTainer container when used).\n'
                              'Using a configuration file is incompatible with the arguments listed below '
                              '(i.e. --model --percentile --threshold --threshold_clusters --final_dimensions '
                              '--voxels_size --interpolation --brainmask_descriptor --pvs_descriptor '
                              '--pvs2_descriptor --wmh_descriptor --cmb_descriptor).'),
                        default=None)

    # Manual input
    parser.add_argument('--model',
                        default=None,
                        help='path to model descriptor')

    parser.add_argument('--percentile',
                        type=float,
                        default=99,
                        help='Threshold value expressed as percentile')

    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Value of the treshold to apply to the image')

    parser.add_argument('--threshold_clusters',
                        type=float,
                        default=0.2,
                        help='Threshold to compute clusters metrics')

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
    return parser


def setArgsAndCheck(inParser):
    args = inParser.parse_args()
    if args.container and not args.model_config:
        inParser.error(
            'Using a container (denoted with the "--container" argument) requires '
            'a configuration file (.yml) do none was give.')
    if os.path.isdir(args.output) and bool(os.listdir(args.output)):
        inParser.error(
            'The output directory already exists and is not empty.'
        )

    if args.model_config:  # Parse the config file
        with open(args.model_config, 'r') as file:
            yaml_content = yaml.safe_load(file)
        parameters = yaml_content['parameters']
        args.model = yaml_content['model_path']  # only used when not with container
        args.percentile = parameters['percentile']
        args.threshold = parameters['threshold']
        args.threshold_clusters = parameters['threshold_clusters']
        args.final_dimensions = (int(dm) for dm in parameters['final_dimensions'].split(' '))
        args.voxels_size = (float(vx) for vx in parameters['voxels_size'].split(' '))
        args.interpolation = parameters['interpolation']
        args.brainmask_descriptor = parameters['brainmask_descriptor']
        args.pvs_descriptor = parameters['PVS_descriptor']
        args.pvs2_descriptor = parameters['PVS2_descriptor']
        args.wmh_descriptor = parameters['WMH_descriptor']
        args.cmb_descriptor = parameters['CMB_descriptor']

    if args.container:
        args.model = '/mnt/model'

    if 'all' in args.prediction:
        args.prediction = ['PVS2', 'WMH', 'CMB']
    if not isinstance(args.prediction, list):  # When only one input
        args.prediction = [args.prediction]
    return args


def checkInputForPred(wfargs):
    # wfargs['PREDICTION'] is a combination of ['PVS', 'PVS2', 'WMH', 'CMB']
    for pred in wfargs['PREDICTION']:
        if not os.path.exists(wfargs[f'{pred}_DESCRIPTOR']):
            errormsg = ('The AI model descriptor for the segmentation of {pred} was not found. '
                        'Check if the model paths were properly setup in the configuration file (.yml).\n'
                        f'The path given for the model descriptor was: {wfargs[f"{pred}_DESCRIPTOR"]}')
            raise FileNotFoundError(errormsg)


def main():

    parser = shivaParser()
    args = setArgsAndCheck(parser)

    # GRAB_PATTERN = '%s/%s/*.nii*'  # Now directly specified in the workflow generators
    # synthseg = args.synthseg  # Unused for now

    if args.input_type == 'json':  # TODO: Homogenize with the .yml file
        with open(args.input, 'r') as json_in:
            subject_dict = json.load(json_in)

        out_dir = subject_dict['parameters']['out_dir']
        subject_directory = subject_dict["files_dir"]
        subject_list = os.listdir(subject_directory)
        brainmask_descriptor = subject_dict['parameters']['brainmask_descriptor']
        if subject_dict['parameters']['WMH_descriptor']:
            wmh_descriptor = subject_dict['parameters']['WMH_descriptor']
        else:
            wmh_descriptor = None
        if subject_dict['parameters']['PVS_descriptor']:
            pvs_descriptor = subject_dict['parameters']['PVS_descriptor']
        else:
            pvs_descriptor = None
        if subject_dict['parameters']['CMB_descriptor']:
            cmb_descriptor = subject_dict['parameters']['CMB_descriptor']
        else:
            cmb_descriptor = None

    if args.input_type == 'standard' or args.input_type == 'BIDS':
        subject_directory = args.input
        out_dir = args.output
        subject_list = os.listdir(subject_directory)
        brainmask_descriptor = os.path.join(args.model, args.brainmask_descriptor)
        wmh_descriptor = os.path.join(args.model, args.wmh_descriptor)
        pvs_descriptor = os.path.join(args.model, args.pvs_descriptor)
        pvs2_descriptor = os.path.join(args.model, args.pvs2_descriptor)
        cmb_descriptor = os.path.join(args.model, args.cmb_descriptor)

    wfargs = {'SUBJECT_LIST': subject_list,
              'DATA_DIR': subject_directory,  # Default base_directory for the dataGrabber
              'BASE_DIR': out_dir,  # Default base_dir for each workflow
              'INPUT_TYPE': args.input_type,
              'PREDICTION': args.prediction,
              'WF_DIRS': {'preproc': 'shiva_preprocessing', 'pred': 'predictor_workflow'},  # Name of the preprocessing wf, used to link wf for now
              'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
              'WMH_DESCRIPTOR': wmh_descriptor,
              'PVS_DESCRIPTOR': pvs_descriptor,
              'PVS2_DESCRIPTOR': pvs2_descriptor,
              'CMB_DESCRIPTOR': cmb_descriptor,
              'CONTAINER': args.container,  # TODO: Adapt other scripts to run without Apptainer
              'MODELS_PATH': args.model,
              'GPU': args.gpu,
              'ANONYMIZED': False,  # TODO: Why False though?
              'INTERPOLATION': args.interpolation,
              'PERCENTILE': args.percentile,
              'THRESHOLD': args.threshold,
              'THRESHOLD_CLUSTERS': args.threshold_clusters,
              'IMAGE_SIZE': tuple(args.final_dimensions),
              'RESOLUTION': tuple(args.voxels_size),
              'ORIENTATION': 'RAS'}

    checkInputForPred(wfargs)

    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # TODO: merge the workflows and do the ckecks inside
    wf_preproc = genWorkflowPreproc(**wfargs)
    wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    # If necessary to modify defaults:
    # wf_preproc.get_node('conform').inputs.dimensions = (256, 256, 256)
    # wf_preproc.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
    # wf_preproc.get_node('conform').inputs.orientation = 'RAS'
    # wf_preproc.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)
    wf_preproc.run(plugin='Linear')

    wf_predict = genWorkflowPredict(**wfargs)
    wf_predict.run(plugin='Linear')

    wf_post = genWorkflowPost(**wfargs)
    # wf_post.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf_predict.name)
    wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}  # TODO: Is there even the possibility that it is True?
    wf_post.run(plugin='Linear')

    if 'CMB' in args.prediction:  # TODO: Check if SWI preproc needs T1/dual preproc or is stand-alone
        wfargs.update({'WF_SWI_DIRS': {'preproc': 'shiva_preprocessing_swi', 'pred': 'SWI_predictor_workflow'}})
        swi_wf_preproc = genWorkflowSWI(**wfargs)
        swi_wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        # If necessary to modify defaults:
        # swi_wf_preproc.get_node('conform').inputs.dimensions = (256, 256, 256)
        # swi_wf_preproc.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
        # swi_wf_preproc.get_node('conform').inputs.orientation = 'RAS'
        # swi_wf_preproc.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)
        swi_wf_preproc.run(plugin='Linear')

        swi_wf_predict = genWorkflowPredictSWI(**wfargs)
        swi_wf_predict.run(plugin='Linear')

        swi_wf_post = genWorkflowPostSWI(**wfargs)
        swi_wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        swi_wf_post.run(plugin='Linear')




if __name__ == "__main__":
    main()
