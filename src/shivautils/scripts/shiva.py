#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.workflows.SWI_postprocessing import genWorkflow as genWorkflowPostSWI
from shivautils.workflows.SWI_predict import genWorkflow as genWorkflowPredictSWI
from shivautils.workflows.SWI_preprocessing import genWorkflow as genWorkflowSWI
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.predict import genWorkflow as genWorkflowPredict
from shivautils.workflows.dual_predict import genWorkflow as genWorkflowDualPredict
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.dual_preprocessing import genWorkflow as genWorkflowDualPreproc
from nipype import config
from nipype.pipeline.engine import Workflow
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

    parser.add_argument('--use_container',
                        action='store_true',
                        help='Wether or not to use containerised processes (mainly for SWOmed).')

    parser.add_argument('--container',
                        action='store_true',
                        help='Wether or not process is launched from inside a container.')

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
                        help='Percentile of the data to keep when doing image normalisation (to remove hotspots)')

    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Treshold to binarise estimated brain mask')

    parser.add_argument('--threshold_clusters',
                        type=float,
                        default=0.2,
                        help='Threshold to compute clusters metrics')

    parser.add_argument('--min_pvs_size',  # TODO: add to the yaml config file
                        type=int,
                        default=7,
                        help='Size (in voxels) below which segmented PVS are discarded')

    parser.add_argument('--min_wmh_size',  # TODO: add to the yaml config file
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented WMH are discarded')

    parser.add_argument('--min_cmb_size',  # TODO: add to the yaml config file
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented CMB are discarded')

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


def set_args_and_check(inParser):
    args = inParser.parse_args()
    if args.container and not args.model_config:
        inParser.error(
            'Using a container (denoted with the "--container" argument) requires '
            'a configuration file (.yml) but none was given.')
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
        args.final_dimensions = tuple(parameters['final_dimensions'])
        args.voxels_size = tuple(parameters['voxels_size'])
        args.interpolation = parameters['interpolation']
        args.brainmask_descriptor = parameters['brainmask_descriptor']
        args.pvs_descriptor = parameters['PVS_descriptor']
        args.pvs2_descriptor = parameters['PVS2_descriptor']
        args.wmh_descriptor = parameters['WMH_descriptor']
        args.cmb_descriptor = parameters['CMB_descriptor']
        if 'min_pvs_size' in parameters.keys():
            args.min_pvs_size = parameters['min_pvs_size']
        if 'min_wmh_size' in parameters.keys():
            args.min_wmh_size = parameters['min_wmh_size']
        if 'min_cmb_size' in parameters.keys():
            args.min_cmb_size = parameters['min_cmb_size']

    if args.container:
        args.model = '/mnt/model'

    if 'all' in args.prediction:
        args.prediction = ['PVS2', 'WMH', 'CMB']
    if not isinstance(args.prediction, list):  # When only one input
        args.prediction = [args.prediction]
    return args


def check_input_for_pred(wfargs):
    # wfargs['PREDICTION'] is a combination of ['PVS', 'PVS2', 'WMH', 'CMB']
    for pred in wfargs['PREDICTION']:
        if not os.path.exists(wfargs[f'{pred}_DESCRIPTOR']):
            errormsg = ('The AI model descriptor for the segmentation of {pred} was not found. '
                        'Check if the model paths were properly setup in the configuration file (.yml).\n'
                        f'The path given for the model descriptor was: {wfargs[f"{pred}_DESCRIPTOR"]}')
            raise FileNotFoundError(errormsg)


def update_wf_grabber(wf, data_struct, dual):
    """Updates the workflow datagrabber to work with the different types on input
    """
    datagrabber = wf.get_node('dataGrabber')

    if data_struct in ['standard', 'json']:
        if dual:
            datagrabber.inputs.field_template = {'t1': '%s/%s/*_raw.nii.gz',
                                                 'flair': '%s/%s/*_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 't1']],
                                                'flair': [['subject_id', 'flair']]}
        else:
            datagrabber.inputs.field_template = {'t1': '%s/%s/*_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 't1']]}

    if data_struct == 'BIDS':
        if dual:
            datagrabber.inputs.field_template = {'t1': '%s/anat/%s_T1_raw.nii.gz',
                                                 'flair': '%s/anat/%s_FLAIR_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']],
                                                'flair': [['subject_id', 'subject_id']]}
        else:
            datagrabber.inputs.field_template = {'t1': '%s/anat/%s_T1_raw.nii.gz'}
            datagrabber.inputs.template_args = {'t1': [['subject_id', 'subject_id']]}
    return wf


def main():

    parser = shivaParser()
    args = set_args_and_check(parser)

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

    if args.prediction == ['PVS']:
        dual = False
    elif 'PVS2' in args.prediction or 'WMH' in args.prediction:
        dual = True

    wfargs = {
        'SUB_WF': True,  # Denotes that the workflows are stringed together
        'SUBJECT_LIST': subject_list,
        'DATA_DIR': subject_directory,  # Default base_directory for the dataGrabber
        'BASE_DIR': out_dir,  # Default base_dir for each workflow
        'PREDICTION': args.prediction,  # TODO: remove?
        'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
        'WMH_DESCRIPTOR': wmh_descriptor,
        'PVS_DESCRIPTOR': pvs_descriptor,
        'PVS2_DESCRIPTOR': pvs2_descriptor,  # TODO: Compatible SMOmed ?
        'CMB_DESCRIPTOR': cmb_descriptor,
        'CONTAINER': not args.use_container,  # store "False" because of legacy meaning of the variable. Only when used by SMOmed usually
        'MODELS_PATH': args.model,
        'GPU': args.gpu,
        'ANONYMIZED': False,  # TODO: Why False though?
        'INTERPOLATION': args.interpolation,
        'PERCENTILE': args.percentile,
        'THRESHOLD': args.threshold,
        'THRESHOLD_CLUSTERS': args.threshold_clusters,
        'MIN_PVS_SIZE': args.min_pvs_size,
        'MIN_WMH_SIZE': args.min_wmh_size,
        'MIN_CMB_SIZE': args.min_cmb_size,
        'IMAGE_SIZE': tuple(args.final_dimensions),
        'RESOLUTION': tuple(args.voxels_size),
        'ORIENTATION': 'RAS'}

    check_input_for_pred(wfargs)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # Declaration of the workflows
    main_wf = Workflow('full_workflow')
    main_wf.base_dir = wfargs['BASE_DIR']
    if dual:
        wf_preproc = genWorkflowDualPreproc(**wfargs)
    else:
        wf_preproc = genWorkflowPreproc(**wfargs)
    wf_preproc = update_wf_grabber(wf_preproc, args.input_type, dual)
    wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    # wf_preproc.write_graph(graph2use='orig', dotfilename='graph.svg', format='svg')
    # wf_preproc.run(plugin='Linear')

    # Prepare prediction workflows
    pred_wfs = {}  # Dict that will contain all prediction sub_workflows
    for PRED in args.prediction:
        biomarker = PRED.lower()
        if biomarker == 'pvs2':
            biomarker = 'pvs'
        if dual:
            wf_pred = genWorkflowDualPredict(**wfargs, PRED=PRED)
        else:
            wf_pred = genWorkflowPredict(**wfargs, PRED=PRED)
        wf_pred.name = f'{biomarker}_predictor_workflow'
        wf_pred.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        pred_wfs[biomarker] = wf_pred

    wf_post = genWorkflowPost(**wfargs)
    main_wf.add_nodes([wf_preproc, wf_post] + pred_wfs)

    main_wf.connect(wf_preproc, 'preproc_out_node.preproc_out_dict', wf_post, 'post_proc_input_node.preproc_dict')
    for biomarker, wf_pred in pred_wfs.items():
        main_wf.connect(wf_preproc, 'preproc_out_node.preproc_out_dict', wf_pred, 'input_parser.in_dict')
        main_wf.connect(wf_pred, 'predict_out_node.predict_out_dict', wf_post, f'post_proc_input_node.{biomarker}_pred_dict')

    # wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    # wf_post.run(plugin='Linear')

    main_wf.write_graph(graph2use='orig', dotfilename='graph.svg', format='svg')
    main_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    main_wf.run(plugin='Linear')

    if 'CMB' in args.prediction:  # TODO: Check if SWI preproc needs T1/dual preproc or is stand-alone
        wfargs.update({'WF_SWI_DIRS': {'preproc': 'shiva_preprocessing_swi', 'pred': 'SWI_predictor_workflow'}})
        swi_wf_preproc = genWorkflowSWI(**wfargs)
        swi_wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        swi_wf_preproc.run(plugin='Linear')

        swi_wf_predict = genWorkflowPredictSWI(**wfargs)
        swi_wf_predict.run(plugin='Linear')

        swi_wf_post = genWorkflowPostSWI(**wfargs)
        swi_wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        swi_wf_post.run(plugin='Linear')


if __name__ == "__main__":
    main()
