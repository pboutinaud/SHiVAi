#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.interfaces.shiva import Predict
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.post_processing import set_wf_shapers
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.dual_preprocessing import genWorkflow as genWorkflowDualPreproc
from shivautils.workflows.preprocessing_swi_reg import gen_workflow_swi
from shivautils.interfaces.image import Join_Prediction_metrics
from nipype import config
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
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

    parser.add_argument('--retry',
                        action='store_true',
                        help='Relaunch the pipeline from where it stopped')

    parser.add_argument('--anonymize',
                        action='store_true',
                        help='Anonymize the report')

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

    parser.add_argument('--min_pvs_size',
                        type=int,
                        default=5,
                        help='Size (in voxels) below which segmented PVS are discarded')

    parser.add_argument('--min_wmh_size',
                        type=int,
                        default=1,
                        help='Size (in voxels) below which segmented WMH are discarded')

    parser.add_argument('--min_cmb_size',
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
    if (os.path.isdir(args.output)
        and bool(os.listdir(args.output))
            and not args.retry):
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
        if 'min_pvs_size' in parameters.keys():
            args.min_pvs_size = parameters['min_pvs_size']
        if 'min_wmh_size' in parameters.keys():
            args.min_wmh_size = parameters['min_wmh_size']
        if 'min_cmb_size' in parameters.keys():
            args.min_cmb_size = parameters['min_cmb_size']
        args.final_dimensions = tuple(parameters['final_dimensions'])
        args.voxels_size = tuple(parameters['voxels_size'])
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


def check_input_for_pred(wfargs):
    """
    Checks if the AI model is available for the predictions that are supposed to run
    """
    # wfargs['PREDICTION'] is a combination of ['PVS', 'PVS2', 'WMH', 'CMB']
    for pred in wfargs['PREDICTION']:
        if not os.path.exists(wfargs[f'{pred}_DESCRIPTOR']):
            errormsg = ('The AI model descriptor for the segmentation of {pred} was not found. '
                        'Check if the model paths were properly setup in the configuration file (.yml).\n'
                        f'The path given for the model descriptor was: {wfargs[f"{pred}_DESCRIPTOR"]}')
            raise FileNotFoundError(errormsg)


def update_wf_grabber(wf, data_struct, acquisitions):
    """Updates the workflow datagrabber to work with the different types on input
    acquisitions example: [('img1', 't1'), ('img2', 'flair')]
    """
    datagrabber = wf.get_node('datagrabber')
    if data_struct in ['standard', 'json']:
        # e.g: {'img1': '%s/t1/%s_T1_raw.nii.gz'}
        datagrabber.inputs.field_template = {acq[0]: f'%s/{acq[1]}/*_raw.nii*' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id']] for acq in acquisitions}

    if data_struct == 'BIDS':
        # e.g: {'img1': '%s/anat/%s_T1_raw.nii.gz}
        datagrabber.inputs.field_template = {acq[0]: f'%s/anat/%s_{acq[1].upper()}_raw.nii*' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id', 'subject_id']] for acq in acquisitions}
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

    wfargs = {
        'SUB_WF': True,  # Denotes that the workflows are stringed together
        'SUBJECT_LIST': subject_list,
        'DATA_DIR': subject_directory,  # Default base_directory for the datagrabber
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
        'ANONYMIZED': args.anonymize,  # TODO: Why False though?
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

    # Check if the AI models aire available for the predictions
    check_input_for_pred(wfargs)

    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(wfargs['PREDICTION'])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('full_workflow')
    main_wf.base_dir = wfargs['BASE_DIR']

    # Start by initialisin the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', wfargs['SUBJECT_LIST'])

    # First, preproc and postproc
    if with_t1:
        acquisitions = [('img1', 't1')]
        if with_flair:
            acquisitions.append(('img2', 'flair'))
            wf_preproc = genWorkflowDualPreproc(**wfargs, wf_name='shiva_dual_preprocessing')
        else:
            wf_preproc = genWorkflowPreproc(**wfargs, wf_name='shiva_t1_preprocessing')
        if with_swi:  # Adding another preproc wf for swi, using t1 mask
            acquisitions.append(('img3', 'swi'))
            wf_preproc_cmb = gen_workflow_swi(**wfargs, wf_name='shiva_swi_preprocessing')
        wf_preproc = update_wf_grabber(wf_preproc, args.input_type, acquisitions)
    elif with_swi and not with_t1:  # CMB alone
        acquisitions = [('img1', 'swi')]
        wf_preproc = genWorkflowPreproc(**wfargs, wf_name='shiva_swi_preprocessing')
        wf_preproc = update_wf_grabber(wf_preproc, args.input_type, acquisitions)

    wf_post = genWorkflowPost(**wfargs)
    main_wf.add_nodes([wf_preproc, wf_post])

    # All connections between preproc and postproc
    main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'datagrabber.subject_id')
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(wf_preproc, 'conform.resampled', wf_post, 'qc_crop_box.img_apply_to')
    main_wf.connect(wf_preproc, 'hard_brain_mask.thresholded', wf_post, 'qc_crop_box.brainmask')
    main_wf.connect(wf_preproc, 'crop.bbox1', wf_post, 'qc_crop_box.bbox1')
    main_wf.connect(wf_preproc, 'crop.bbox2', wf_post, 'qc_crop_box.bbox2')
    main_wf.connect(wf_preproc, 'crop.cdg_ijk', wf_post, 'qc_crop_box.cdg_ijk')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_post, 'qc_overlay_brainmask.brainmask')
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', wf_post, 'qc_overlay_brainmask.img_ref')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_post, 'summary_report.brainmask')

    if with_flair:
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', wf_post, 'qc_coreg_FLAIR_T1.path_image')
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', wf_post, 'qc_coreg_FLAIR_T1.path_ref_image')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_post, 'qc_coreg_FLAIR_T1.path_brainmask')

    if with_t1 and with_swi:
        main_wf.add_nodes([wf_preproc_cmb])
        main_wf.connect(wf_preproc, 'datagrabber.img3', wf_preproc_cmb, 'conform.img')
        main_wf.connect(wf_preproc, 'crop.cropped', wf_preproc_cmb, 'swi_to_t1.fixed_image')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_preproc_cmb, 'mask_to_swi.input_image')
        main_wf.connect(wf_preproc_cmb, 'mask_to_swi.output_image', wf_post, 'qc_overlay_brainmask_swi.brainmask')
        main_wf.connect(wf_preproc_cmb, 'swi_intensity_normalisation.intensity_normalized', wf_post, 'qc_overlay_brainmask_swi.img_ref')

    # Then prediction nodes and their connections
    # PVS
    if 'PVS' in args.prediction or 'PVS2' in args.prediction:
        pvs_predictor_node = Node(Predict(), name=f"predict_pvs")
        pvs_predictor_node.inputs.out_filename = 'pvs_map.nii.gz'
        pvs_predictor_node.inputs.model = wfargs['MODELS_PATH']
        if with_flair:
            pvs_predictor_node.inputs.descriptor = wfargs['PVS2_DESCRIPTOR']
            main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', pvs_predictor_node, "flair")
        else:
            pvs_predictor_node.inputs.descriptor = wfargs['PVS_DESCRIPTOR']
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', pvs_predictor_node, "t1")
        main_wf.connect(pvs_predictor_node, 'segmentation', wf_post, 'prediction_metrics_pvs.img')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_pvs.brain_seg')  # TODO: SynthSeg
        # Merge all csv files
        prediction_metrics_pvs_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_pvs_all")
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', prediction_metrics_pvs_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_pvs_all, 'subject_id')

    # WMH
    if 'WMH' in args.prediction:
        wmh_predictor_node = Node(Predict(), name=f"predict_wmh")
        wmh_predictor_node.inputs.out_filename = 'wmh_map.nii.gz'
        wmh_predictor_node.inputs.model = wfargs['MODELS_PATH']
        wmh_predictor_node.inputs.descriptor = wfargs['WMH_DESCRIPTOR']
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', wmh_predictor_node, "t1")
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', wmh_predictor_node, "flair")
        main_wf.connect(wmh_predictor_node, 'segmentation', wf_post, 'prediction_metrics_wmh.img')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_wmh.brain_seg')  # TODO: SynthSeg
        # Merge all csv files
        prediction_metrics_wmh_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_wmh_all")
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', prediction_metrics_wmh_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_wmh_all, 'subject_id')

    # CMB
    if 'CMB' in args.prediction:
        cmb_predictor_node = Node(Predict(), name=f"predict_cmb")
        cmb_predictor_node.inputs.out_filename = 'cmb_map.nii.gz'
        cmb_predictor_node.inputs.model = wfargs['MODELS_PATH']
        cmb_predictor_node.inputs.descriptor = wfargs['CMB_DESCRIPTOR']
        # Merge all csv files
        prediction_metrics_cmb_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_cmb_all")
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_stats_csv', prediction_metrics_cmb_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_cmb_all, 'subject_id')
        if with_t1:
            main_wf.connect(wf_preproc_cmb, 'swi_intensity_normalisation.intensity_normalized', cmb_predictor_node, "swi")
            main_wf.connect(cmb_predictor_node, 'segmentation', wf_post, 'swi_pred_to_t1.input_image')
            main_wf.connect(wf_preproc_cmb, 'swi_to_t1.forward_transforms', wf_post, 'swi_pred_to_t1.transforms')
            main_wf.connect(wf_preproc, 'crop.cropped', wf_post, 'swi_pred_to_t1.reference_image')
        else:
            main_wf.connect(cmb_predictor_node, 'segmentation', wf_post, 'prediction_metrics_cmb.img')
            main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', cmb_predictor_node, "swi")
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_cmb.brain_seg')  # TODO: SynthSeg

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')
    wf_post.get_node('summary_report').inputs.wf_graph = os.path.abspath(wf_graph)

    # Finally the data sinks
    # Initialising the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(wfargs['BASE_DIR'], 'results')
    main_wf.connect(subject_iterator, 'subject_id', sink_node_subjects, 'container')
    main_wf.connect(wf_post, 'summary_report.summary', sink_node_subjects, 'report')

    sink_node_all = Node(DataSink(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(wfargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the sinks
    if with_t1:
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, 't1_preproc')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', sink_node_subjects, 't1_preproc.brain_mask')
        main_wf.connect(wf_preproc, 'mask_to_img1.resampled_image', sink_node_subjects, 't1_preproc.brain_mask_raw_space')
        main_wf.connect(wf_preproc, 'crop.bbox1_file', sink_node_subjects, 't1_preproc.@bb1')
        main_wf.connect(wf_preproc, 'crop.bbox2_file', sink_node_subjects, 't1_preproc.@bb2')
        main_wf.connect(wf_preproc, 'crop.cdg_ijk_file', sink_node_subjects, 't1_preproc.@cdg')
        if with_flair:
            main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', sink_node_subjects, 'flair_preproc')
        if with_swi:
            main_wf.connect(wf_preproc_cmb, 'swi_intensity_normalisation.intensity_normalized', sink_node_subjects, 'swi_preproc')
            main_wf.connect(wf_preproc_cmb, 'mask_to_swi.output_image', sink_node_subjects, 'swi_preproc.brain_mask')
            main_wf.connect(wf_preproc_cmb, 'swi_to_t1.warped_image', sink_node_subjects, 'swi_preproc.reg_to_t1')
            main_wf.connect(wf_preproc_cmb, 'crop_swi.bbox1_file', sink_node_subjects, 'swi_preproc.@bb1')
            main_wf.connect(wf_preproc_cmb, 'crop_swi.bbox2_file', sink_node_subjects, 'swi_preproc.@bb2')
            main_wf.connect(wf_preproc_cmb, 'crop_swi.cdg_ijk_file', sink_node_subjects, 'swi_preproc.@cdg')
    elif with_swi and not with_t1:
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, 'swi_preproc')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', sink_node_subjects, 'swi_preproc.brain_mask')
        main_wf.connect(wf_preproc, 'mask_to_img1.resampled_image', sink_node_subjects, 'swi_preproc.brain_mask_raw_space')
        main_wf.connect(wf_preproc, 'crop_swi.bbox1_file', sink_node_subjects, 'swi_preproc.@bb1')
        main_wf.connect(wf_preproc, 'crop_swi.bbox2_file', sink_node_subjects, 'swi_preproc.@bb2')
        main_wf.connect(wf_preproc, 'crop_swi.cdg_ijk_file', sink_node_subjects, 'swi_preproc.@cdg')

    if 'PVS' in args.prediction or 'PVS2' in args.prediction:
        main_wf.connect(pvs_predictor_node, 'segmentation', sink_node_subjects, 'pvs_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', sink_node_subjects, 'pvs_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_census_csv', sink_node_subjects, 'pvs_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.labelled_biomarkers', sink_node_subjects, 'pvs_segmentation.@labeled')
        main_wf.connect(prediction_metrics_pvs_all, 'metrics_predictions_csv', sink_node_all, 'pvs_metrics')

    if 'WMH' in args.prediction:
        main_wf.connect(wmh_predictor_node, 'segmentation', sink_node_subjects, 'wmh_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', sink_node_subjects, 'wmh_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_census_csv', sink_node_subjects, 'wmh_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.labelled_biomarkers', sink_node_subjects, 'wmh_segmentation.@labeled')
        main_wf.connect(prediction_metrics_wmh_all, 'metrics_predictions_csv', sink_node_all, 'wmh_metrics')

    if 'CMB' in args.prediction:
        if with_t1:
            space = 't1_space'
            main_wf.connect(cmb_predictor_node, 'segmentation', sink_node_subjects, 'cmb_segmentation_swi_space')
            main_wf.connect(wf_post, 'swi_pred_to_t1.output_image', sink_node_subjects, f'cmb_segmentation_{space}')
        else:
            space = 'swi_space'
            main_wf.connect(cmb_predictor_node, 'segmentation', sink_node_subjects, f'cmb_segmentation_{space}')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_stats_csv', sink_node_subjects, f'cmb_segmentation_{space}.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_census_csv', sink_node_subjects, f'cmb_segmentation_{space}.@census')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.labelled_biomarkers', sink_node_subjects, f'cmb_segmentation_{space}.@labeled')
        main_wf.connect(prediction_metrics_cmb_all, 'metrics_predictions_csv', sink_node_all, f'cmb_metrics_{space}')

    sink_node_all.inputs.wf_graph = wf_graph

    # Run the workflow
    main_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    main_wf.run(plugin='Linear')


if __name__ == "__main__":
    main()
