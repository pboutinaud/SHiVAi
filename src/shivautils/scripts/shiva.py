#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.interfaces.shiva import Predict, PredictSingularity
from shivautils.utils.misc import set_wf_shapers  # , as_list
from shivautils.utils.parsing import shivaParser, set_args_and_check
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.dual_preprocessing import genWorkflow as genWorkflowDualPreproc
from shivautils.workflows.preprocessing_swi_reg import graft_workflow_swi
from shivautils.workflows.preprocessing_premasked import genWorkflow as genWorkflow_preproc_masked
from shivautils.interfaces.post import Join_Prediction_metrics, Join_QC_metrics
from nipype import config
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
import os
import json

# sys.path.append('/mnt/devt')


def check_input_for_pred(wfargs):
    """
    Checks if the AI model is available for the predictions that are supposed to run
    """
    # wfargs['PREDICTION'] is a combination of ['PVS', 'PVS2', 'WMH', 'CMB', 'LAC']
    for pred in wfargs['PREDICTION']:
        if not os.path.exists(wfargs[f'{pred}_DESCRIPTOR']):
            errormsg = (f'The AI model descriptor for the segmentation of {pred} was not found. '
                        'Check if the model paths were properly setup in the configuration file (.yml).\n'
                        f'The path given for the model descriptor was: {wfargs[f"{pred}_DESCRIPTOR"]}')
            raise FileNotFoundError(errormsg)


def update_wf_grabber(wf, data_struct, acquisitions, seg=None):
    """
    Updates the workflow datagrabber to work with the different types on input
        wf: workflow with the datagrabber
        data_struct ('standard', 'BIDS', 'json')
        acquisitions example: [('img1', 't1'), ('img2', 'flair')]
        seg ('masked', 'synthseg'): type of brain segmentation available (if any)
    """
    datagrabber = wf.get_node('datagrabber')
    if data_struct in ['standard', 'json']:
        # e.g: {'img1': '%s/t1/%s_T1_raw.nii.gz'}
        datagrabber.inputs.field_template = {acq[0]: f'%s/{acq[1]}/*.nii*' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id']] for acq in acquisitions}

    if data_struct == 'BIDS':
        # e.g: {'img1': '%s/anat/%s_T1_raw.nii.gz}
        datagrabber.inputs.field_template = {acq[0]: f'%s/anat/%s_{acq[1].upper()}*.nii*' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id', 'subject_id']] for acq in acquisitions}

    if seg == 'masked':
        datagrabber.inputs.field_template['brainmask'] = datagrabber.inputs.field_template['img1']
        datagrabber.inputs.template_args['brainmask'] = datagrabber.inputs.template_args['img1']
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
        # subject_list = os.listdir(subject_directory)
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
        if subject_dict['parameters']['LAC_descriptor']:
            lac_descriptor = subject_dict['parameters']['LAC_descriptor']
        else:
            lac_descriptor = None

    if args.input_type == 'standard' or args.input_type == 'BIDS':
        subject_directory = args.input
        out_dir = args.output
        brainmask_descriptor = os.path.join(args.model, args.brainmask_descriptor)
        wmh_descriptor = os.path.join(args.model, args.wmh_descriptor)
        pvs_descriptor = os.path.join(args.model, args.pvs_descriptor)
        pvs2_descriptor = os.path.join(args.model, args.pvs2_descriptor)
        cmb_descriptor = os.path.join(args.model, args.cmb_descriptor)
        lac_descriptor = os.path.join(args.model, args.lac_descriptor)

    if args.synthseg:
        raise NotImplemented('Sorry, using MRI Synthseg results is not implemented yet')
        # seg = 'synthseg'
    elif args.masked:
        seg = 'masked'
    else:
        seg = None

    # Plugin arguments for predictions (shiva pred and synthseg)
    pred_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    reg_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    if 'pred' in args.node_plugin_args.keys():
        pred_plugin_args = args.node_plugin_args['pred']
    if 'reg' in args.node_plugin_args.keys():
        reg_plugin_args = args.node_plugin_args['reg']

    wfargs = {
        'SUB_WF': True,  # Denotes that the workflows are stringed together
        'SUBJECT_LIST': args.sub_list,
        'DATA_DIR': subject_directory,  # Default base_directory for the datagrabber
        'BASE_DIR': out_dir,  # Default base_dir for each workflow
        'PREDICTION': args.prediction,  # Needed by the postproc for now
        'BRAIN_SEG': seg,
        'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
        'WMH_DESCRIPTOR': wmh_descriptor,
        'PVS_DESCRIPTOR': pvs_descriptor,
        'PVS2_DESCRIPTOR': pvs2_descriptor,
        'CMB_DESCRIPTOR': cmb_descriptor,
        'LAC_DESCRIPTOR': lac_descriptor,
        'CONTAINER_IMAGE': args.container_image,
        'SYNTHSEG_IMAGE': args.synthseg_image,
        'CONTAINERIZE_NODES': args.containerized_nodes,
        # 'CONTAINER': True #  legacy variable. Only when used by SMOmed usually
        'MODELS_PATH': args.model,
        'GPU': args.gpu,
        'MASK_ON_GPU': args.mask_on_gpu,
        'REG_PLUGIN_ARGS': reg_plugin_args,
        'PRED_PLUGIN_ARGS': pred_plugin_args,
        'ANONYMIZED': args.anonymize,  # TODO: Improve + defacing
        'INTERPOLATION': args.interpolation,
        'PERCENTILE': args.percentile,
        'THRESHOLD': args.threshold,
        'THRESHOLD_CLUSTERS': args.threshold_clusters,
        'MIN_PVS_SIZE': args.min_pvs_size,
        'MIN_WMH_SIZE': args.min_wmh_size,
        'MIN_CMB_SIZE': args.min_cmb_size,
        'MIN_LAC_SIZE': args.min_lac_size,
        'IMAGE_SIZE': tuple(args.final_dimensions),
        'RESOLUTION': tuple(args.voxels_size),
        'ORIENTATION': 'RAS'}

    # Check if the AI models are available for the predictions
    check_input_for_pred(wfargs)

    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(wfargs['PREDICTION'])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('main_workflow')
    main_wf.base_dir = wfargs['BASE_DIR']

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', wfargs['SUBJECT_LIST'])

    # First, initialise the proper preproc and update its datagrabber

    acquisitions = []
    if with_t1:
        acquisitions.append(('img1', 't1'))
        if with_flair:
            acquisitions.append(('img2', 'flair'))
            wf_preproc = genWorkflowDualPreproc(**wfargs, wf_name='shiva_dual_preprocessing')
        else:
            wf_name = 'shiva_t1_preprocessing'
            if wfargs['BRAIN_SEG'] is not None:
                wf_preproc = genWorkflow_preproc_masked(**wfargs, wf_name=wf_name)
            else:
                wf_preproc = genWorkflowPreproc(**wfargs, wf_name=wf_name)
        if with_swi:  # Adding the swi preprocessing steps to the preproc workflow
            acquisitions.append(('img3', 'swi'))
            cmb_preproc_wf_name = 'swi_preprocessing'
            wf_preproc = graft_workflow_swi(wf_preproc, **wfargs, wf_name=cmb_preproc_wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, args.input_type, acquisitions, seg)
    elif with_swi and not with_t1:  # CMB alone
        acquisitions.append(('img1', 'swi'))
        wf_name = 'shiva_swi_preprocessing'
        if wfargs['BRAIN_SEG'] is not None:
            wf_preproc = genWorkflow_preproc_masked(**wfargs, wf_name=wf_name)
        else:
            wf_preproc = genWorkflowPreproc(**wfargs, wf_name=wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, args.input_type, acquisitions, seg)

    # Then initialise the post proc and add the nodes to the main wf
    wf_post = genWorkflowPost(**wfargs)
    main_wf.add_nodes([wf_preproc, wf_post])

    # Set all the connections between preproc and postproc
    main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'datagrabber.subject_id')
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_crop_box.crop_brain_img', wf_post, 'summary_report.crop_brain_img')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask.overlayed_brainmask', wf_post, 'summary_report.overlayed_brainmask_1')
    if with_swi and with_t1:
        main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask_swi.overlayed_brainmask', wf_post, 'summary_report.overlayed_brainmask_2')
    if with_flair:
        main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_coreg_FLAIR_T1.qc_coreg', wf_post, 'summary_report.isocontour_slides_FLAIR_T1')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_post, 'summary_report.brainmask')

    # Joining the individual QC metrics
    qc_joiner = JoinNode(Join_QC_metrics(),
                         joinsource=subject_iterator,
                         joinfield=['csv_files', 'subject_id'],
                         name='qc_joiner')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_metrics.csv_qc_metrics', qc_joiner, 'csv_files')
    main_wf.connect(subject_iterator, 'subject_id', qc_joiner, 'subject_id')
    if args.prev_qc is not None:
        qc_joiner.inputs.population_csv_file = args.prev_qc

    # Then prediction nodes and their connections
    segmentation_wf = Workflow('Segmentation')  # facultative workflow for organization purpose
    # PVS
    if 'PVS' in args.prediction or 'PVS2' in args.prediction:
        if wfargs['CONTAINERIZE_NODES']:
            predict_pvs = Node(PredictSingularity(), name="predict_pvs")
            predict_pvs.inputs.snglrt_bind = [
                (wfargs['BASE_DIR'], wfargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (wfargs['MODELS_PATH'], wfargs['MODELS_PATH'], 'ro')]
            predict_pvs.inputs.out_filename = '/mnt/data/pvs_map.nii.gz'
            predict_pvs.inputs.snglrt_enable_nvidia = True
            predict_pvs.inputs.snglrt_image = wfargs['CONTAINER_IMAGE']
        else:
            predict_pvs = Node(Predict(), name="predict_pvs")
            predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'
        predict_pvs.inputs.model = wfargs['MODELS_PATH']
        predict_pvs.plugin_args = wfargs['PRED_PLUGIN_ARGS']
        if 'PVS2' in args.prediction:
            predict_pvs.inputs.descriptor = wfargs['PVS2_DESCRIPTOR']
        else:
            predict_pvs.inputs.descriptor = wfargs['PVS_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_pvs])
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_pvs.t1')
        if 'PVS2' in args.prediction:
            main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_pvs.flair')
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation', wf_post, 'prediction_metrics_pvs.img')
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
        if wfargs['CONTAINERIZE_NODES']:
            predict_wmh = Node(PredictSingularity(), name="predict_wmh")
            predict_wmh.inputs.snglrt_bind = [
                (wfargs['BASE_DIR'], wfargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (wfargs['MODELS_PATH'], wfargs['MODELS_PATH'], 'ro')]
            predict_wmh.inputs.out_filename = '/mnt/data/wmh_map.nii.gz'
            predict_wmh.inputs.snglrt_enable_nvidia = True
            predict_wmh.inputs.snglrt_image = wfargs['CONTAINER_IMAGE']
        else:
            predict_wmh = Node(Predict(), name="predict_wmh")
            predict_wmh.inputs.out_filename = 'wmh_map.nii.gz'
        predict_wmh.inputs.model = wfargs['MODELS_PATH']
        predict_wmh.plugin_args = wfargs['PRED_PLUGIN_ARGS']
        predict_wmh.inputs.descriptor = wfargs['WMH_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_wmh])

        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_wmh.t1')
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_wmh.flair')
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', wf_post, 'prediction_metrics_wmh.img')
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
        if wfargs['CONTAINERIZE_NODES']:
            predict_cmb = Node(PredictSingularity(), name="predict_cmb")
            predict_cmb.inputs.snglrt_bind = [
                (wfargs['BASE_DIR'], wfargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (wfargs['MODELS_PATH'], wfargs['MODELS_PATH'], 'ro')]
            predict_cmb.inputs.snglrt_enable_nvidia = True
            predict_cmb.inputs.snglrt_image = wfargs['CONTAINER_IMAGE']
            predict_cmb.inputs.out_filename = '/mnt/data/cmb_map.nii.gz'

        else:
            predict_cmb = Node(Predict(), name="predict_cmb")
            predict_cmb.inputs.out_filename = 'cmb_map.nii.gz'
        predict_cmb.inputs.model = wfargs['MODELS_PATH']
        predict_cmb.plugin_args = wfargs['PRED_PLUGIN_ARGS']
        predict_cmb.inputs.descriptor = wfargs['CMB_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_cmb])

        # Merge all csv files
        prediction_metrics_cmb_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_cmb_all")
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_stats_csv', prediction_metrics_cmb_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_cmb_all, 'subject_id')
        if with_t1:
            main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.swi_intensity_normalisation.intensity_normalized', segmentation_wf, 'predict_cmb.swi')
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', wf_post, 'swi_pred_to_t1.input_image')
            main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.swi_to_t1.forward_transforms', wf_post, 'swi_pred_to_t1.transforms')
            main_wf.connect(wf_preproc, 'crop.cropped', wf_post, 'swi_pred_to_t1.reference_image')
        else:
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', wf_post, 'prediction_metrics_cmb.img')
            main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_cmb.swi')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_cmb.brain_seg')  # TODO: SynthSeg

    # Lacuna
    if 'LAC' in args.prediction:
        if wfargs['CONTAINERIZE_NODES']:
            predict_lac = Node(PredictSingularity(), name="predict_lac")
            predict_lac.inputs.snglrt_bind = [
                (wfargs['BASE_DIR'], wfargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (wfargs['MODELS_PATH'], wfargs['MODELS_PATH'], 'ro')]
            predict_lac.inputs.out_filename = '/mnt/data/lac_map.nii.gz'
            predict_lac.inputs.snglrt_enable_nvidia = True
            predict_lac.inputs.snglrt_image = wfargs['CONTAINER_IMAGE']
        else:
            predict_lac = Node(Predict(), name="predict_lac")
            predict_lac.inputs.out_filename = 'lac_map.nii.gz'
        predict_lac.inputs.model = wfargs['MODELS_PATH']
        predict_lac.plugin_args = wfargs['PRED_PLUGIN_ARGS']
        predict_lac.inputs.descriptor = wfargs['LAC_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_lac])

        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_lac.t1')
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_lac.flair')
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', wf_post, 'prediction_metrics_lac.img')
        main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_lac.brain_seg')  # TODO: SynthSeg
        # Merge all csv files
        prediction_metrics_lac_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_lac_all")
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', prediction_metrics_lac_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_lac_all, 'subject_id')

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')
    # wf_post.get_node('summary_report').inputs.wf_graph = os.path.abspath(wf_graph)

    # Finally the data sinks
    # Initializing the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(wfargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('flair_to_t1__Warped_img_normalized', 'flair_to_t1_cropped_intensity_normed')
    ]
    # main_wf.connect(subject_iterator, 'subject_id', sink_node_subjects, 'container')
    main_wf.connect(wf_post, 'summary_report.summary', sink_node_subjects, 'report')

    sink_node_all = Node(DataSink(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(wfargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the sinks
    # Preproc
    if with_t1:
        img1 = 't1'
    elif with_swi and not with_t1:
        img1 = 'swi'
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, f'shiva_preproc.{img1}_preproc')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask')
    if wfargs['BRAIN_SEG'] is None:
        main_wf.connect(wf_preproc, 'mask_to_img1.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask_raw_space')
    main_wf.connect(wf_preproc, 'crop.bbox1_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb1')
    main_wf.connect(wf_preproc, 'crop.bbox2_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb2')
    main_wf.connect(wf_preproc, 'crop.cdg_ijk_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@cdg')
    if with_flair:
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', sink_node_subjects, 'shiva_preproc.flair_preproc')
    if with_swi and with_t1:
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.swi_intensity_normalisation.intensity_normalized', sink_node_subjects, 'shiva_preproc.swi_preproc')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.mask_to_swi.output_image', sink_node_subjects, 'shiva_preproc.swi_preproc.@brain_mask')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.swi_to_t1.warped_image', sink_node_subjects, 'shiva_preproc.swi_preproc.@reg_to_t1')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.swi_to_t1.forward_transforms', sink_node_subjects, 'shiva_preproc.swi_preproc.@reg_to_t1_transf')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.crop_swi.bbox1_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@bb1')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.crop_swi.bbox2_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@bb2')
        main_wf.connect(wf_preproc, f'{cmb_preproc_wf_name}.crop_swi.cdg_ijk_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@cdg')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_metrics.csv_qc_metrics', sink_node_subjects, 'shiva_preproc.qc_metrics')

    # Pred and postproc
    if 'PVS' in args.prediction or 'PVS2' in args.prediction:
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation', sink_node_subjects, 'segmentations.pvs_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_census_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.labelled_biomarkers', sink_node_subjects, 'segmentations.pvs_segmentation.@labeled')
        main_wf.connect(prediction_metrics_pvs_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.pvs_metrics')

    if 'WMH' in args.prediction:
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', sink_node_subjects, 'segmentations.wmh_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_census_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.labelled_biomarkers', sink_node_subjects, 'segmentations.wmh_segmentation.@labeled')
        main_wf.connect(prediction_metrics_wmh_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.wmh_metrics')

    if 'CMB' in args.prediction:
        if with_t1:
            space = 't1-space'
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', sink_node_subjects, 'segmentations.cmb_segmentation_swi-space')
            main_wf.connect(wf_post, 'swi_pred_to_t1.output_image', sink_node_subjects, f'segmentations.cmb_segmentation_{space}')
        else:
            space = 'swi-space'
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', sink_node_subjects, f'segmentations.cmb_segmentation_{space}')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_stats_csv', sink_node_subjects, f'segmentations.cmb_segmentation_{space}.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_census_csv', sink_node_subjects, f'segmentations.cmb_segmentation_{space}.@census')
        main_wf.connect(wf_post, 'prediction_metrics_cmb.labelled_biomarkers', sink_node_subjects, f'segmentations.cmb_segmentation_{space}.@labeled')
        main_wf.connect(prediction_metrics_cmb_all, 'prediction_metrics_csv', sink_node_all, f'segmentations.cmb_metrics_{space}')

    if 'LAC' in args.prediction:
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', sink_node_subjects, 'segmentations.lac_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', sink_node_subjects, 'segmentations.lac_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_census_csv', sink_node_subjects, 'segmentations.lac_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_lac.labelled_biomarkers', sink_node_subjects, 'segmentations.lac_segmentation.@labeled')
        main_wf.connect(prediction_metrics_lac_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.lac_metrics')

    main_wf.connect(qc_joiner, 'qc_metrics_csv', sink_node_all, 'preproc_qc')
    main_wf.connect(qc_joiner, 'bad_qc_subs', sink_node_all, 'preproc_qc.@bad_qc_subs')
    main_wf.connect(qc_joiner, 'qc_plot_svg', sink_node_all, 'preproc_qc.@qc_plot_svg')
    if args.prev_qc is not None:
        main_wf.connect(qc_joiner, 'csv_pop_file', sink_node_all, 'preproc_qc.@preproc_qc_pop')
        main_wf.connect(qc_joiner, 'pop_bad_subjects_file', sink_node_all, 'preproc_qc.@pop_bad_subjects')
    sink_node_all.inputs.wf_graph = wf_graph

    # Run the workflow
    if args.keep_all:
        config.enable_provenance()
        main_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    if args.debug:
        main_wf.config['execution']['stop_on_first_crash'] = 'True'
    main_wf.run(plugin=args.run_plugin, plugin_args=args.run_plugin_args)


if __name__ == "__main__":
    main()
