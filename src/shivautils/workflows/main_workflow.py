"""
Main workflow generator, with conditional piping (wf shape depends on the prediction types)
"""
from shivautils.interfaces.shiva import Predict, PredictSingularity
from shivautils.utils.misc import set_wf_shapers  # , as_list
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.dual_preprocessing import genWorkflow as genWorkflowDualPreproc
from shivautils.workflows.preprocessing_swi_reg import graft_workflow_swi
from shivautils.workflows.preprocessing_premasked import genWorkflow as genWorkflow_preproc_masked
from shivautils.workflows.preprocessing_synthseg import genWorkflow as genWorkflow_preproc_synthseg
from shivautils.interfaces.post import Join_Prediction_metrics, Join_QC_metrics
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink, DataGrabber
import os


def update_wf_grabber(wf, data_struct, acquisitions):
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

    # if seg == 'masked':
    #     datagrabber.inputs.field_template['brainmask'] = datagrabber.inputs.field_template['img1']
    #     datagrabber.inputs.template_args['brainmask'] = datagrabber.inputs.template_args['img1']
    return wf


def generate_main_wf(**kwargs) -> Workflow:
    """
    Generate a full processing workflow, with prepoc, pred, and postproc.
    """
    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs['PREDICTION'])

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('main_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # Name the preproc workflow
    if with_t1:
        if with_flair:
            wf_name = 'shiva_dual_preprocessing'
        else:
            wf_name = 'shiva_t1_preprocessing'
    elif with_swi and not with_t1:
        wf_name = 'shiva_swi_preprocessing'

    # Initialise the proper preproc depending on the input images and the type of preproc, and update its datagrabber
    acquisitions = []
    input_type = kwargs['PREP_SETTINGS']['input_type']

    if with_t1:
        acquisitions.append(('img1', 't1'))
        if kwargs['BRAIN_SEG'] == 'masked':
            wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'synthseg':
            wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
        else:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        if with_flair:
            acquisitions.append(('img2', 'flair'))
            wf_preproc = genWorkflowDualPreproc(wf_preproc, **kwargs)
        if with_swi:  # Adding the swi preprocessing steps to the preproc workflow
            acquisitions.append(('img3', 'swi'))
            cmb_preproc_wf_name = 'swi_preprocessing'
            wf_preproc = graft_workflow_swi(wf_preproc, **kwargs, wf_name=cmb_preproc_wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, input_type, acquisitions)
    elif with_swi and not with_t1:  # CMB alone
        acquisitions.append(('img1', 'swi'))
        if kwargs['BRAIN_SEG'] == 'masked':
            wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'synthseg':
            wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
        else:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, input_type, acquisitions)

    # Then initialise the post proc and add the nodes to the main wf
    wf_post = genWorkflowPost(**kwargs)
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

    # If there are data from previous QC entered as inputs for statistical purpose:
    prev_qc = kwargs['PREP_SETTINGS']['prev_qc']
    if prev_qc is not None:
        qc_joiner.inputs.population_csv_file = prev_qc

    # Then prediction nodes and their connections
    segmentation_wf = Workflow('Segmentation')  # facultative workflow for organization purpose
    # PVS
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_pvs = Node(PredictSingularity(), name="predict_pvs")
            predict_pvs.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_pvs.inputs.out_filename = '/mnt/data/pvs_map.nii.gz'
            predict_pvs.inputs.snglrt_enable_nvidia = True
            predict_pvs.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_pvs = Node(Predict(), name="predict_pvs")
            predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'
        predict_pvs.inputs.model = kwargs['MODELS_PATH']
        predict_pvs.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        if 'PVS2' in kwargs['PREDICTION']:
            predict_pvs.inputs.descriptor = kwargs['PVS2_DESCRIPTOR']
        else:
            predict_pvs.inputs.descriptor = kwargs['PVS_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_pvs])
        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized',
                        segmentation_wf, 'predict_pvs.t1')
        if 'PVS2' in kwargs['PREDICTION']:
            main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized',
                            segmentation_wf, 'predict_pvs.flair')
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation',
                        wf_post, 'cluster_labelling_pvs.biomarker_raw')

        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_preproc, 'custom_parc.brain_parc',
                            wf_post, 'custom_pvs_parc.brain_seg')
        else:
            main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',
                            wf_post, 'cluster_labelling_pvs.brain_seg')
            main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',
                            wf_post, 'prediction_metrics_pvs.brain_seg')

        # Merge all csv files
        prediction_metrics_pvs_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_pvs_all")
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', prediction_metrics_pvs_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_pvs_all, 'subject_id')

    # WMH
    if 'WMH' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_wmh = Node(PredictSingularity(), name="predict_wmh")
            predict_wmh.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_wmh.inputs.out_filename = '/mnt/data/wmh_map.nii.gz'
            predict_wmh.inputs.snglrt_enable_nvidia = True
            predict_wmh.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_wmh = Node(Predict(), name="predict_wmh")
            predict_wmh.inputs.out_filename = 'wmh_map.nii.gz'
        predict_wmh.inputs.model = kwargs['MODELS_PATH']
        predict_wmh.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_wmh.inputs.descriptor = kwargs['WMH_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_wmh])

        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_wmh.t1')
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_wmh.flair')
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', wf_post, 'prediction_metrics_wmh.img')
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_preproc, 'custom_parc.brain_parc',  wf_post, 'custom_wmh_parc.brain_seg')
        else:
            main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_wmh.brain_seg')
        # Merge all csv files
        prediction_metrics_wmh_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_wmh_all")
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', prediction_metrics_wmh_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_wmh_all, 'subject_id')

    # CMB
    if 'CMB' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_cmb = Node(PredictSingularity(), name="predict_cmb")
            predict_cmb.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_cmb.inputs.snglrt_enable_nvidia = True
            predict_cmb.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
            predict_cmb.inputs.out_filename = '/mnt/data/cmb_map.nii.gz'

        else:
            predict_cmb = Node(Predict(), name="predict_cmb")
            predict_cmb.inputs.out_filename = 'cmb_map.nii.gz'
        predict_cmb.inputs.model = kwargs['MODELS_PATH']
        predict_cmb.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_cmb.inputs.descriptor = kwargs['CMB_DESCRIPTOR']

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

        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_preproc, 'custom_parc.brain_parc', wf_post, 'custom_cmb_parc.brain_seg')
        else:
            main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', wf_post, 'prediction_metrics_cmb.brain_seg')

    # Lacuna
    if 'LAC' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_lac = Node(PredictSingularity(), name="predict_lac")
            predict_lac.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_lac.inputs.out_filename = '/mnt/data/lac_map.nii.gz'
            predict_lac.inputs.snglrt_enable_nvidia = True
            predict_lac.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_lac = Node(Predict(), name="predict_lac")
            predict_lac.inputs.out_filename = 'lac_map.nii.gz'
        predict_lac.inputs.model = kwargs['MODELS_PATH']
        predict_lac.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_lac.inputs.descriptor = kwargs['LAC_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_lac])

        main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_lac.t1')
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, 'predict_lac.flair')
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', wf_post, 'prediction_metrics_lac.img')
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_preproc, 'custom_parc.brain_parc', wf_post, 'custom_lac_parc.brain_seg')
        else:
            main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded',  wf_post, 'prediction_metrics_lac.brain_seg')

        # Merge all csv files
        prediction_metrics_lac_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_lac_all")
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', prediction_metrics_lac_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_lac_all, 'subject_id')

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')

    # Finally the data sinks
    # Initializing the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('flair_to_t1__Warped_img_normalized', 'flair_to_t1_cropped_intensity_normed')
    ]
    main_wf.connect(wf_post, 'summary_report.summary', sink_node_subjects, 'report')

    sink_node_all = Node(DataSink(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the sinks
    # Preproc
    if with_t1:
        img1 = 't1'
    elif with_swi and not with_t1:
        img1 = 'swi'
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, f'shiva_preproc.{img1}_preproc')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask')
    if kwargs['BRAIN_SEG'] is None:
        main_wf.connect(wf_preproc, 'mask_to_img1.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask_raw_space')
    if kwargs['BRAIN_SEG'] == 'synthseg':
        main_wf.connect(wf_preproc, 'synthseg.segmentation', sink_node_subjects, 'shiva_preproc.synthseg')
        main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, 'shiva_preproc.synthseg.@cropped')
        main_wf.connect(wf_preproc, 'custom_parc.brain_parc', sink_node_subjects, 'shiva_preproc.synthseg.@custom')
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
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation', sink_node_subjects, 'segmentations.pvs_segmentation')
        main_wf.connect(wf_post, 'cluster_labelling_pvs.labelled_biomarkers', sink_node_subjects, 'segmentations.pvs_segmentation.@labeled')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_census_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@census')
        main_wf.connect(prediction_metrics_pvs_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.pvs_metrics')
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_post, 'custom_pvs_parc.brain_seg', sink_node_subjects, 'segmentations.pvs_segmentation.@parc')
            main_wf.connect(wf_post, 'custom_pvs_parc.region_dict', sink_node_subjects, 'segmentations.pvs_segmentation.@parc_dict')
    if 'WMH' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', sink_node_subjects, 'segmentations.wmh_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_census_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.labelled_biomarkers', sink_node_subjects, 'segmentations.wmh_segmentation.@labeled')
        main_wf.connect(prediction_metrics_wmh_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.wmh_metrics')
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_post, 'custom_wmh_parc.brain_seg', sink_node_subjects, 'segmentations.wmh_segmentation.@parc')
            main_wf.connect(wf_post, 'custom_wmh_parc.region_dict', sink_node_subjects, 'segmentations.wmh_segmentation.@parc_dict')

    if 'CMB' in kwargs['PREDICTION']:
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
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_post, 'custom_cmb_parc.brain_seg', sink_node_subjects, f'segmentations.cmb_segmentation_{space}.@parc')
            main_wf.connect(wf_post, 'custom_cmb_parc.region_dict', sink_node_subjects, f'segmentations.cmb_segmentation_{space}.@parc_dict')

    if 'LAC' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', sink_node_subjects, 'segmentations.lac_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', sink_node_subjects, 'segmentations.lac_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_census_csv', sink_node_subjects, 'segmentations.lac_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_lac.labelled_biomarkers', sink_node_subjects, 'segmentations.lac_segmentation.@labeled')
        main_wf.connect(prediction_metrics_lac_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.lac_metrics')
        if kwargs['BRAIN_SEG'] == 'synthseg':
            main_wf.connect(wf_post, 'custom_lac_parc.brain_seg', sink_node_subjects, 'segmentations.lac_segmentation.@parc')
            main_wf.connect(wf_post, 'custom_lac_parc.region_dict', sink_node_subjects, 'segmentations.lac_segmentation.@parc_dict')

    main_wf.connect(qc_joiner, 'qc_metrics_csv', sink_node_all, 'preproc_qc')
    main_wf.connect(qc_joiner, 'bad_qc_subs', sink_node_all, 'preproc_qc.@bad_qc_subs')
    main_wf.connect(qc_joiner, 'qc_plot_svg', sink_node_all, 'preproc_qc.@qc_plot_svg')
    if prev_qc is not None:
        main_wf.connect(qc_joiner, 'csv_pop_file', sink_node_all, 'preproc_qc.@preproc_qc_pop')
        main_wf.connect(qc_joiner, 'pop_bad_subjects_file', sink_node_all, 'preproc_qc.@pop_bad_subjects')
    sink_node_all.inputs.wf_graph = wf_graph
    return main_wf


def generate_main_wf_grab_preproc(**kwargs) -> Workflow:
    """
    Generate a full processing workflow, without prepoc as it will grab the preprocessed data from the results folder from
    a previous run
    Mostly copy and paste from generate_main_wf, but does not copy the old preprocessed data and QC to the new results folder.
    """
    # Get the folder with the preprocessed data
    preproc_res = kwargs['PREP_SETTINGS']['preproc_res']

    # First, compare the subjects from the input folder/subject list and the ones in the input results folder
    available_preproc = os.listdir(os.path.join(preproc_res, 'qc_metrics'))  # There should always be a 'qc_metrics' folder in the results
    missing_subj = sorted(list(set(kwargs['SUBJECT_LIST']) - set(available_preproc)))
    if missing_subj:
        miss_subj_file = os.path.join(kwargs['BASE_DIR'], 'missing_datasets.txt')
        with open(miss_subj_file, 'w') as file:
            for subID in missing_subj:
                # write each item on a new line
                file.write(f"{subID}\n")
        error_msg = (
            '\nSome of the datasets from the input folder did not have corresponding preprocessed data in the input results folder '
            f'({len(missing_subj)} out of {len(available_preproc)}). The IDs of the culprits have been written in a text file:'
            f'\n{miss_subj_file}\n'
            'To continue, you can:\n'
            '   - Process the datasets which lack preprocessed data the normal way with the full shiva pipeline (to complete the preprocessed folder)\n'
            '   - Remove the datasets from the subject list if you are providing one with --sub_list\n'
            '   - Add the datasets to the exlusion list with --exclusion_list (if you are not using --sub_list). You can even provide the text file generated here as is.\n'
        )
        raise ValueError(error_msg)

    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs['PREDICTION'])

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('main_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # Initializin the preprocessed data grabber
    preproc_grabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['t1_intensity_normalized',
                   'flair_intensity_normalized',
                   'swi_intensity_normalized',
                   'swi_to_t1_transforms',
                   'brain_seg',  # TODO: Make it brainmask and Synthseg compatible
                   ]),
        name='preproc_grabber')
    preproc_grabber.inputs.base_directory = preproc_res
    preproc_grabber.inputs.template = '*.nii.gz'  # unused placeholder (but required)
    preproc_grabber.inputs.raise_on_empty = True
    preproc_grabber.inputs.sort_filelist = True

    # Set the preproc datagrabber input files
    field_template = {}
    template_args = {}
    if with_t1:
        field_template['t1_intensity_normalized'] = 't1_preproc/%s/*_cropped_intensity_normed.nii.gz'
        template_args['t1_intensity_normalized'] = [['subject_id']]
        if kwargs['BRAIN_SEG'] == 'synthseg':
            pass  # TODO
        else:
            field_template['brain_seg'] = 't1_preproc/%s/brainmask_cropped.nii.gz'
            template_args['brain_seg'] = [['subject_id']]
        if with_flair:
            field_template['flair_intensity_normalized'] = 'flair_preproc/%s/flair_to_t1_cropped_intensity_normed.nii.gz'
            template_args['flair_intensity_normalized'] = [['subject_id']]
    elif with_swi:
        field_template['swi_intensity_normalized'] = 'swi_preproc/%s/*_cropped_intensity_normed.nii.gz'
        template_args['swi_intensity_normalized'] = [['subject_id']]
        if not with_t1:
            if kwargs['BRAIN_SEG'] == 'synthseg':
                pass  # TODO
            else:
                field_template['brain_seg'] = 'swi_preproc/%s/brainmask_cropped.nii.gz'
                template_args['brain_seg'] = [['subject_id']]
    preproc_grabber.inputs.field_template = field_template
    preproc_grabber.inputs.template_args = template_args
    main_wf.connect(subject_iterator, 'subject_id', preproc_grabber, 'subject_id')

    # Then initialise the post proc and add the nodes to the main wf
    wf_post = genWorkflowPost(**kwargs)
    main_wf.add_nodes([wf_post])

    # Set all the connections between preproc and postproc
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(preproc_grabber, 'brain_seg', wf_post, 'summary_report.brainmask')

    # Then prediction nodes and their connections
    segmentation_wf = Workflow('Segmentation')  # facultative workflow for organization purpose
    # PVS
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_pvs = Node(PredictSingularity(), name="predict_pvs")
            predict_pvs.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_pvs.inputs.out_filename = '/mnt/data/pvs_map.nii.gz'
            predict_pvs.inputs.snglrt_enable_nvidia = True
            predict_pvs.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_pvs = Node(Predict(), name="predict_pvs")
            predict_pvs.inputs.out_filename = 'pvs_map.nii.gz'
        predict_pvs.inputs.model = kwargs['MODELS_PATH']
        predict_pvs.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        if 'PVS2' in kwargs['PREDICTION']:
            predict_pvs.inputs.descriptor = kwargs['PVS2_DESCRIPTOR']
        else:
            predict_pvs.inputs.descriptor = kwargs['PVS_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_pvs])
        main_wf.connect(preproc_grabber, 't1_intensity_normalized', segmentation_wf, 'predict_pvs.t1')
        if 'PVS2' in kwargs['PREDICTION']:
            main_wf.connect(preproc_grabber, 'flair_intensity_normalized', segmentation_wf, 'predict_pvs.flair')
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation', wf_post, 'prediction_metrics_pvs.img')
        main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, 'cluster_labelling_pvs.brain_seg')
        main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, 'prediction_metrics_pvs.brain_seg')

        # Merge all csv files
        prediction_metrics_pvs_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_pvs_all")
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', prediction_metrics_pvs_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_pvs_all, 'subject_id')

    # WMH
    if 'WMH' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_wmh = Node(PredictSingularity(), name="predict_wmh")
            predict_wmh.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_wmh.inputs.out_filename = '/mnt/data/wmh_map.nii.gz'
            predict_wmh.inputs.snglrt_enable_nvidia = True
            predict_wmh.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_wmh = Node(Predict(), name="predict_wmh")
            predict_wmh.inputs.out_filename = 'wmh_map.nii.gz'
        predict_wmh.inputs.model = kwargs['MODELS_PATH']
        predict_wmh.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_wmh.inputs.descriptor = kwargs['WMH_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_wmh])

        main_wf.connect(preproc_grabber, 't1_intensity_normalized', segmentation_wf, 'predict_wmh.t1')
        main_wf.connect(preproc_grabber, 'flair_intensity_normalized', segmentation_wf, 'predict_wmh.flair')
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', wf_post, 'prediction_metrics_wmh.img')
        main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, 'prediction_metrics_wmh.brain_seg')  # TODO: SynthSeg
        # Merge all csv files
        prediction_metrics_wmh_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_wmh_all")
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', prediction_metrics_wmh_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_wmh_all, 'subject_id')

    # CMB
    if 'CMB' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_cmb = Node(PredictSingularity(), name="predict_cmb")
            predict_cmb.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_cmb.inputs.snglrt_enable_nvidia = True
            predict_cmb.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
            predict_cmb.inputs.out_filename = '/mnt/data/cmb_map.nii.gz'

        else:
            predict_cmb = Node(Predict(), name="predict_cmb")
            predict_cmb.inputs.out_filename = 'cmb_map.nii.gz'
        predict_cmb.inputs.model = kwargs['MODELS_PATH']
        predict_cmb.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_cmb.inputs.descriptor = kwargs['CMB_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_cmb])

        # Merge all csv files
        prediction_metrics_cmb_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_cmb_all")
        main_wf.connect(wf_post, 'prediction_metrics_cmb.biomarker_stats_csv', prediction_metrics_cmb_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_cmb_all, 'subject_id')
        main_wf.connect(preproc_grabber, 'swi_intensity_normalized', segmentation_wf, 'predict_cmb.swi')
        if with_t1:
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', wf_post, 'swi_pred_to_t1.input_image')
            main_wf.connect(preproc_grabber, 'swi_to_t1_transforms', wf_post, 'swi_pred_to_t1.transforms')
            main_wf.connect(preproc_grabber, 't1_intensity_normalized', wf_post, 'swi_pred_to_t1.reference_image')
        else:
            main_wf.connect(segmentation_wf, 'predict_cmb.segmentation', wf_post, 'prediction_metrics_cmb.img')
        main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, 'prediction_metrics_cmb.brain_seg')  # TODO: SynthSeg

    # Lacuna
    if 'LAC' in kwargs['PREDICTION']:
        if kwargs['CONTAINERIZE_NODES']:
            predict_lac = Node(PredictSingularity(), name="predict_lac")
            predict_lac.inputs.snglrt_bind = [
                (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
                ('`pwd`', '/mnt/data', 'rw'),
                (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
            predict_lac.inputs.out_filename = '/mnt/data/lac_map.nii.gz'
            predict_lac.inputs.snglrt_enable_nvidia = True
            predict_lac.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']
        else:
            predict_lac = Node(Predict(), name="predict_lac")
            predict_lac.inputs.out_filename = 'lac_map.nii.gz'
        predict_lac.inputs.model = kwargs['MODELS_PATH']
        predict_lac.plugin_args = kwargs['PRED_PLUGIN_ARGS']
        predict_lac.inputs.descriptor = kwargs['LAC_DESCRIPTOR']

        segmentation_wf.add_nodes([predict_lac])

        main_wf.connect(preproc_grabber, 't1_intensity_normalized', segmentation_wf, 'predict_lac.t1')
        main_wf.connect(preproc_grabber, 'flair_intensity_normalized', segmentation_wf, 'predict_lac.flair')
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', wf_post, 'prediction_metrics_lac.img')
        main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, 'prediction_metrics_lac.brain_seg')  # TODO: SynthSeg
        # Merge all csv files
        prediction_metrics_lac_all = JoinNode(Join_Prediction_metrics(),
                                              joinsource=subject_iterator,
                                              joinfield=['csv_files', 'subject_id'],
                                              name="prediction_metrics_lac_all")
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', prediction_metrics_lac_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_lac_all, 'subject_id')

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')

    # Finally the data sinks
    # Initializing the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('flair_to_t1__Warped_img_normalized', 'flair_to_t1_cropped_intensity_normed')
    ]
    main_wf.connect(wf_post, 'summary_report.summary', sink_node_subjects, 'report')

    sink_node_all = Node(DataSink(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the sinks
    # Pred and postproc
    if 'PVS' in kwargs['PREDICTION'] or 'PVS2' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_pvs.segmentation', sink_node_subjects, 'segmentations.pvs_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_stats_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.biomarker_census_csv', sink_node_subjects, 'segmentations.pvs_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_pvs.labelled_biomarkers', sink_node_subjects, 'segmentations.pvs_segmentation.@labeled')
        main_wf.connect(prediction_metrics_pvs_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.pvs_metrics')

    if 'WMH' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_wmh.segmentation', sink_node_subjects, 'segmentations.wmh_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_stats_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.biomarker_census_csv', sink_node_subjects, 'segmentations.wmh_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_wmh.labelled_biomarkers', sink_node_subjects, 'segmentations.wmh_segmentation.@labeled')
        main_wf.connect(prediction_metrics_wmh_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.wmh_metrics')

    if 'CMB' in kwargs['PREDICTION']:
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

    if 'LAC' in kwargs['PREDICTION']:
        main_wf.connect(segmentation_wf, 'predict_lac.segmentation', sink_node_subjects, 'segmentations.lac_segmentation')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_stats_csv', sink_node_subjects, 'segmentations.lac_segmentation.@metrics')
        main_wf.connect(wf_post, 'prediction_metrics_lac.biomarker_census_csv', sink_node_subjects, 'segmentations.lac_segmentation.@census')
        main_wf.connect(wf_post, 'prediction_metrics_lac.labelled_biomarkers', sink_node_subjects, 'segmentations.lac_segmentation.@labeled')
        main_wf.connect(prediction_metrics_lac_all, 'prediction_metrics_csv', sink_node_all, 'segmentations.lac_metrics')

    sink_node_all.inputs.wf_graph = wf_graph
    return main_wf


def generate_main_wf_preproc(**kwargs) -> Workflow:
    """
    Generate a processing workflow with only prepoc
    Untested, probably unfinished.
    """
    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs['PREDICTION'])

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('main_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # First, initialise the proper preproc and update its datagrabber
    acquisitions = []
    input_type = kwargs['PREP_SETTINGS']['input_type']
    if with_t1:
        acquisitions.append(('img1', 't1'))
        if with_flair:
            acquisitions.append(('img2', 'flair'))
            wf_preproc = genWorkflowDualPreproc(**kwargs, wf_name='shiva_dual_preprocessing')
            # if needed, genWorkflow_preproc_masked is used from inside genWorkflowDualPreproc
        else:
            wf_name = 'shiva_t1_preprocessing'
            if kwargs['BRAIN_SEG'] == 'masked':
                wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
            elif kwargs['BRAIN_SEG'] == 'synthseg':
                wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
            else:
                wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        if with_swi:  # Adding the swi preprocessing steps to the preproc workflow
            acquisitions.append(('img3', 'swi'))
            cmb_preproc_wf_name = 'swi_preprocessing'
            wf_preproc = graft_workflow_swi(wf_preproc, **kwargs, wf_name=cmb_preproc_wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, input_type, acquisitions)
    elif with_swi and not with_t1:  # CMB alone
        acquisitions.append(('img1', 'swi'))
        wf_name = 'shiva_swi_preprocessing'
        if kwargs['BRAIN_SEG'] == 'masked':
            wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'synthseg':
            wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
        else:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        wf_preproc = update_wf_grabber(wf_preproc, input_type, acquisitions)

    # Then initialise the post proc and add the nodes to the main wf
    wf_post = genWorkflowPost(**kwargs)
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

    # Initializing the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('flair_to_t1__Warped_img_normalized', 'flair_to_t1_cropped_intensity_normed')
    ]
    main_wf.connect(wf_post, 'summary_report.summary', sink_node_subjects, 'report')
    # Connecting the sinks
    if with_t1:
        img1 = 't1'
    elif with_swi and not with_t1:
        img1 = 'swi'
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, f'shiva_preproc.{img1}_preproc')
    main_wf.connect(wf_preproc, 'hard_post_brain_mask.thresholded', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask')
    if kwargs['BRAIN_SEG'] is None:
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
    return main_wf
