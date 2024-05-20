"""
Main workflow generator, with conditional piping (wf shape depends on the prediction types)
"""
from shivautils.interfaces.shiva import Predict, PredictSingularity
from shivautils.utils.misc import set_wf_shapers  # , as_list
from shivautils.workflows.post_processing import genWorkflow as genWorkflowPost
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.dual_preprocessing import graft_img2_preproc
from shivautils.workflows.preprocessing_swi_reg import graft_workflow_swi
from shivautils.workflows.swomed_graft_infiles import graft_swomed_infiles
from shivautils.workflows.preprocessing_shiva_masking import genWorkflow as genWorkflow_preproc_shiva_mask
from shivautils.workflows.preprocessing_premasked import genWorkflow as genWorkflow_preproc_masked
from shivautils.workflows.preprocessing_synthseg import genWorkflow as genWorkflow_preproc_synthseg
from shivautils.workflows.preprocessing_synthseg_precomp import genWorkflow as genWorkflow_preproc_synthseg_precomp
from shivautils.workflows.preprocessing_swomed_pre_synthseg import genWorkflow as genWorkflow_preproc_synthseg_swomed
from shivautils.workflows.preprocessing_custom_seg import genWorkflow as genWorkflow_preproc_custom_seg
from shivautils.workflows.predict_wf import genWorkflow as genWorkflow_prediction
from shivautils.workflows.dcm2nii_grafting import graft_dcm2nii
from shivautils.interfaces.post import Join_Prediction_metrics, Join_QC_metrics
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber
from shivautils.interfaces.datasink import DataSink_CSV_and_PDF_safe
import os


def update_wf_grabber(wf, acquisitions, datatype, kwargs):
    """
    Updates the workflow datagrabber to work with the different types on input
        wf: workflow with the datagrabber
        acquisitions example: [('img1', 't1'), ('img2', 'flair')]
        datatype ('nifti' or 'dicom')
        custom_seg (bool): wether there is a custom segmentation (brain mask or brain parc) available
    """
    files = '' if datatype == 'dicom' else '*.nii*'  # no files for dcm, just the whole folder
    datagrabber = wf.get_node('datagrabber')
    data_struct = kwargs['PREP_SETTINGS']['input_type']
    if data_struct in ['standard', 'json']:
        # e.g: {'img1': '%s/t1/%s_T1_raw.nii.gz'}
        datagrabber.inputs.field_template = {acq[0]: f'%s/{acq[1]}/{files}' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id']] for acq in acquisitions}
        if kwargs['BRAIN_SEG'] == 'custom':
            datagrabber.inputs.field_template['seg'] = f'%s/seg/*.nii*'  # We expect a nifti here, as dicom is unlikely
            datagrabber.inputs.template_args['seg'] = [['subject_id']]

    if data_struct == 'BIDS':
        if datatype == 'dicom':
            raise ValueError('BIDS data structure not compatible with DICOM input')
        # e.g: {'img1': '%s/anat/%s_T1_raw.nii.gz}
        datagrabber.inputs.field_template = {acq[0]: f'%s/anat/%s_{acq[1].upper()}*.nii*' for acq in acquisitions}
        datagrabber.inputs.template_args = {acq[0]: [['subject_id', 'subject_id']] for acq in acquisitions}
        if kwargs['BRAIN_SEG'] == 'custom':  # TODO: Correct this for proper bids format. It should actually be in the "derived" folder...
            datagrabber.inputs.field_template['seg'] = '%s/anat/%s_*seg*.nii*'
            datagrabber.inputs.template_args['seg'] = [['subject_id', 'subject_id']]

    if data_struct == 'swomed':
        in_files_dict = kwargs['PREP_SETTINGS']['swomed_input']
        for imgN, acq in acquisitions:
            setattr(datagrabber.inputs, imgN, in_files_dict[acq])
        if 'seg' in in_files_dict:
            datagrabber.inputs.seg = in_files_dict['seg']
        if 'synthseg_vol' in in_files_dict:
            datagrabber.inputs.seg = in_files_dict['synthseg_vol']
        if 'synthseg_qc' in in_files_dict:
            datagrabber.inputs.seg = in_files_dict['synthseg_qc']

    if datatype == 'dicom':
        wf = graft_dcm2nii(wf, **kwargs)

    return wf


def generate_main_wf(**kwargs) -> Workflow:
    """
    Generate a full processing workflow, with prepoc, pred, and postproc.
    """
    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs['PREDICTION'])

    if kwargs['USE_T1']:  # Override the default with_t1 deduced from the predictions
        with_t1 = True

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
    file_type = kwargs['PREP_SETTINGS']['file_type']

    if with_t1:
        # What main acquisition to use
        if kwargs['ACQUISITIONS']['t1-like']:
            acquisitions.append(('img1', kwargs['ACQUISITIONS']['t1-like']))
        else:
            acquisitions.append(('img1', 't1'))

        # What type of preprocessing (basic / synthseg / premasked / custom input)
        if 'shiva' in kwargs['BRAIN_SEG']:
            wf_preproc = genWorkflow_preproc_shiva_mask(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'premasked':
            wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
        elif 'synthseg' in kwargs['BRAIN_SEG']:
            if kwargs['PREP_SETTINGS']['input_type'] == 'swomed':
                wf_preproc = genWorkflow_preproc_synthseg_swomed(**kwargs, wf_name=wf_name)
                # seg_cleaning = wf_preproc.get_node('seg_cleaning')
                # datagrabber = wf_preproc.get_node('datagrabber')
                # wf_preproc.connect(datagrabber, 'seg', seg_cleaning, 'input_seg')
            else:
                if kwargs['BRAIN_SEG'] == 'synthseg_precomp':
                    wf_preproc = genWorkflow_preproc_synthseg_precomp(**kwargs, wf_name=wf_name)
                else:
                    wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
            wf_preproc = genWorkflow_preproc_custom_seg(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is None:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        else:
            raise NotImplementedError(f'The brain segmentation type "{kwargs["BRAIN_SEG"]}" was not recognized')

        # Checking if dual preprocessing is needed (and chich type of secondary aquisition)
        if with_flair:
            if kwargs['ACQUISITIONS']['flair-like']:
                acquisitions.append(('img2', kwargs['ACQUISITIONS']['flair-like']))
            else:
                acquisitions.append(('img2', 'flair'))
            wf_preproc = graft_img2_preproc(wf_preproc, **kwargs)

        # Checking if SWI (or equivalent) need to be preprocessed
        if with_swi:  # Adding the swi preprocessing steps to the preproc workflow
            if kwargs['ACQUISITIONS']['swi-like']:
                acquisitions.append(('img3', kwargs['ACQUISITIONS']['swi-like']))
            else:
                acquisitions.append(('img3', 'swi'))
            wf_preproc = graft_workflow_swi(wf_preproc, **kwargs)

    elif with_swi and not with_t1:  # CMB alone
        if kwargs['ACQUISITIONS']['swi-like']:
            acquisitions.append(('img1', kwargs['ACQUISITIONS']['swi-like']))
        else:
            acquisitions.append(('img1', 'swi'))
        if 'shiva' in kwargs['BRAIN_SEG']:
            wf_preproc = genWorkflow_preproc_shiva_mask(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'premasked':
            wf_preproc = genWorkflow_preproc_masked(**kwargs, wf_name=wf_name)
        elif 'synthseg' in kwargs['BRAIN_SEG']:
            if kwargs['PREP_SETTINGS']['input_type'] == 'swomed':
                wf_preproc = genWorkflow_preproc_synthseg_swomed(**kwargs, wf_name=wf_name)
            else:
                if kwargs['BRAIN_SEG'] == 'synthseg_precomp':
                    wf_preproc = genWorkflow_preproc_synthseg_precomp(**kwargs, wf_name=wf_name)
                else:
                    wf_preproc = genWorkflow_preproc_synthseg(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
            wf_preproc = genWorkflow_preproc_custom_seg(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is None:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        else:
            raise NotImplementedError(f'The brain segmentation type "{kwargs["BRAIN_SEG"]}" was not recognized')

    # Swap the datagrabber for a direct file input in SWOMed case, if not with Synthseg
    if kwargs['PREP_SETTINGS']['input_type'] == 'swomed' and not 'synthseg' in kwargs['BRAIN_SEG']:
        wf_preproc = graft_swomed_infiles(wf_preproc)

    # Updating the datagrabber with all this info
    wf_preproc = update_wf_grabber(wf_preproc, acquisitions, file_type, kwargs)

    # Then initialise the post proc and add the nodes to the main wf
    wf_post = genWorkflowPost(**kwargs)
    main_wf.add_nodes([wf_preproc, wf_post])

    # Set all the connections between preproc and postproc
    main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'datagrabber.subject_id')
    if kwargs['BRAIN_SEG'] == 'synthseg_precomp':
        main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'synthseg_grabber.subject_id')
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_crop_box.crop_brain_img', wf_post, 'summary_report.crop_brain_img')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask.overlayed_brainmask', wf_post, 'summary_report.overlayed_brainmask_1')
    if with_swi and with_t1:
        main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask_swi.overlayed_brainmask', wf_post, 'summary_report.overlayed_brainmask_2')
    if with_flair:
        main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_coreg_FLAIR_T1.qc_coreg', wf_post, 'summary_report.isocontour_slides_FLAIR_T1')
    main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, 'summary_report.brainmask')

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

    # Then prediction workflow and all its connections, and other prediction-specific connections
    segmentation_wf = genWorkflow_prediction(**kwargs)
    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers([pred])
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()
        # Connection with inputs
        if pred_with_t1:
            main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, f'predict_{lpred}.t1')
        if pred_with_flair:
            main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', segmentation_wf, f'predict_{lpred}.flair')
        if pred_with_swi:
            if with_t1:  # t1 used in the wf, not necessarily in the pred
                main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_intensity_normalisation.intensity_normalized', segmentation_wf, f'predict_{lpred}.swi')
                main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_intensity_normalisation.intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')
                main_wf.connect(wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image', wf_post, f'{lpred}_overlay_node.fov_mask')
            else:
                main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', segmentation_wf, f'predict_{lpred}.swi')
                main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')
                main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, f'{lpred}_overlay_node.fov_mask')
        else:
            main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, f'{lpred}_overlay_node.fov_mask')
            if pred in ['WMH', 'LAC']:  # Using FLAIR as background for WMH and LAC for the pred overlay node
                main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')
            else:
                main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')
        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, f'{lpred}_overlay_node.brainmask')

        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, f'cluster_labelling_{lpred}.biomarker_raw')

        if pred_with_swi and with_t1:
            if 'synthseg' in kwargs['BRAIN_SEG']:
                main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_to_t1.forward_transforms', wf_post, 'seg_to_swi.transforms')
                main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, 'seg_to_swi.reference_image')
                main_wf.connect(wf_preproc, 'custom_parc.brain_parc', wf_post, 'seg_to_swi.input_image')
            elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
                main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_to_t1.forward_transforms', wf_post, 'seg_to_swi.transforms')
                main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, 'seg_to_swi.reference_image')
                main_wf.connect(wf_preproc, 'seg_to_crop.resampled_image', wf_post, 'seg_to_swi.input_image')
            else:
                main_wf.connect(wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image', wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image', wf_post, 'prediction_metrics_cmb.brain_seg')
        else:
            if 'synthseg' in kwargs['BRAIN_SEG']:
                main_wf.connect(wf_preproc, 'custom_parc.brain_parc', wf_post, f'custom_{lpred}_parc.brain_seg')
            elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
                main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(wf_preproc, 'seg_to_crop.resampled_image', wf_post, f'prediction_metrics_{lpred}.brain_seg')
            else:
                main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', wf_post, f'prediction_metrics_{lpred}.brain_seg')

        # Merge all csv files
        prediction_metrics_all = JoinNode(Join_Prediction_metrics(),
                                          joinsource=subject_iterator,
                                          joinfield=['csv_files', 'subject_id'],
                                          name=f'prediction_metrics_{lpred}_all')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv',
                        prediction_metrics_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id',
                        prediction_metrics_all, 'subject_id')

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')

    # Finally the data sinks
    # Initializing the data sinks
    sink_node_subjects = Node(DataSink_CSV_and_PDF_safe(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('_resampled_defaced_cropped_img_normalized', '_defaced_cropped_intensity_normed'),
        ('flair_to_t1__Warped_defaced_img_normalized', 'flair_to_t1_defaced_cropped_intensity_normed')
    ]
    main_wf.connect(wf_post, 'summary_report.pdf_report', sink_node_subjects, 'report')

    sink_node_all = Node(DataSink_CSV_and_PDF_safe(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the sinks
    # Preproc
    if with_t1:
        img1 = 't1'
    elif with_swi and not with_t1:
        img1 = 'swi'
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, f'shiva_preproc.{img1}_preproc')
    main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask')
    if file_type == 'dicom':
        main_wf.connect(wf_preproc, 'dicom2nifti_img1.converted_files', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@converted')
        main_wf.connect(wf_preproc, 'dicom2nifti_img1.bids', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@converted_bids')
    # if kwargs['BRAIN_SEG'] is None:
    #     main_wf.connect(wf_preproc, 'mask_to_img1.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask_raw_space')
    if 'synthseg' in kwargs['BRAIN_SEG']:
        main_wf.connect(wf_preproc, 'seg_cleaning.ouput_seg', sink_node_subjects, 'shiva_preproc.synthseg')
        main_wf.connect(wf_preproc, 'seg_cleaning.sunk_islands', sink_node_subjects, 'shiva_preproc.synthseg.@removed')
        main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, 'shiva_preproc.synthseg.@cropped')
        main_wf.connect(wf_preproc, 'custom_parc.brain_parc', sink_node_subjects, 'shiva_preproc.synthseg.@custom')
        if kwargs['PREP_SETTINGS']['input_type'] == 'swomed':
            main_wf.connect(wf_preproc, 'datagrabber.synthseg_vol', sink_node_subjects, 'shiva_preproc.synthseg.@vol')
            main_wf.connect(wf_preproc, 'datagrabber.synthseg_qc', sink_node_subjects, 'shiva_preproc.synthseg.@qc')
        else:
            main_wf.connect(wf_preproc, 'synthseg.volumes', sink_node_subjects, 'shiva_preproc.synthseg.@vol')
    elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
        main_wf.connect(wf_preproc, 'seg_to_crop.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@seg')
    main_wf.connect(wf_preproc, 'crop.bbox1_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb1')
    main_wf.connect(wf_preproc, 'crop.bbox2_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb2')
    main_wf.connect(wf_preproc, 'crop.cdg_ijk_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@cdg')
    if with_flair:
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', sink_node_subjects, 'shiva_preproc.flair_preproc')
        if file_type == 'dicom':
            main_wf.connect(wf_preproc, 'dicom2nifti_img2.converted_files', sink_node_subjects, f'shiva_preproc.flair_preproc.@converted')
            main_wf.connect(wf_preproc, 'dicom2nifti_img2.bids', sink_node_subjects, f'shiva_preproc.flair_preproc.@converted_bids')
    if with_swi and with_t1:
        main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_intensity_normalisation.intensity_normalized', sink_node_subjects, 'shiva_preproc.swi_preproc')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image', sink_node_subjects, 'shiva_preproc.swi_preproc.@brain_mask')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_to_t1.warped_image', sink_node_subjects, 'shiva_preproc.swi_preproc.@reg_to_t1')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.swi_to_t1.forward_transforms', sink_node_subjects, 'shiva_preproc.swi_preproc.@reg_to_t1_transf')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.crop_swi.bbox1_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@bb1')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.crop_swi.bbox2_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@bb2')
        main_wf.connect(wf_preproc, 'cmb_preprocessing.crop_swi.cdg_ijk_file', sink_node_subjects, 'shiva_preproc.swi_preproc.@cdg')
        if file_type == 'dicom':
            main_wf.connect(wf_preproc, 'dicom2nifti_img3.converted_files', sink_node_subjects, f'shiva_preproc.swi_preproc.@converted')
            main_wf.connect(wf_preproc, 'dicom2nifti_img3.bids', sink_node_subjects, f'shiva_preproc.swi_preproc.@converted_bids')
    main_wf.connect(wf_preproc, 'preproc_qc_workflow.qc_metrics.csv_qc_metrics', sink_node_subjects, 'shiva_preproc.qc_metrics')

    # Pred and postproc
    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers([pred])
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()
        if pred_with_swi:
            space = '_swi-space'
        else:
            space = ''

        prediction_metrics_all = main_wf.get_node(f'prediction_metrics_{lpred}_all')
        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}')
        main_wf.connect(wf_post, f'cluster_labelling_{lpred}.labelled_biomarkers', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@labelled')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_wide_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics_wide')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_census_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@census')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_wide_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}.@wide')
        if 'synthseg' in kwargs['BRAIN_SEG']:
            main_wf.connect(wf_post, f'custom_{lpred}_parc.brain_seg', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc')
            main_wf.connect(wf_post, f'custom_{lpred}_parc.region_dict_json', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc_dict')

        if pred_with_swi and with_t1:
            # main_wf.connect(wf_post, 'swi_clust_to_t1.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation_t1-space')  # TODO at some point
            if 'synthseg' in kwargs['BRAIN_SEG']:
                main_wf.connect(wf_post, 'seg_to_swi.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@custom_parc')

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
    if kwargs['USE_T1']:  # Override the default with_t1 deduced from the predictions
        with_t1 = True

    # Declaration of the main workflow, it is modular and will contain smaller workflows
    main_wf = Workflow('main_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    # Start by initialising the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # Initialising the preprocessed data grabber
    preproc_grabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['t1_intensity_normalized',
                   'flair_intensity_normalized',
                   'swi_intensity_normalized',
                   'swi_to_t1_transforms',
                   'brain_mask',
                   'brain_seg',
                   'swi2t1_transforms',  # for CMB when whith_t1 and brain parc
                   'brain_mask_swi',  # for CMB when whith_t1 without brain parc
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
        field_template['brain_mask'] = 't1_preproc/%s/brainmask_cropped.nii.gz'
        template_args['brain_mask'] = [['subject_id']]
        if with_flair:
            field_template['flair_intensity_normalized'] = 'flair_preproc/%s/*_cropped_intensity_normed.nii.gz'
            template_args['flair_intensity_normalized'] = [['subject_id']]
    if with_swi:
        field_template['swi_intensity_normalized'] = 'swi_preproc/%s/*_cropped_intensity_normed.nii.gz'
        template_args['swi_intensity_normalized'] = [['subject_id']]
        if not with_t1:
            field_template['brain_mask'] = 'swi_preproc/%s/brainmask_cropped*.nii.gz'
            template_args['brain_mask'] = [['subject_id']]
        else:
            if 'synthseg' in kwargs['BRAIN_SEG'] or (kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None):
                field_template['swi2t1_transforms'] = 'swi_preproc/%s/swi_to_t1_0GenericAffine.mat'
                template_args['swi2t1_transforms'] = [['subject_id']]
            else:
                field_template['brain_mask_swi'] = 'swi_preproc/%s/brainmask_cropped_swi-space.nii.gz'
                template_args['template_args'] = [['subject_id']]

    if 'synthseg' in kwargs['BRAIN_SEG']:
        field_template['brain_seg'] = 'synthseg/%s/derived_parc.nii.gz'
        template_args['brain_seg'] = [['subject_id']]
    elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
        if with_t1:
            field_template['brain_seg'] = 't1_preproc/%s/custom_seg_cropped.nii.gz'
        elif with_swi and not with_t1:
            field_template['brain_seg'] = 'swi_preproc/%s/custom_seg_cropped.nii.gz'
        template_args['brain_seg'] = [['subject_id']]

    preproc_grabber.inputs.field_template = field_template
    preproc_grabber.inputs.template_args = template_args
    main_wf.connect(subject_iterator, 'subject_id', preproc_grabber, 'subject_id')

    # Then initialise the post proc and summary sink node
    wf_post = genWorkflowPost(**kwargs)
    main_wf.add_nodes([wf_post])

    sink_node_all = Node(DataSink_CSV_and_PDF_safe(infields=['wf_graph']), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Set generic connections between preproc and postproc
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(preproc_grabber, 'brain_mask', wf_post, 'summary_report.brainmask')

    # Then prediction nodes and their connections and post proc
    segmentation_wf = genWorkflow_prediction(**kwargs)
    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers([pred])
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()

        if pred_with_t1:
            main_wf.connect(preproc_grabber, 't1_intensity_normalized', segmentation_wf, f'predict_{lpred}.t1')
            main_wf.connect(preproc_grabber, 't1_intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')
        if pred_with_flair:
            main_wf.connect(preproc_grabber, 'flair_intensity_normalized', segmentation_wf, f'predict_{lpred}.flair')
        if pred_with_swi:
            main_wf.connect(preproc_grabber, 'swi_intensity_normalized', segmentation_wf, f'predict_{lpred}.swi')
            main_wf.connect(preproc_grabber, 'swi_intensity_normalized', wf_post, f'{lpred}_overlay_node.img_ref')

        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, f'{lpred}_overlay_node.brainmask')
        main_wf.connect(preproc_grabber, 'brain_mask', wf_post, f'{lpred}_overlay_node.fov_mask')

        # main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, f'prediction_metrics_{lpred}.img')
        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', wf_post, f'cluster_labelling_{lpred}.biomarker_raw')

        if pred_with_swi and with_t1:  # Case with registration steps
            if 'synthseg' in kwargs['BRAIN_SEG'] or (kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None):
                main_wf.connect(preproc_grabber, 'swi2t1_transforms',
                                wf_post, 'seg_to_swi.transforms')
                main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation',
                                wf_post, 'seg_to_swi.reference_image')
                main_wf.connect(preproc_grabber, 'brain_seg',
                                wf_post, 'seg_to_swi.input_image')
            else:
                main_wf.connect(preproc_grabber, 'brain_mask_swi',
                                wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(preproc_grabber, 'brain_mask_swi',
                                wf_post, 'prediction_metrics_cmb.brain_seg')
        else:
            if 'synthseg' in kwargs['BRAIN_SEG']:
                main_wf.connect(preproc_grabber, 'brain_seg',  wf_post, f'custom_{lpred}_parc.brain_seg')
            elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
                main_wf.connect(preproc_grabber, 'brain_mask',  wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(preproc_grabber, 'brain_seg', wf_post, f'prediction_metrics_{lpred}.brain_seg')
            else:
                main_wf.connect(preproc_grabber, 'brain_mask',  wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(preproc_grabber, 'brain_mask',  wf_post, f'prediction_metrics_{lpred}.brain_seg')

        # Merge all csv files
        prediction_metrics_all = JoinNode(Join_Prediction_metrics(),
                                          joinsource=subject_iterator,
                                          joinfield=['csv_files', 'subject_id'],
                                          name=f'prediction_metrics_{lpred}_all')
        if pred_with_swi:
            space = '_swi-space'
        else:
            space = ''
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv', prediction_metrics_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id', prediction_metrics_all, 'subject_id')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_wide_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}.@wide')

    # The workflow graph
    wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')

    # Finally the data sinks
    # Initializing the data sinks
    sink_node_subjects = Node(DataSink_CSV_and_PDF_safe(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    # Name substitutions in the results
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('_resampled_defaced_cropped_img_normalized', '_defaced_cropped_intensity_normed'),
        ('flair_to_t1__Warped_defaced_img_normalized', 'flair_to_t1_defaced_cropped_intensity_normed')
    ]
    main_wf.connect(wf_post, 'summary_report.pdf_report', sink_node_subjects, 'report')

    # Connecting the sinks
    # Pred and postproc
    for pred in kwargs['PREDICTION']:
        if pred == 'PVS2':
            pred = 'PVS'
        if pred == 'CMB':
            space = '_swi-space'
        else:
            space = ''
        lpred = pred.lower()
        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentation', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_wide_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics_wide')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_census_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@census')
        main_wf.connect(wf_post, f'cluster_labelling_{lpred}.labelled_biomarkers', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@labeled')
        if 'synthseg' in kwargs['BRAIN_SEG']:
            main_wf.connect(wf_post, f'custom_{lpred}_parc.brain_seg', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc')
            main_wf.connect(wf_post, f'custom_{lpred}_parc.region_dict_json', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc_dict')
        if pred == 'CMB' and with_t1:
            # main_wf.connect(wf_post, 'swi_clust_to_t1.output_image', sink_node_subjects, f'segmentations.cmb_segmentation_t1-space')  # not implemented yet
            if 'synthseg' in kwargs['BRAIN_SEG']:
                main_wf.connect(wf_post, 'seg_to_swi.output_image', sink_node_subjects, 'segmentations.cmb_segmentation_swi-space.@custom_parc')

    sink_node_all.inputs.wf_graph = wf_graph
    return main_wf
