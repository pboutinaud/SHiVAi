"""
Main workflow generator, with conditional piping (wf shape depends on the prediction types)
"""
from shivai.utils.misc import get_aquisitions_mapping, get_first_item, set_wf_shapers, get_img_acquisitions  # , as_list
from shivai.workflows.post_processing import genWorkflow as genWorkflowPost
from shivai.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivai.workflows.dual_preprocessing import graft_img2_preproc
from shivai.workflows.preprocessing_swi_reg import graft_workflow_swi
from shivai.workflows.swomed_graft_infiles import graft_swomed_infiles
from shivai.workflows.preprocessing_shiva_masking import genWorkflow as genWorkflow_preproc_shiva_mask
from shivai.workflows.preprocessing_premasked import genWorkflow as genWorkflow_preproc_masked
from shivai.workflows.preprocessing_synthseg import genWorkflow as genWorkflow_preproc_synthseg
from shivai.workflows.preprocessing_synthseg_precomp import genWorkflow as genWorkflow_preproc_synthseg_precomp
from shivai.workflows.preprocessing_swomed_pre_synthseg import genWorkflow as genWorkflow_preproc_synthseg_swomed
from shivai.workflows.preprocessing_custom_seg import genWorkflow as genWorkflow_preproc_custom_seg
from shivai.workflows.preprocessing_fs_precomp import genWorkflow as genWorkflow_preproc_fs
from shivai.workflows.predict_wf import genWorkflow as genWorkflow_prediction
from shivai.workflows.dcm2nii_grafting import graft_dcm2nii
from shivai.interfaces.post import Join_Prediction_metrics, Join_QC_metrics
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber
from shivai.interfaces.datasink import DataSink_CSV_and_PDF_safe
import os


def update_wf_grabber(wf, acquisitions, datatype, kwargs, grabber_name='datagrabber', datadir=''):
    """
    Updates (mutate) the workflow datagrabber to work with the different types on input
        wf: workflow with the datagrabber
        acquisitions example: [('img1', 't1'), ('img2', 'flair')]
        datatype ('nifti' or 'dicom')
        custom_seg (bool): wether there is a custom segmentation (brain mask or brain parc) available
        grabber_name (str): If the datagrabber node has a different name than 'datagrabber', specify it here.
            Typically used in generate_main_wf_grab_preproc.
        datadir (str): If the datagrabber node has a different data directory than kwargs['DATA_DIR'], specify it here.
            Typically used in generate_main_wf_grab_preproc.
    """
    files = '' if datatype == 'dicom' else '*.nii*'  # no files for dcm, just the whole folder
    datagrabber = wf.get_node(grabber_name)
    if not datagrabber.inputs.field_template:
        datagrabber.inputs.field_template = {}
    if not datagrabber.inputs.template_args:  # Should not happend, as default is {outfield: [[infield]]} (e.g. {'img1': [['subject_id']]})
        datagrabber.inputs.template_args = {}
    data_struct = kwargs['PREP_SETTINGS']['input_type']
    if data_struct in ['standard', 'json']:
        # e.g: {'img1': '%s/t1/%s_T1_raw.nii.gz'}
        datagrabber.inputs.field_template.update({acq[0]: os.path.join(datadir, f'%s/{acq[1]}/{files}') for acq in acquisitions})
        datagrabber.inputs.template_args.update({acq[0]: [['subject_id']] for acq in acquisitions})
        if kwargs['BRAIN_SEG'] == 'custom':
            datagrabber.inputs.field_template['seg'] = os.path.join(datadir, '%s/seg/*.nii*')  # We expect a nifti here, as dicom is unlikely
            datagrabber.inputs.template_args['seg'] = [['subject_id']]
        if kwargs['BRAIN_SEG'] == 'fs_precomp':
            datagrabber.inputs.field_template['seg'] = os.path.join(datadir, '%s/seg/aparc+aseg.*')  # We expect nii or mgz here
            datagrabber.inputs.template_args['seg'] = [['subject_id']]

    if data_struct == 'BIDS':
        if datatype == 'dicom':
            raise ValueError('BIDS data structure not compatible with DICOM input')
        # e.g: {'img1': '%s/anat/%s_T1_raw.nii.gz}
        datagrabber.inputs.field_template.update({acq[0]: os.path.join(datadir, f'%s/anat/%s_{acq[1].upper()}*.nii*') for acq in acquisitions})
        datagrabber.inputs.template_args.update({acq[0]: [['subject_id', 'subject_id']] for acq in acquisitions})
        if kwargs['BRAIN_SEG'] == 'custom':  # TODO: Correct this for proper bids format. It should actually be in the "derived" folder...
            datagrabber.inputs.field_template['seg'] = os.path.join(datadir, '%s/anat/%s_*seg*.nii*')
            datagrabber.inputs.template_args['seg'] = [['subject_id', 'subject_id']]
        if kwargs['BRAIN_SEG'] == 'fs_precomp':
            datagrabber.inputs.field_template['seg'] = os.path.join(datadir, '%s/anat/*aparc+aseg.*')
            datagrabber.inputs.template_args['seg'] = [['subject_id']]

    if data_struct == 'swomed':
        in_files_dict = kwargs['PREP_SETTINGS']['swomed_input']
        for imgN, acq in acquisitions:
            setattr(datagrabber.inputs, imgN, in_files_dict[acq])
        if 'seg' in in_files_dict:
            datagrabber.inputs.seg = in_files_dict['seg']
        if 'synthseg_vol' in in_files_dict:
            datagrabber.inputs.synthseg_vol = in_files_dict['synthseg_vol']
        if 'synthseg_qc' in in_files_dict:
            datagrabber.inputs.synthseg_qc = in_files_dict['synthseg_qc']

    if datatype == 'dicom':
        graft_dcm2nii(wf, **kwargs)

    # Clean up the datagrabber inputs to remove any empty fields (e.g. if no flair is provided)
    # Specifically removes template_args default keys that are not in field_template
    nok_fields = datagrabber.inputs.field_template.keys() ^ datagrabber.inputs.template_args.keys()
    for field in nok_fields:
        if field in datagrabber.inputs.field_template:
            del datagrabber.inputs.field_template[field]
        if field in datagrabber.inputs.template_args:
            del datagrabber.inputs.template_args[field]


def res_to_dict(sub_ids, in_files):
    dict_files = {s: f for s, f in zip(sub_ids, in_files)}
    return dict_files


def dict_to_res(sub_id, files_dict):
    return files_dict[sub_id]


# %% Helper functions for shared prediction/postproc/sink logic
# The "preproc_images" dict maps logical "image" (i.e. images being piped through the workflow) names to (node, output_field) tuples.
# Expected keys (all optional except 'brain_mask'):
#   'brain_mask', 't1', 'flair', 'swi',
#   'brain_seg', 'swi-to-t1', 'brain_mask_swi',
#   'swi_img_ref', 'swi_fov_mask'  (only for live preproc with SWI+T1)
#   'flair-to-t1', 't1-native', 'flair-native', 'swi-native'
# Optional QC keys (only from live preprocessing):
#   'crop_brain_img', 'overlayed_brainmask_1', 'overlayed_brainmask_2', 'isocontour_slides_FLAIR_T1'

def _build_preproc_joiners(main_wf, subject_iterator, preproc_images, with_t1, with_flair, with_swi):
    """Build JoinNodes to aggregate per-subject preprocessing outputs into dicts for the prediction workflow.
    Returns a dict of joiners keyed by 'mask', 't1', 'flair', 'swi'."""
    joiners = {}

    mask_node, mask_field = preproc_images['brain_mask']
    preproc_joiner_mask = JoinNode(Function(input_names=['sub_ids', 'in_files'],
                                            output_names=['files_dict'],
                                            function=res_to_dict),
                                   joinsource=subject_iterator,
                                   joinfield=['sub_ids', 'in_files'],
                                   name='preproc_joiner_mask')
    main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_mask, 'sub_ids')
    main_wf.connect(mask_node, mask_field, preproc_joiner_mask, 'in_files')
    joiners['mask'] = preproc_joiner_mask

    if with_t1 and 't1' in preproc_images:
        t1_node, t1_field = preproc_images['t1']
        preproc_joiner_t1 = JoinNode(Function(input_names=['sub_ids', 'in_files'],
                                              output_names=['files_dict'],
                                              function=res_to_dict),
                                     joinsource=subject_iterator,
                                     joinfield=['sub_ids', 'in_files'],
                                     name='preproc_joiner_t1')
        main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_t1, 'sub_ids')
        main_wf.connect(t1_node, t1_field, preproc_joiner_t1, 'in_files')
        joiners['t1'] = preproc_joiner_t1

    if with_flair and 'flair' in preproc_images:
        flair_node, flair_field = preproc_images['flair']
        preproc_joiner_flair = JoinNode(Function(input_names=['sub_ids', 'in_files'],
                                                 output_names=['files_dict'],
                                                 function=res_to_dict),
                                        joinsource=subject_iterator,
                                        joinfield=['sub_ids', 'in_files'],
                                        name='preproc_joiner_flair')
        main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_flair, 'sub_ids')
        main_wf.connect(flair_node, flair_field, preproc_joiner_flair, 'in_files')
        joiners['flair'] = preproc_joiner_flair

    if with_swi and 'swi' in preproc_images:
        swi_node, swi_field = preproc_images['swi']
        preproc_joiner_swi = JoinNode(Function(input_names=['sub_ids', 'in_files'],
                                               output_names=['files_dict'],
                                               function=res_to_dict),
                                      joinsource=subject_iterator,
                                      joinfield=['sub_ids', 'in_files'],
                                      name='preproc_joiner_swi')
        main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_swi, 'sub_ids')
        main_wf.connect(swi_node, swi_field, preproc_joiner_swi, 'in_files')
        joiners['swi'] = preproc_joiner_swi

    return joiners


def _connect_prediction(main_wf, subject_iterator, joiners, segmentation_wf, **kwargs):
    """Connect the prediction workflow and postprocessing workflow using the preproc image map.
    Returns seg_getters dict."""
    seg_getters = {}
    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers({'PREDICTION': [pred]})
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()

        # Connection with prediction inputs
        # main_wf.connect(joiners['mask'], 'files_dict', segmentation_wf, f'predict_{lpred}.brainmask_files')
        if pred_with_t1:
            main_wf.connect(joiners['t1'], 'files_dict', segmentation_wf, f'predict_{lpred}.primary_image_file')
            if pred_with_flair:
                main_wf.connect(joiners['flair'], 'files_dict', segmentation_wf, f'predict_{lpred}.second_image_file')
        if pred_with_swi:
            main_wf.connect(joiners['swi'], 'files_dict', segmentation_wf, f'predict_{lpred}.primary_image_file')

        # Seg getter node
        seg_getters[pred] = Node(Function(input_names=['sub_id', 'files_dict'],
                                          output_names=['segmentation'],
                                          function=dict_to_res),
                                 name=f'seg_getter_{lpred}')
        main_wf.connect(segmentation_wf, f'predict_{lpred}.segmentations', seg_getters[pred], 'files_dict')
        main_wf.connect(subject_iterator, 'subject_id', seg_getters[pred], 'sub_id')
    return seg_getters


def _connect_postproc(main_wf, seg_getters, subject_iterator, wf_post, preproc_images, with_t1, with_flair, with_swi, **kwargs):
    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers({'PREDICTION': [pred]})
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()

        main_wf.connect(seg_getters[pred], 'segmentation', wf_post, f'{lpred}_overlay_node.brainmask')
        main_wf.connect(seg_getters[pred], 'segmentation', wf_post, f'cluster_labelling_{lpred}.biomarker_raw')

        # Overlay node connections: img_ref and fov_mask
        if pred_with_swi:
            if 'swi_img_ref' in preproc_images:
                # Live preproc with SWI+T1: use specific SWI preprocessing outputs
                ref_node, ref_field = preproc_images['swi_img_ref']
                main_wf.connect(ref_node, ref_field, wf_post, f'{lpred}_overlay_node.img_ref')
                fov_node, fov_field = preproc_images['swi_fov_mask']
                main_wf.connect(fov_node, fov_field, wf_post, f'{lpred}_overlay_node.fov_mask')
            else:
                # Grab preproc or SWI-only: use the swi and brain_mask images
                swi_node, swi_field = preproc_images['swi']
                main_wf.connect(swi_node, swi_field, wf_post, f'{lpred}_overlay_node.img_ref')
                mask_node, mask_field = preproc_images['brain_mask']
                main_wf.connect(mask_node, mask_field, wf_post, f'{lpred}_overlay_node.fov_mask')
        else:
            mask_node, mask_field = preproc_images['brain_mask']
            main_wf.connect(mask_node, mask_field, wf_post, f'{lpred}_overlay_node.fov_mask')
            if pred in ['WMH', 'LAC']:
                flair_node, flair_field = preproc_images['flair']
                main_wf.connect(flair_node, flair_field, wf_post, f'{lpred}_overlay_node.img_ref')
            else:
                t1_node, t1_field = preproc_images['t1']
                main_wf.connect(t1_node, t1_field, wf_post, f'{lpred}_overlay_node.img_ref')

        if pred_with_swi and with_t1:
            if 'synthseg' in kwargs['BRAIN_SEG'] or 'precomp' in kwargs['BRAIN_SEG']:
                swi2t1_node, swi2t1_field = preproc_images['swi-to-t1']
                main_wf.connect(swi2t1_node, swi2t1_field, wf_post, 'seg_to_swi.transforms')
                main_wf.connect(seg_getters[pred], 'segmentation', wf_post, 'seg_to_swi.reference_image')
                seg_node, seg_field = preproc_images['brain_seg']
                main_wf.connect(seg_node, seg_field, wf_post, 'seg_to_swi.input_image')
            elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
                swi2t1_node, swi2t1_field = preproc_images['swi-to-t1']
                main_wf.connect(swi2t1_node, swi2t1_field, wf_post, 'seg_to_swi.transforms')
                main_wf.connect(seg_getters[pred], 'segmentation', wf_post, 'seg_to_swi.reference_image')
                seg_node, seg_field = preproc_images['brain_seg']
                main_wf.connect(seg_node, seg_field, wf_post, 'seg_to_swi.input_image')
            else:
                swi_mask_node, swi_mask_field = preproc_images['brain_mask_swi']
                main_wf.connect(swi_mask_node, swi_mask_field, wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(swi_mask_node, swi_mask_field, wf_post, 'prediction_metrics_cmb.brain_seg')
            ref_node_swi, ref_field_swi = preproc_images['swi-native']
            ref_node_t1, ref_field_t1 = preproc_images['t1-native']
            ref_node_trans, ref_field_trans = preproc_images['swi-to-t1']
            main_wf.connect(ref_node_swi, ref_field_swi, wf_post, 'cmb_to_native.target_image')
            main_wf.connect(ref_node_t1, ref_field_t1, wf_post, 'cmb_to_native_t1.target_image')
            main_wf.connect(ref_node_trans, (ref_field_trans, get_first_item), wf_post, 'cmb_to_native_t1.transform_affine')
        else:
            if 'synthseg' in kwargs['BRAIN_SEG'] or kwargs['BRAIN_SEG'] == 'fs_precomp':
                seg_node, seg_field = preproc_images['brain_seg']
                main_wf.connect(seg_node, seg_field, wf_post, f'custom_{lpred}_parc.brain_seg')
            elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
                mask_node, mask_field = preproc_images['brain_mask']
                main_wf.connect(mask_node, mask_field, wf_post, f'cluster_labelling_{lpred}.brain_seg')
                seg_node, seg_field = preproc_images['brain_seg']
                main_wf.connect(seg_node, seg_field, wf_post, f'prediction_metrics_{lpred}.brain_seg')
            else:
                mask_node, mask_field = preproc_images['brain_mask']
                main_wf.connect(mask_node, mask_field, wf_post, f'cluster_labelling_{lpred}.brain_seg')
                main_wf.connect(mask_node, mask_field, wf_post, f'prediction_metrics_{lpred}.brain_seg')
            ref_node_mig1, ref_field_img1 = preproc_images['t1-native'] if with_t1 else preproc_images['swi-native']
            main_wf.connect(ref_node_mig1, ref_field_img1, wf_post, f'{lpred}_to_native.target_image')
            if with_flair and not kwargs['PREP_SETTINGS']['prereg_flair']:
                ref_node_flair, ref_field_flair = preproc_images['flair-native']
                ref_node_trans, ref_field_trans = preproc_images['flair-to-t1']
                main_wf.connect(ref_node_flair, ref_field_flair, wf_post, f'{lpred}_to_flair-native.target_image')
                main_wf.connect(ref_node_trans, (ref_field_trans, get_first_item), wf_post, f'{lpred}_to_flair-native.transform_affine')

        # Merge all csv files
        prediction_metrics_all = JoinNode(Join_Prediction_metrics(),
                                          joinsource=subject_iterator,
                                          joinfield=['csv_files', 'subject_id'],
                                          name=f'prediction_metrics_{lpred}_all')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv',
                        prediction_metrics_all, 'csv_files')
        main_wf.connect(subject_iterator, 'subject_id',
                        prediction_metrics_all, 'subject_id')


def _connect_pred_sinks(main_wf, seg_getters, wf_post, sink_node_subjects, sink_node_all, **kwargs):
    """Connect prediction/postprocessing outputs to data sink nodes."""
    with_t1, with_flair, _ = set_wf_shapers(kwargs)

    main_wf.connect(wf_post, 'summary_report.pdf_report', sink_node_subjects, 'report')

    t1_acq, flair_acq, swi_acq = get_img_acquisitions(kwargs)

    for pred in kwargs['PREDICTION']:
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers({'PREDICTION': [pred]})
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()
        if pred_with_swi:
            space = f'_{swi_acq}-space'
        else:
            space = ''

        prediction_metrics_all = main_wf.get_node(f'prediction_metrics_{lpred}_all')
        main_wf.connect(seg_getters[pred], 'segmentation', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}')
        main_wf.connect(wf_post, f'cluster_labelling_{lpred}.labelled_biomarkers', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@labelled')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_stats_wide_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@metrics_wide')
        main_wf.connect(wf_post, f'prediction_metrics_{lpred}.biomarker_census_csv', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@census')
        if pred == 'CMB' and with_t1:
            main_wf.connect(wf_post, 'cmb_to_native.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@native')
            main_wf.connect(wf_post, 'cmb_to_native_t1.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@native_t1')
        else:
            main_wf.connect(wf_post, f'{lpred}_to_native.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@native')
            if with_flair and not kwargs['PREP_SETTINGS']['prereg_flair']:
                main_wf.connect(wf_post, f'{lpred}_to_flair-native.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@native_flair')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}')
        main_wf.connect(prediction_metrics_all, 'prediction_metrics_wide_csv', sink_node_all, f'segmentations.{lpred}_metrics{space}.@wide')
        if 'synthseg' in kwargs['BRAIN_SEG'] or kwargs['BRAIN_SEG'] == 'fs_precomp':
            main_wf.connect(wf_post, f'custom_{lpred}_parc.brain_seg', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc')
            main_wf.connect(wf_post, f'custom_{lpred}_parc.region_dict_json', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@parc_dict')

        if pred_with_swi and with_t1:
            if 'synthseg' in kwargs['BRAIN_SEG'] or kwargs['BRAIN_SEG'] == 'fs_precomp':
                main_wf.connect(wf_post, 'seg_to_swi.output_image', sink_node_subjects, f'segmentations.{lpred}_segmentation{space}.@custom_parc')


def generate_main_wf(**kwargs) -> Workflow:
    """
    Generate a full processing workflow, with prepoc, pred, and postproc.
    """
    # %% Initializing the general data
    # Set the booleans to shape the main workflow
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs)

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

    # %% Preprocessing
    # Initialise the proper preproc depending on the input images and the type of preproc, and update its datagrabber
    acquisitions = []
    file_type = kwargs['PREP_SETTINGS']['file_type']

    if with_t1:
        # What type of preprocessing (basic / synthseg / premasked / custom input)
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
        elif kwargs['BRAIN_SEG'] == 'fs_precomp':
            wf_preproc = genWorkflow_preproc_fs(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
            wf_preproc = genWorkflow_preproc_custom_seg(**kwargs, wf_name=wf_name)
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is None:
            wf_preproc = genWorkflowPreproc(**kwargs, wf_name=wf_name)
        else:
            raise NotImplementedError(f'The brain segmentation type "{kwargs["BRAIN_SEG"]}" was not recognized')

        # Checking if dual preprocessing is needed (and chich type of secondary aquisition)
        if with_flair:
            graft_img2_preproc(wf_preproc, **kwargs)

        # Checking if SWI (or equivalent) need to be preprocessed
        if with_swi:  # Adding the swi preprocessing steps to the preproc workflow
            graft_workflow_swi(wf_preproc, **kwargs)

    elif with_swi and not with_t1:  # CMB alone
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
        graft_swomed_infiles(wf_preproc)

    acquisitions = get_aquisitions_mapping(kwargs)
    # Updating the datagrabber with all this info
    update_wf_grabber(wf_preproc, acquisitions, file_type, kwargs)

    # datagrabber - iterator connection
    main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'datagrabber.subject_id')
    if kwargs['BRAIN_SEG'] == 'synthseg_precomp':
        main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'synthseg_grabber.subject_id')

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

    if not kwargs['PREP_SETTINGS']['preproc_only']:
        # %% Build the preproc image map from live preprocessing outputs
        preproc_images = {
            'brain_mask': (wf_preproc, 'mask_to_crop.resampled_image'),
        }
        if with_t1:
            preproc_images['t1'] = (wf_preproc, 'img1_final_intensity_normalization.intensity_normalized')
            preproc_images['t1-native'] = (wf_preproc, 'correct_affine_img1.corrected_img')
        if with_flair:
            preproc_images['flair'] = (wf_preproc, 'img2_final_intensity_normalization.intensity_normalized')
            if not kwargs['PREP_SETTINGS']['prereg_flair']:
                preproc_images['flair-native'] = (wf_preproc, 'correct_affine_flair.corrected_img')
                preproc_images['flair-to-t1'] = (wf_preproc, 'flair_to_t1.forward_transforms')
        if with_swi:
            if with_t1:
                preproc_images['swi-native'] = (wf_preproc, 'cmb_preprocessing.correct_affine_swi.corrected_img')
                preproc_images['swi-to-t1'] = (wf_preproc, 'cmb_preprocessing.swi_to_t1.forward_transforms')
                preproc_images['swi'] = (wf_preproc, 'cmb_preprocessing.swi_intensity_normalisation.intensity_normalized')
                # SWI+T1 specific overlay references
                preproc_images['swi_img_ref'] = (wf_preproc, 'cmb_preprocessing.swi_intensity_normalisation.intensity_normalized')
                preproc_images['swi_fov_mask'] = (wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image')
                if 'synthseg' not in kwargs['BRAIN_SEG'] and kwargs['BRAIN_SEG'] != 'fs_precomp' and kwargs['BRAIN_SEG'] != 'custom':
                    preproc_images['brain_mask_swi'] = (wf_preproc, 'cmb_preprocessing.mask_to_crop_swi.resampled_image')

            else:
                preproc_images['swi'] = (wf_preproc, 'img1_final_intensity_normalization.intensity_normalized')
                preproc_images['swi-native'] = (wf_preproc, 'correct_affine_img1.corrected_img')

        # Brain seg image (depends on BRAIN_SEG type)
        if 'synthseg' in kwargs['BRAIN_SEG'] or kwargs['BRAIN_SEG'] == 'fs_precomp':
            preproc_images['brain_seg'] = (wf_preproc, 'custom_parc.brain_parc')
        elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
            preproc_images['brain_seg'] = (wf_preproc, 'seg_to_crop.resampled_image')

        # QC images (only from live preprocessing)
        preproc_images['crop_brain_img'] = (wf_preproc, 'preproc_qc_workflow.qc_crop_box.crop_brain_img')
        preproc_images['overlayed_brainmask_1'] = (wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask.overlayed_brainmask')
        if with_swi and with_t1:
            preproc_images['overlayed_brainmask_2'] = (wf_preproc, 'preproc_qc_workflow.qc_overlay_brainmask_swi.overlayed_brainmask')
        if with_flair and not kwargs['PREP_SETTINGS']['prereg_flair']:
            preproc_images['isocontour_slides_FLAIR_T1'] = (wf_preproc, 'preproc_qc_workflow.qc_coreg_FLAIR_T1.qc_coreg')

        # Build joiners, postproc, prediction, and connect everything
        joiners = _build_preproc_joiners(main_wf, subject_iterator, preproc_images, with_t1, with_flair, with_swi)

        wf_post = genWorkflowPost(**kwargs)

        # Connect QC images to summary report
        main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
        mask_node, mask_field = preproc_images['brain_mask']
        main_wf.connect(mask_node, mask_field, wf_post, 'summary_report.brainmask')
        if 'crop_brain_img' in preproc_images:
            qc_node, qc_field = preproc_images['crop_brain_img']
            main_wf.connect(qc_node, qc_field, wf_post, 'summary_report.crop_brain_img')
        if 'overlayed_brainmask_1' in preproc_images:
            qc_node, qc_field = preproc_images['overlayed_brainmask_1']
            main_wf.connect(qc_node, qc_field, wf_post, 'summary_report.overlayed_brainmask_1')
        if 'overlayed_brainmask_2' in preproc_images:
            qc_node, qc_field = preproc_images['overlayed_brainmask_2']
            main_wf.connect(qc_node, qc_field, wf_post, 'summary_report.overlayed_brainmask_2')
        if 'isocontour_slides_FLAIR_T1' in preproc_images:
            qc_node, qc_field = preproc_images['isocontour_slides_FLAIR_T1']
            main_wf.connect(qc_node, qc_field, wf_post, 'summary_report.isocontour_slides_FLAIR_T1')

    # %% Then prediction workflow and all its connections
        segmentation_wf = genWorkflow_prediction(**kwargs)
        seg_getters = _connect_prediction(main_wf, subject_iterator, joiners, segmentation_wf, **kwargs)
        _connect_postproc(main_wf, seg_getters, subject_iterator, wf_post, preproc_images, with_t1, with_flair, with_swi, **kwargs)

    # The workflow graph
    wf_graph = None
    datasing_fields = []
    if kwargs['SAVE_GRAPH']:
        wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')
        datasing_fields.append('wf_graph')

    # %% Finally the data sinks
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
    sink_node_all = Node(DataSink_CSV_and_PDF_safe(
        infields=datasing_fields), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    # Connecting the preproc sinks
    if with_t1:
        img1 = 't1'
    elif with_swi and not with_t1:
        img1 = 'swi'
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized', sink_node_subjects, f'shiva_preproc.{img1}_preproc')
    main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@brain_mask')
    if file_type == 'dicom':
        main_wf.connect(wf_preproc, 'dicom2nifti_img1.converted_files', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@converted')
        main_wf.connect(wf_preproc, 'dicom2nifti_img1.bids', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@converted_bids')
    if 'synthseg' in kwargs['BRAIN_SEG']:
        main_wf.connect(wf_preproc, 'seg_cleaning.ouput_seg', sink_node_subjects, 'shiva_preproc.synthseg')
        main_wf.connect(wf_preproc, 'seg_cleaning.sunk_islands', sink_node_subjects, 'shiva_preproc.synthseg.@removed')
        main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, 'shiva_preproc.synthseg.@cropped')
        main_wf.connect(wf_preproc, 'custom_parc.brain_parc', sink_node_subjects, 'shiva_preproc.synthseg.@custom')
        if kwargs['PREP_SETTINGS']['input_type'] == 'swomed':
            main_wf.connect(wf_preproc, 'datagrabber.synthseg_vol', sink_node_subjects, 'shiva_preproc.synthseg.@vol')
            main_wf.connect(wf_preproc, 'datagrabber.synthseg_qc', sink_node_subjects, 'shiva_preproc.synthseg.@qc')
        else:
            if kwargs['BRAIN_SEG'] == 'synthseg_precomp':
                if kwargs['PREP_SETTINGS']['ss_vol']:
                    main_wf.connect(wf_preproc, 'synthseg_grabber.volumes', sink_node_subjects, 'shiva_preproc.synthseg.@vol')
                if kwargs['PREP_SETTINGS']['ss_qc']:
                    main_wf.connect(wf_preproc, 'synthseg_grabber.qc', sink_node_subjects, 'shiva_preproc.synthseg.@qc')
            else:
                if kwargs['PREP_SETTINGS']['ss_vol']:
                    main_wf.connect(wf_preproc, 'synthseg.volumes', sink_node_subjects, 'shiva_preproc.synthseg.@vol')
                if kwargs['PREP_SETTINGS']['ss_qc']:
                    main_wf.connect(wf_preproc, 'synthseg.qc', sink_node_subjects, 'shiva_preproc.synthseg.@qc')
    elif kwargs['BRAIN_SEG'] == 'fs_precomp':
        main_wf.connect(wf_preproc, 'seg_cleaning.ouput_seg', sink_node_subjects, 'shiva_preproc.freesurfer')
        main_wf.connect(wf_preproc, 'seg_cleaning.sunk_islands', sink_node_subjects, 'shiva_preproc.freesurfer.@removed')
        main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', sink_node_subjects, 'shiva_preproc.freesurfer.@cropped')
        main_wf.connect(wf_preproc, 'custom_parc.brain_parc', sink_node_subjects, 'shiva_preproc.freesurfer.@custom')
    elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
        main_wf.connect(wf_preproc, 'seg_to_crop.resampled_image', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@seg')
    main_wf.connect(wf_preproc, 'crop.bbox1_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb1')
    main_wf.connect(wf_preproc, 'crop.bbox2_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@bb2')
    main_wf.connect(wf_preproc, 'crop.cdg_ijk_file', sink_node_subjects, f'shiva_preproc.{img1}_preproc.@cdg')
    if with_flair:
        main_wf.connect(wf_preproc, 'img2_final_intensity_normalization.intensity_normalized', sink_node_subjects, 'shiva_preproc.flair_preproc')
        main_wf.connect(wf_preproc, 'flair_to_t1.forward_transforms', sink_node_subjects, 'shiva_preproc.flair_preproc.@reg_to_t1_transf')
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

    main_wf.connect(qc_joiner, 'qc_metrics_csv', sink_node_all, 'preproc_qc')
    main_wf.connect(qc_joiner, 'bad_qc_subs', sink_node_all, 'preproc_qc.@bad_qc_subs')
    main_wf.connect(qc_joiner, 'qc_plot_png', sink_node_all, 'preproc_qc.@qc_plot_png')
    if prev_qc is not None:
        main_wf.connect(qc_joiner, 'csv_pop_file', sink_node_all, 'preproc_qc.@preproc_qc_pop')
        main_wf.connect(qc_joiner, 'pop_bad_subjects_file', sink_node_all, 'preproc_qc.@pop_bad_subjects')
    if wf_graph is not None:
        sink_node_all.inputs.wf_graph = wf_graph

    if kwargs['PREP_SETTINGS']['preproc_only']:
        return main_wf  # ENDPOINT if just running the preprocessing

    # Pred and postproc sinks
    _connect_pred_sinks(main_wf, seg_getters, wf_post, sink_node_subjects, sink_node_all, **kwargs)
    return main_wf  # ENDPOINT with everything


def generate_main_wf_grab_preproc(**kwargs) -> Workflow:
    """
    Generate a full processing workflow, without prepoc as it will grab the preprocessed data from the results folder from
    a previous run
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
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs)

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
                   'swi-to-t1',  # for CMB when whith_t1 and brain parc
                   'brain_mask_swi',  # for CMB when whith_t1 without brain parc
                   'img1',
                   'img2',
                   'img3',
                   'seg',  # Not used for now, just for compatibily with update_wf_grabber
                   'flair-to-t1'
                   ]),
        name='preproc_grabber')
    preproc_grabber.inputs.base_directory = preproc_res
    preproc_grabber.inputs.template = '*/%s/*.nii.gz'  # unused placeholder (but required)
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
            field_template['flair-to-t1'] = 'flair_preproc/%s/*_0GenericAffine.mat'
            template_args['flair-to-t1'] = [['subject_id']]
    if with_swi:
        field_template['swi_intensity_normalized'] = 'swi_preproc/%s/*_cropped_intensity_normed.nii.gz'
        template_args['swi_intensity_normalized'] = [['subject_id']]
        if with_t1:
            field_template['swi-to-t1'] = 'swi_preproc/%s/*_0GenericAffine.mat'
            template_args['swi-to-t1'] = [['subject_id']]
            if kwargs['BRAIN_SEG'] in ['shiva', 'shiva_gpu', 'premasked']:
                field_template['brain_mask_swi'] = 'swi_preproc/%s/brainmask_cropped_swi-space.nii.gz'
                template_args['brain_mask_swi'] = [['subject_id']]
        else:
            field_template['brain_mask'] = 'swi_preproc/%s/brainmask_cropped*.nii.gz'
            template_args['brain_mask'] = [['subject_id']]

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

    acquisitions = get_aquisitions_mapping(kwargs)
    main_wf.connect(subject_iterator, 'subject_id', preproc_grabber, 'subject_id')
    file_type = kwargs['PREP_SETTINGS']['file_type']
    if file_type == 'dicom':
        raise NotImplementedError('Grabbing preprocessed data from DICOM files is not currently implemented re-using preprocessed images')
    update_wf_grabber(main_wf, acquisitions, file_type, kwargs, grabber_name='preproc_grabber', datadir=kwargs['DATA_DIR'])

    # Build the preproc image map from the preproc_grabber
    preproc_images = {
        'brain_mask': (preproc_grabber, 'brain_mask'),
    }

    if with_t1:
        preproc_images['t1'] = (preproc_grabber, 't1_intensity_normalized')
        preproc_images['t1-native'] = (preproc_grabber, 'img1')
    if with_flair:
        preproc_images['flair'] = (preproc_grabber, 'flair_intensity_normalized')
        preproc_images['flair-native'] = (preproc_grabber, 'img2')
        preproc_images['flair-to-t1'] = (preproc_grabber, 'flair-to-t1')
    if with_swi:
        if with_t1:
            preproc_images['swi-native'] = (preproc_grabber, 'img3')
            preproc_images['swi-to-t1'] = (preproc_grabber, 'swi-to-t1')
            if 'synthseg' not in kwargs['BRAIN_SEG'] and kwargs['BRAIN_SEG'] != 'custom':
                preproc_images['brain_mask_swi'] = (preproc_grabber, 'brain_mask_swi')
        else:
            preproc_images['swi-native'] = (preproc_grabber, 'img1')
        preproc_images['swi'] = (preproc_grabber, 'swi_intensity_normalized')

    # Brain seg image
    if 'synthseg' in kwargs['BRAIN_SEG'] or (kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None):
        preproc_images['brain_seg'] = (preproc_grabber, 'brain_seg')

    # Build joiners, postproc, prediction, and connect everything
    joiners = _build_preproc_joiners(main_wf, subject_iterator, preproc_images, with_t1, with_flair, with_swi)

    wf_post = genWorkflowPost(**kwargs)

    # Connect basic images to summary report (no QC images in grab_preproc mode)
    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(preproc_grabber, 'brain_mask', wf_post, 'summary_report.brainmask')

    segmentation_wf = genWorkflow_prediction(**kwargs)

    seg_getters = _connect_prediction(main_wf, subject_iterator, joiners, segmentation_wf, **kwargs)
    _connect_postproc(main_wf, seg_getters, subject_iterator, wf_post, preproc_images, with_t1, with_flair, with_swi, **kwargs)

    # The workflow graph
    wf_graph = None
    datasing_fields = []
    if kwargs['SAVE_GRAPH']:
        wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')
        datasing_fields.append('wf_graph')

    # Data sinks
    sink_node_subjects = Node(DataSink_CSV_and_PDF_safe(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('_resampled_defaced_cropped_img_normalized', '_defaced_cropped_intensity_normed'),
        ('flair_to_t1__Warped_defaced_img_normalized', 'flair_to_t1_defaced_cropped_intensity_normed')
    ]
    sink_node_all = Node(DataSink_CSV_and_PDF_safe(
        infields=datasing_fields), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    _connect_pred_sinks(main_wf, seg_getters, wf_post, sink_node_subjects, sink_node_all, **kwargs)

    if wf_graph is not None:
        sink_node_all.inputs.wf_graph = wf_graph
    return main_wf


def generate_main_wf_rerun_postproc(**kwargs) -> Workflow:
    """
    Generate a postprocessing-only workflow, grabbing both preprocessed data and
    prediction segmentations from previous results folders.
    No preprocessing or prediction is run — only postprocessing (clustering,
    metrics, report, sinks).
    """
    # %% Get the folders with the preprocessed and prediction data
    prev_res = kwargs['PREP_SETTINGS']['prev_res']

    # %% Validate available subjects
    available_seg = [seg[:seg.index('_segmentation')].upper() for seg in os.listdir(os.path.join(prev_res, 'segmentations'))]
    available_subs = os.listdir(os.path.join(prev_res, 'shiva_preproc', 'qc_metrics'))
    missing_subj = sorted(list(set(kwargs['SUBJECT_LIST']) - set(available_subs)))
    if missing_subj:
        error_msg = (
            '\nSome of the datasets from the input folder did not have corresponding preprocessed data in the input results folder '
            f'({ " ".join(missing_subj) }). \n'
            'To continue, you can:\n'
            '   - Process the datasets which lack preprocessed data the normal way with the full shiva pipeline (to complete the preprocessed folder)\n'
            '   - Remove the datasets from the subject list if you are providing one with --sub_list\n'
            '   - Add the datasets to the exlusion list with --exclusion_list (if you are not using --sub_list). You can even provide the text file generated here as is.\n'
        )
        raise ValueError(error_msg)

    missing_seg = sorted(list(set(['PVS' if p == 'PVS2' else p for p in kwargs['PREDICTION']]) - set(available_seg)))
    if missing_seg:
        error_msg = (
            '\nSome of the requested prediction segmentations were not found in the input results folder '
            f'({ " ".join(missing_seg) }). \n'
            'To continue, you can:\n'
            '   - Process the datasets which lack prediction segmentations the normal way with the full shiva pipeline (to complete the results folder)\n'
            '   - Remove the missing predictions from the list of requested predictions with --predictions\n'
        )
        raise ValueError(error_msg)
    # %% Setup
    with_t1, with_flair, with_swi = set_wf_shapers(kwargs)
    t1_acq, flair_acq, swi_acq = get_img_acquisitions(kwargs)

    main_wf = Workflow('main_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # %% Preproc grabber (same as generate_main_wf_grab_preproc)
    prev_res_grabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1',
                   'img2',
                   'img3',
                   't1_intensity_normalized',
                   'flair_intensity_normalized',
                   'swi_intensity_normalized',
                   'brain_mask',
                   'brain_seg',
                   'swi-to-t1',
                   'flair-to-t1',
                   'brain_mask_swi',
                   'pvs_segmentation',
                   'wmh_segmentation',
                   'cmb_segmentation',
                   'lac_segmentation',
                   ]),
        name='prev_res_grabber')
    prev_res_grabber.inputs.base_directory = prev_res
    prev_res_grabber.inputs.template = '*/%s/*.nii.gz'
    prev_res_grabber.inputs.raise_on_empty = True
    prev_res_grabber.inputs.sort_filelist = True

    field_template = {}
    if with_t1:
        field_template['t1_intensity_normalized'] = 'shiva_preproc/t1_preproc/%s/*_cropped_intensity_normed.nii.gz'
        field_template['brain_mask'] = 'shiva_preproc/t1_preproc/%s/brainmask_cropped.nii.gz'
        if with_flair:
            field_template['flair_intensity_normalized'] = 'shiva_preproc/flair_preproc/%s/*_cropped_intensity_normed.nii.gz'
            field_template['flair-to-t1'] = 'shiva_preproc/flair_preproc/%s/*_0GenericAffine.mat'
    if with_swi:
        field_template['swi_intensity_normalized'] = 'shiva_preproc/swi_preproc/%s/*_cropped_intensity_normed.nii.gz'
        if not with_t1:
            field_template['brain_mask'] = 'shiva_preproc/swi_preproc/%s/brainmask_cropped*.nii.gz'
        else:
            field_template['swi-to-t1'] = 'shiva_preproc/swi_preproc/%s/swi_to_t1_0GenericAffine.mat'
            if kwargs['BRAIN_SEG'] in ['shiva', 'shiva_gpu', 'premasked']:  # Just need the mask, no parcelation
                field_template['brain_mask_swi'] = f'shiva_preproc/swi_preproc/%s/brainmask_cropped_{swi_acq}-space.nii.gz'

    if 'synthseg' in kwargs['BRAIN_SEG']:
        field_template['brain_seg'] = 'shiva_preproc/synthseg/%s/derived_parc.nii.gz'
    elif kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None:
        if with_t1:
            field_template['brain_seg'] = 'shiva_preproc/t1_preproc/%s/custom_seg_cropped.nii.gz'
        elif with_swi and not with_t1:
            field_template['brain_seg'] = 'shiva_preproc/swi_preproc/%s/custom_seg_cropped.nii.gz'

    prev_res_grabber.inputs.field_template = field_template

    # %% Prediction segmentation grabber
    pred_outfields = []
    for pred in kwargs['PREDICTION']:
        if pred == 'PVS2':
            pred = 'PVS'
        pred_outfields.append(f'{pred.lower()}_segmentation')

    pred_field_template = {}
    seg_getters = {}
    for pred in kwargs['PREDICTION']:
        if pred == 'PVS2':
            pred = 'PVS'
        lpred = pred.lower()
        pred_with_t1, pred_with_flair, pred_with_swi = set_wf_shapers({'PREDICTION': [pred]})
        if pred_with_swi:
            space = f'_{swi_acq}-space'
        else:
            space = ''
        pred_field_template[f'{lpred}_segmentation'] = f'segmentations/{lpred}_segmentation{space}/%s_{lpred}_map.nii.gz'
        seg_getters[pred] = Node(IdentityInterface(fields=['segmentation']),
                                 name=f'seg_getter_{lpred}')
        main_wf.connect(prev_res_grabber, f'{lpred}_segmentation', seg_getters[pred], 'segmentation')

    prev_res_grabber.inputs.field_template.update(pred_field_template)

    main_wf.connect(subject_iterator, 'subject_id', prev_res_grabber, 'subject_id')

    file_type = kwargs['PREP_SETTINGS']['file_type']
    if file_type == 'dicom':
        raise NotImplementedError('Grabbing preprocessed data from DICOM files is not currently implemented re-using preprocessed images')
    acquisitions = get_aquisitions_mapping(kwargs)
    update_wf_grabber(main_wf, acquisitions, file_type, kwargs, grabber_name='prev_res_grabber', datadir=kwargs['DATA_DIR'])

    # %% Build the preproc image map from the prev_res_grabber
    preproc_images = {
        'brain_mask': (prev_res_grabber, 'brain_mask'),
    }
    if with_t1:
        preproc_images['t1'] = (prev_res_grabber, 't1_intensity_normalized')
    if with_flair:
        preproc_images['flair'] = (prev_res_grabber, 'flair_intensity_normalized')
    if with_swi:
        preproc_images['swi'] = (prev_res_grabber, 'swi_intensity_normalized')

    if with_t1:
        preproc_images['t1'] = (prev_res_grabber, 't1_intensity_normalized')
        preproc_images['t1-native'] = (prev_res_grabber, 'img1')
    if with_flair:
        preproc_images['flair'] = (prev_res_grabber, 'flair_intensity_normalized')
        preproc_images['flair-native'] = (prev_res_grabber, 'img2')
        preproc_images['flair-to-t1'] = (prev_res_grabber, 'flair-to-t1')
    if with_swi:
        if with_t1:
            preproc_images['swi-native'] = (prev_res_grabber, 'img3')
            preproc_images['swi-to-t1'] = (prev_res_grabber, 'swi-to-t1')
            if 'synthseg' not in kwargs['BRAIN_SEG'] and kwargs['BRAIN_SEG'] != 'fs_precomp' and kwargs['BRAIN_SEG'] != 'custom':
                preproc_images['brain_mask_swi'] = (prev_res_grabber, 'brain_mask_swi')
        else:
            preproc_images['swi-native'] = (prev_res_grabber, 'img1')
        preproc_images['swi'] = (prev_res_grabber, 'swi_intensity_normalized')

    if 'synthseg' in kwargs['BRAIN_SEG'] or (kwargs['BRAIN_SEG'] == 'custom' and kwargs['CUSTOM_LUT'] is not None):
        preproc_images['brain_seg'] = (prev_res_grabber, 'brain_seg')

    # %% Postprocessing workflow and connections
    wf_post = genWorkflowPost(**kwargs)

    main_wf.connect(subject_iterator, 'subject_id', wf_post, 'summary_report.subject_id')
    main_wf.connect(prev_res_grabber, 'brain_mask', wf_post, 'summary_report.brainmask')

    _connect_postproc(main_wf, seg_getters, subject_iterator, wf_post, preproc_images, with_t1, with_flair, with_swi, **kwargs)

    # %% Workflow graph
    wf_graph = None
    datasing_fields = []
    if kwargs['SAVE_GRAPH']:
        wf_graph = main_wf.write_graph(graph2use='colored', dotfilename='graph.svg', format='svg')
        datasing_fields.append('wf_graph')

    # %% Data sinks
    sink_node_subjects = Node(DataSink_CSV_and_PDF_safe(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
        ('_resampled_cropped_img_normalized', '_cropped_intensity_normed'),
        ('_resampled_defaced_cropped_img_normalized', '_defaced_cropped_intensity_normed'),
        ('flair_to_t1__Warped_defaced_img_normalized', 'flair_to_t1_defaced_cropped_intensity_normed')
    ]
    sink_node_all = Node(DataSink_CSV_and_PDF_safe(
        infields=datasing_fields), name='sink_node_all')
    sink_node_all.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node_all.inputs.container = 'results_summary'

    _connect_pred_sinks(main_wf, seg_getters, wf_post, sink_node_subjects, sink_node_all, **kwargs)

    if wf_graph is not None:
        sink_node_all.inputs.wf_graph = wf_graph
    return main_wf
