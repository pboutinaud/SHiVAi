"""
Functions needed by the shiva.py script to run the pipeline
"""

from shivai.workflows.main_workflow import generate_main_wf, generate_main_wf_grab_preproc
from nipype import config
import os
import shutil


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


def shiva(in_dir, out_dir, input_type, file_type, sub_list, prediction, model, brain_seg, synthseg_threads,
          node_plugin_args, prev_qc, preproc_results, replace_t1, replace_flair, replace_swi,
          db_name, custom_LUT, swomed_parc, swomed_ssvol, swomed_ssqc, swomed_t1, swomed_flair, swomed_swi, use_t1, container_image, synthseg_image, containerized_nodes,
          anonymize, interpolation, percentile, threshold, threshold_pvs, threshold_wmh, threshold_cmb,
          threshold_lac, min_pvs_size, min_wmh_size, min_cmb_size, min_lac_size, final_dimensions,
          voxels_size, keep_all, debug, remove_intermediates, run_plugin, run_plugin_args,
          brainmask_descriptor, wmh_descriptor, pvs_descriptor, pvs2_descriptor, cmb_descriptor, lac_descriptor,
          **kwargs):
    """
    Function that build and run the SHiVAi workflow using the input argument from the parser.

    All the input arguments should be contained in the the parser arguments. So For more details about them,
    check shivaParser.
    """
    # if input_type == 'json':  # TODO: Homogenize with the .yml file
    #     with open(in_dir, 'r') as json_in:
    #         subject_dict = json.load(json_in)

    #     out_dir = subject_dict['parameters']['out_dir']
    #     subject_directory = subject_dict["files_dir"]
    #     # subject_list = os.listdir(subject_directory)
    #     brainmask_descriptor = subject_dict['parameters']['brainmask_descriptor']
    #     if subject_dict['parameters']['WMH_descriptor']:
    #         wmh_descriptor = subject_dict['parameters']['WMH_descriptor']
    #     else:
    #         wmh_descriptor = None
    #     if subject_dict['parameters']['PVS_descriptor']:
    #         pvs_descriptor = subject_dict['parameters']['PVS_descriptor']
    #     else:
    #         pvs_descriptor = None
    #     if subject_dict['parameters']['CMB_descriptor']:
    #         cmb_descriptor = subject_dict['parameters']['CMB_descriptor']
    #     else:
    #         cmb_descriptor = None
    #     if subject_dict['parameters']['LAC_descriptor']:
    #         lac_descriptor = subject_dict['parameters']['LAC_descriptor']
    #     else:
    #         lac_descriptor = None

    if input_type in ['standard', 'BIDS', 'swomed']:
        subject_directory = in_dir
        descriptor_paths = {  # This dict will be injected in wfargs below
            'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
            'WMH_DESCRIPTOR': wmh_descriptor,
            'PVS_DESCRIPTOR': pvs_descriptor,
            'PVS2_DESCRIPTOR': pvs2_descriptor,
            'CMB_DESCRIPTOR': cmb_descriptor,
            'LAC_DESCRIPTOR': lac_descriptor
        }
        for desc_k, desc_path in descriptor_paths.items():
            if desc_path is not None and not os.path.exists(desc_path):
                # When it's a relative path from the "model" path (default behaviour)
                descriptor_paths[desc_k] = os.path.join(model, desc_path)

    ss_threads = 0
    if brain_seg == 'synthseg_cpu':
        ss_threads = synthseg_threads

    # Plugin arguments for predictions (shiva pred and synthseg)
    pred_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    reg_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    if 'pred' in node_plugin_args.keys():
        pred_plugin_args = node_plugin_args['pred']
    if 'reg' in node_plugin_args.keys():
        reg_plugin_args = node_plugin_args['reg']

    # Prepping the paths for SWOMed
    t1_name = 't1' if not replace_t1 else replace_t1
    flair_name = 'flair' if not replace_flair else replace_flair
    swi_name = 'swi' if not replace_swi else replace_swi
    pre_path_dict = {t1_name: swomed_t1, flair_name: swomed_flair, swi_name: swomed_swi,
                     'seg': swomed_parc, 'synthseg_vol': swomed_ssvol, 'synthseg_qc': swomed_ssqc}
    in_path_dict = {k: val for k, val in pre_path_dict.items() if val is not None}
    # in_path_dict['base_dir'] = os.path.commonprefix(list(pre_path_dict.values()))
    # in_path_dict = {k: val.removeprefix(in_path_dict['base_dir']) for k, val in pre_path_dict.items()}

    wf_prep = {
        'input_type': input_type,
        'file_type': file_type,
        'prev_qc': prev_qc,
        'preproc_res': preproc_results,
        'swomed_input': in_path_dict,
    }

    # Acquisitions per prediction:
    pred_acqui = {
        't1-like': replace_t1,
        'flair-like': replace_flair,
        'swi-like': replace_swi,
    }

    # wfargs are settings shared between workflows. It's clearer to have them all in one dict and pass it around
    wfargs = {
        'PREP_SETTINGS': wf_prep,
        'SUB_WF': True,  # Denotes that the workflows are stringed together
        'SUBJECT_LIST': sub_list,
        'DB': db_name,
        'DATA_DIR': subject_directory,  # Default base_directory for the datagrabber
        'BASE_DIR': out_dir,  # Default base_dir for each workflow
        'PREDICTION': prediction,  # Needed by the postproc for now
        'BRAIN_SEG': brain_seg,
        'SYNTHSEG_ON_CPU': ss_threads,  # Number of threads to use for Synthseg on CPUs
        'CUSTOM_LUT': custom_LUT,
        **descriptor_paths,
        'ACQUISITIONS': pred_acqui,
        'USE_T1': use_t1,
        'CONTAINER_IMAGE': container_image,
        'SYNTHSEG_IMAGE': synthseg_image,
        'CONTAINERIZE_NODES': containerized_nodes,
        # 'CONTAINER': True #  legacy variable. Only when used by SMOmed usually
        'MODELS_PATH': model,
        'GPU': None,  # gpu,
        'REG_PLUGIN_ARGS': reg_plugin_args,
        'PRED_PLUGIN_ARGS': pred_plugin_args,
        'ANONYMIZED': anonymize,
        'INTERPOLATION': interpolation,
        'PERCENTILE': percentile,
        'THRESHOLD': threshold,
        'THRESHOLD_PVS': threshold_pvs,
        'THRESHOLD_WMH': threshold_wmh,
        'THRESHOLD_CMB': threshold_cmb,
        'THRESHOLD_LAC': threshold_lac,
        'MIN_PVS_SIZE': min_pvs_size,
        'MIN_WMH_SIZE': min_wmh_size,
        'MIN_CMB_SIZE': min_cmb_size,
        'MIN_LAC_SIZE': min_lac_size,
        'IMAGE_SIZE': tuple(final_dimensions),
        'RESOLUTION': tuple(voxels_size),
        'ORIENTATION': 'RAS'}

    # Check if the AI models are available for the predictions
    check_input_for_pred(wfargs)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # Run the workflow
    if preproc_results is None:
        main_wf = generate_main_wf(**wfargs)
    else:
        main_wf = generate_main_wf_grab_preproc(**wfargs)

    if keep_all:
        config.enable_provenance()
        main_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    if debug:
        main_wf.config['execution']['stop_on_first_crash'] = 'True'
    main_wf.run(plugin=run_plugin, plugin_args=run_plugin_args)

    # Remove empty dir (I don't know how to prevent its creation)
    useless_folder = os.path.join(out_dir, 'results', 'results_summary', 'trait_added')
    if os.path.exists(useless_folder) and not os.listdir(useless_folder):
        os.rmdir(useless_folder)

    # Remove intermediate files if asked
    if remove_intermediates:
        workflow_dir = os.path.join(out_dir, 'main_workflow')
        shutil.rmtree(workflow_dir)
