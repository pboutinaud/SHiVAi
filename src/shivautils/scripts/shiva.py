#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.utils.parsing import shivaParser, set_args_and_check
from shivautils.workflows.main_workflow import generate_main_wf, generate_main_wf_grab_preproc
from nipype import config
import os
import shutil
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


def main():

    parser = shivaParser()
    args = set_args_and_check(parser)

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

    ss_threads = 0
    if args.brain_seg == 'synthseg_cpu':
        ss_threads = args.synthseg_threads

    # Plugin arguments for predictions (shiva pred and synthseg)
    pred_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4 --gpus 1'}
    reg_plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    if 'pred' in args.node_plugin_args.keys():
        pred_plugin_args = args.node_plugin_args['pred']
    if 'reg' in args.node_plugin_args.keys():
        reg_plugin_args = args.node_plugin_args['reg']

    wf_prep = {
        'input_type': args.input_type,
        'prev_qc': args.prev_qc,
        'preproc_res': args.preproc_results
    }

    # Acquisitions per prediction:
    pred_acqui = {
        't1-like': args.replace_t1,
        'flair-like': args.replace_flair,
        'swi-like': args.replace_swi,
    }

    # wfargs are settings shared between workflows. It's clearer to have them all in one dict and pass it around
    wfargs = {
        'PREP_SETTINGS': wf_prep,
        'SUB_WF': True,  # Denotes that the workflows are stringed together
        'SUBJECT_LIST': args.sub_list,
        'DB': args.db_name,
        'DATA_DIR': subject_directory,  # Default base_directory for the datagrabber
        'BASE_DIR': out_dir,  # Default base_dir for each workflow
        'PREDICTION': args.prediction,  # Needed by the postproc for now
        'BRAIN_SEG': args.brain_seg,
        'SYNTHSEG_ON_CPU': ss_threads,  # Number of threads to use for Synthseg on CPUs
        'CUSTOM_LUT': args.custom_LUT,
        'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
        'WMH_DESCRIPTOR': wmh_descriptor,
        'PVS_DESCRIPTOR': pvs_descriptor,
        'PVS2_DESCRIPTOR': pvs2_descriptor,
        'CMB_DESCRIPTOR': cmb_descriptor,
        'LAC_DESCRIPTOR': lac_descriptor,
        'ACQUISITIONS': pred_acqui,
        'USE_T1': args.use_t1,
        'CONTAINER_IMAGE': args.container_image,
        'SYNTHSEG_IMAGE': args.synthseg_image,
        'CONTAINERIZE_NODES': args.containerized_nodes,
        # 'CONTAINER': True #  legacy variable. Only when used by SMOmed usually
        'MODELS_PATH': args.model,
        'GPU': None,  # args.gpu,
        'REG_PLUGIN_ARGS': reg_plugin_args,
        'PRED_PLUGIN_ARGS': pred_plugin_args,
        'ANONYMIZED': args.anonymize,
        'NO_QC': args.noQC,
        'INTERPOLATION': args.interpolation,
        'PERCENTILE': args.percentile,
        'THRESHOLD': args.threshold,
        'THRESHOLD_PVS': args.threshold_pvs,
        'THRESHOLD_WMH': args.threshold_wmh,
        'THRESHOLD_CMB': args.threshold_cmb,
        'THRESHOLD_LAC': args.threshold_lac,
        'MIN_PVS_SIZE': args.min_pvs_size,
        'MIN_WMH_SIZE': args.min_wmh_size,
        'MIN_CMB_SIZE': args.min_cmb_size,
        'MIN_LAC_SIZE': args.min_lac_size,
        'IMAGE_SIZE': tuple(args.final_dimensions),
        'RESOLUTION': tuple(args.voxels_size),
        'ORIENTATION': 'RAS'}

    # Check if the AI models are available for the predictions
    check_input_for_pred(wfargs)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    # Run the workflow
    if args.preproc_results is None:
        main_wf = generate_main_wf(**wfargs)
    else:
        main_wf = generate_main_wf_grab_preproc(**wfargs)

    if args.keep_all:
        config.enable_provenance()
        main_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    if args.debug:
        main_wf.config['execution']['stop_on_first_crash'] = 'True'
    main_wf.run(plugin=args.run_plugin, plugin_args=args.run_plugin_args)

    # Remove empty dir (I don't know how to prevent its creation)
    useless_folder = os.path.join(out_dir, 'results', 'results_summary', 'trait_added')
    if os.path.exists(useless_folder) and not os.listdir(useless_folder):
        os.rmdir(useless_folder)

    # Remove intermediate files if asked
    if args.remove_intermediates:
        workflow_dir = os.path.join(out_dir, 'main_workflow')
        shutil.rmtree(workflow_dir)


if __name__ == "__main__":
    main()
