"""
Simplified DORA workflow for PVS detection from T1 images.

Runs: T1 preprocessing (shiva brain masking) → PVS prediction → cluster labelling → output.
No statistics, no report, no QC aggregation.

Designed to run inside a Docker container:
    docker run --rm --network none --gpus '"device=0"' \
        -v /path/to/subject/:/input/:ro \
        -v /path/to/output/:/output/ \
        <image>:latest

Input:  /input/<subject_id>_<modality>.nii.gz
Output: /output/<subject_id>_pvs_posterior.nii.gz  (raw prediction probability map)
        /output/<subject_id>_pvs_mask.nii.gz       (binarized thresholded prediction)
"""

import os
import glob
import argparse

from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function

from shivai.workflows.preprocessing_shiva_masking import genWorkflow as genWorkflow_preproc_shiva_mask
from shivai.workflows.predict_wf import genWorkflow as genWorkflow_prediction
from shivai.interfaces.image import Label_clusters


# ── Helper functions (used as nipype Function node targets) ──────────────────

def res_to_dict(sub_ids, in_files):
    """Aggregate per-subject results into a {subject_id: file} dict."""
    return {s: f for s, f in zip(sub_ids, in_files)}


def dict_to_res(sub_id, files_dict):
    """Extract a single subject's file from a {subject_id: file} dict."""
    return files_dict[sub_id]


def save_dora_outputs(raw_prediction, labelled_clusters, subject_id, output_dir):
    """Save the raw prediction (posterior) and binarized mask to the output directory."""
    import nibabel as nib
    import numpy as np
    import os
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    # Raw prediction → posterior
    posterior_path = os.path.join(output_dir, f'{subject_id}_pvs_posterior.nii.gz')
    shutil.copy2(raw_prediction, posterior_path)

    # Labelled clusters → binary mask
    lab_img = nib.load(labelled_clusters)
    mask_data = (lab_img.get_fdata() > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, lab_img.affine, lab_img.header)
    mask_path = os.path.join(output_dir, f'{subject_id}_pvs_mask.nii.gz')
    nib.save(mask_img, mask_path)

    return posterior_path, mask_path


# ── Workflow generator ───────────────────────────────────────────────────────

def generate_dora_wf(**kwargs) -> Workflow:
    """
    Generate a simplified PVS-only workflow:
    T1 preprocessing (shiva masking) → PVS prediction → cluster labelling → output files.

    Required kwargs (see ``build_kwargs`` for the full list):
        BASE_DIR, DATA_DIR, OUTPUT_DIR, SUBJECT_LIST,
        PREDICTION=['PVS'], BRAIN_SEG='shiva',
        BRAINMASK_DESCRIPTOR, PVS_DESCRIPTOR, MODELS_PATH,
        IMAGE_SIZE, THRESHOLD_PVS, MIN_PVS_SIZE,
        + all kwargs consumed by the preprocessing and prediction generators.
    """
    # ── Main workflow ────────────────────────────────────────────────────────
    main_wf = Workflow('dora_workflow')
    main_wf.base_dir = kwargs['BASE_DIR']

    # Subject iterator (typically single-element for a container run)
    subject_iterator = Node(
        IdentityInterface(fields=['subject_id'], mandatory_inputs=True),
        name='subject_iterator')
    subject_iterator.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # ── Preprocessing (shiva brain masking, T1 only) ─────────────────────────
    wf_preproc = genWorkflow_preproc_shiva_mask(**kwargs, wf_name='dora_preprocessing')

    # Override the datagrabber for our flat input structure:
    #   /input/<subject_id>_<modality>.nii.gz
    datagrabber = wf_preproc.get_node('datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.template = '*.nii.gz'
    datagrabber.inputs.field_template = {'img1': '%s_*.nii.gz'}
    datagrabber.inputs.template_args = {'img1': [['subject_id']]}

    main_wf.connect(subject_iterator, 'subject_id', wf_preproc, 'datagrabber.subject_id')

    # ── JoinNodes (aggregate per-subject outputs into dicts for prediction) ──
    preproc_joiner_mask = JoinNode(
        Function(input_names=['sub_ids', 'in_files'],
                 output_names=['files_dict'],
                 function=res_to_dict),
        joinsource=subject_iterator,
        joinfield=['sub_ids', 'in_files'],
        name='preproc_joiner_mask')
    main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_mask, 'sub_ids')
    main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', preproc_joiner_mask, 'in_files')

    preproc_joiner_t1 = JoinNode(
        Function(input_names=['sub_ids', 'in_files'],
                 output_names=['files_dict'],
                 function=res_to_dict),
        joinsource=subject_iterator,
        joinfield=['sub_ids', 'in_files'],
        name='preproc_joiner_t1')
    main_wf.connect(subject_iterator, 'subject_id', preproc_joiner_t1, 'sub_ids')
    main_wf.connect(wf_preproc, 'img1_final_intensity_normalization.intensity_normalized',
                    preproc_joiner_t1, 'in_files')

    # ── PVS prediction ───────────────────────────────────────────────────────
    segmentation_wf = genWorkflow_prediction(**kwargs)
    main_wf.connect(preproc_joiner_mask, 'files_dict', segmentation_wf, 'predict_pvs.brainmask_files')
    main_wf.connect(preproc_joiner_t1, 'files_dict', segmentation_wf, 'predict_pvs.primary_image_file')

    # Extract per-subject segmentation from prediction dict
    seg_getter_pvs = Node(
        Function(input_names=['sub_id', 'files_dict'],
                 output_names=['segmentation'],
                 function=dict_to_res),
        name='seg_getter_pvs')
    main_wf.connect(segmentation_wf, 'predict_pvs.segmentations', seg_getter_pvs, 'files_dict')
    main_wf.connect(subject_iterator, 'subject_id', seg_getter_pvs, 'sub_id')

    # ── Cluster labelling (threshold + connected-component filtering) ────────
    cluster_labelling = Node(Label_clusters(), name='cluster_labelling_pvs')
    cluster_labelling.inputs.thr_cluster_val = kwargs['THRESHOLD_PVS']
    cluster_labelling.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1
    cluster_labelling.inputs.out_name = 'pvs_labelled_clusters.nii.gz'

    main_wf.connect(seg_getter_pvs, 'segmentation', cluster_labelling, 'biomarker_raw')
    main_wf.connect(wf_preproc, 'mask_to_crop.resampled_image', cluster_labelling, 'brain_seg')

    # ── Save outputs (posterior + binary mask) ───────────────────────────────
    save_node = Node(
        Function(input_names=['raw_prediction', 'labelled_clusters', 'subject_id', 'output_dir'],
                 output_names=['posterior_path', 'mask_path'],
                 function=save_dora_outputs),
        name='save_outputs')
    save_node.inputs.output_dir = kwargs['OUTPUT_DIR']

    main_wf.connect(seg_getter_pvs, 'segmentation', save_node, 'raw_prediction')
    main_wf.connect(cluster_labelling, 'labelled_biomarkers', save_node, 'labelled_clusters')
    main_wf.connect(subject_iterator, 'subject_id', save_node, 'subject_id')

    return main_wf


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='DORA: simplified PVS detection (T1 → preproc → prediction → output)')

    parser.add_argument('--input_dir', type=str, default='/input/',
                        help='Directory containing <subject_id>_<modality>.nii.gz')
    parser.add_argument('--output_dir', type=str, default='/output/',
                        help='Directory for output files')
    parser.add_argument('--work_dir', type=str, default='/tmp/dora_work',
                        help='Nipype working directory')

    # Model paths
    parser.add_argument('--models_path', type=str, required=True,
                        help='Base path to the model files')
    parser.add_argument('--brainmask_descriptor', type=str, default=None,
                        help='Brainmask model descriptor JSON (relative to models_path)')
    parser.add_argument('--pvs_descriptor', type=str, default=None,
                        help='PVS model descriptor JSON (relative to models_path)')

    # Processing
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='PVS prediction threshold')
    parser.add_argument('--min_pvs_size', type=int, default=5,
                        help='Minimum PVS cluster size in voxels')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (-1 for CPU)')

    # TODO: add any additional arguments as needed
    return parser.parse_args()


def build_kwargs(args):
    """Build the kwargs dict expected by the workflow generators."""

    # Discover subject IDs from input filenames: <subject_id>_<modality>.nii.gz
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.nii.gz')))
    if not input_files:
        raise FileNotFoundError(f'No .nii.gz files found in {args.input_dir}')

    subject_list = []
    for f in input_files:
        basename = os.path.basename(f).replace('.nii.gz', '')
        # subject_id = everything before the last underscore (modality follows)
        parts = basename.split('_')
        subject_id = '_'.join(parts[:-1]) if len(parts) > 1 else basename
        if subject_id and subject_id not in subject_list:
            subject_list.append(subject_id)

    if not subject_list:
        raise ValueError(
            'Could not extract subject IDs from input filenames. '
            'Expected format: <subject_id>_<modality>.nii.gz')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    return {
        # Directories
        'BASE_DIR': args.work_dir,
        'DATA_DIR': args.input_dir,
        'OUTPUT_DIR': args.output_dir,
        'SUBJECT_LIST': subject_list,

        # Prediction / segmentation
        'PREDICTION': ['PVS'],
        'BRAIN_SEG': 'shiva',
        'USE_T1': True,

        # Model paths
        'MODELS_PATH': args.models_path,
        'BRAINMASK_DESCRIPTOR': args.brainmask_descriptor,   # TODO: set sensible default
        'PVS_DESCRIPTOR': args.pvs_descriptor,               # TODO: set sensible default

        # Container settings (already inside a container → no nested containerisation)
        'CONTAINERIZE_NODES': False,
        'CONTAINER_RUNTIME': None,
        'CONTAINER_IMAGE': None,

        # Image processing parameters
        'IMAGE_SIZE': (160, 214, 176),
        'RESOLUTION': (1.0, 1.0, 1.0),
        'TOLERANCE': (0.0, 0.0, 0.0),
        'ORIENTATION': 'RAS',
        'AFFINE_CORREC_THRESHOLD': 0.005,
        'PERCENTILE': 99.0,
        'THRESHOLD': 0.5,
        'INTERPOLATION': 'WelchWindowedSinc',

        # PVS thresholds
        'THRESHOLD_PVS': args.threshold,
        'MIN_PVS_SIZE': args.min_pvs_size,

        # GPU / performance
        'GPU': args.gpu if args.gpu >= 0 else None,  # TODO: verify convention with predict_wf
        'AI_THREADS': 8,        # TODO: make configurable if needed
        'BATCH_SIZE': 20,       # TODO: make configurable if needed
        'PRED_PLUGIN_ARGS': {},
        'REG_PLUGIN_ARGS': {},

        # Preprocessing settings
        'PREP_SETTINGS': {
            'input_type': 'standard',
            'file_type': 'nifti',
            'preproc_only': False,
            'prev_qc': None,
            'preproc_res': None,
            'prereg_flair': False,
        },

        # Acquisitions
        'ACQUISITIONS': {
            't1-like': None,
            'flair-like': None,
            'swi-like': None,
            'inverse_t2': False,
        },

        # Misc
        'CUSTOM_LUT': None,
        'ANONYMIZED': False,
        'SAVE_GRAPH': False,
        'SUB_WF': True,
        'DB': None,
        'SWI_FILE_NUM': None,

        # TODO: add any further kwargs required by the workflow generators
    }


def main():
    args = parse_args()
    kwargs = build_kwargs(args)
    wf = generate_dora_wf(**kwargs)

    wf.run()


if __name__ == '__main__':
    main()
