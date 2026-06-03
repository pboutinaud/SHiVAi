"""
Simplified DORA workflow for PVS detection from T1 or T2 images.

Runs: preprocessing (shiva brain masking) → PVS prediction → cluster labelling
      → register to native space → output.
No statistics, no report, no QC aggregation.

Designed to run inside a Docker container:
    docker run --rm --network none --gpus '"device=0"' \\
        -v /path/to/subject/:/input/:ro \\
        -v /path/to/output/:/output/ \\
        <image>:latest

Input:  /input/<subject_id>_<modality>.nii.gz
Output: /output/<subject_id>_pvs_posterior.nii.gz  (float64, probability map in [0, 1])
        /output/<subject_id>_pvs_mask.nii.gz       (uint8, binary mask: 0 = bg, 1 = PVS)
"""

import argparse
from pathlib import Path

from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber

from shivai.interfaces.shiva import Predict
from shivai.workflows.shiva_mask_wf import genWorkflow as gen_masking_wf
from shivai.interfaces.image import (Threshold, Normalization, CorrectAffine,
                                     Conform, Crop, Resample_from_to, Label_clusters, 
                                     Labelled_Clusters_Registration)


# ── Helper functions (used as nipype Function node targets) ──────────────────

def res_to_dict(sub_ids, in_files):
    """Aggregate per-subject results into a {subject_id: file} dict."""
    return {s: f for s, f in zip(sub_ids, in_files)}


def dict_to_res(sub_id, files_dict):
    """Extract a single subject's file from a {subject_id: file} dict."""
    return files_dict[sub_id]


def save_dora_outputs(raw_prediction, labelled_clusters, subject_id, output_dir):
    """Save the raw prediction (posterior) and binarized mask to the output directory.

    Output dtypes: posterior = float64, mask = uint8.
    """
    import nibabel as nib
    import numpy as np
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Raw prediction → posterior (float64)
    pred_img = nib.load(raw_prediction)
    posterior_data = pred_img.get_fdata().astype(np.float64)
    posterior_img = nib.Nifti1Image(posterior_data, pred_img.affine, pred_img.header)
    posterior_img.set_data_dtype(np.float64)
    posterior_path = str(out / f'{subject_id}_pvs_posterior.nii.gz')
    nib.save(posterior_img, posterior_path)

    # Labelled clusters → binary mask (uint8)
    lab_img = nib.load(labelled_clusters)
    mask_data = (lab_img.get_fdata() > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, lab_img.affine, lab_img.header)
    mask_img.set_data_dtype(np.uint8)
    mask_path = str(out / f'{subject_id}_pvs_mask.nii.gz')
    nib.save(mask_img, mask_path)

    return posterior_path, mask_path


def detect_modality(input_dir):
    """Detect modality from the input NIfTI filename.

    Expected format: <subject_id>_<modality>.nii.gz
    Returns (subject_id, modality, is_t2).
    """
    nifti_files = sorted(Path(input_dir).glob('*.nii*'))
    if not nifti_files:
        raise FileNotFoundError(f'No .nii.gz nor .nii files found in {input_dir}')
    if len(nifti_files) > 1:
        raise ValueError(f'Expected exactly one NIfTI file in {input_dir}, found {len(nifti_files)}')

    fname = nifti_files[0].name.replace('.nii.gz', '').replace('.nii', '')
    parts = fname.split('_')
    if len(parts) < 2:
        raise ValueError(f'Cannot parse subject_id and modality from "{fname}". '
                         'Expected format: <subject_id>_<modality>.nii.gz')
    modality = parts[-1]
    subject_id = '_'.join(parts[:-1])
    is_t2 = modality.lower() in ('t2w', 't2', 't2star')
    return subject_id, modality, is_t2


# ── Workflow generator ───────────────────────────────────────────────────────

def generate_dora_wf(**kwargs) -> Workflow:
    """
    Generate a simplified PVS-only workflow:
    preprocessing (shiva masking) → PVS prediction → cluster labelling
    → register to native space → output files.

    Required kwargs:
        BASE_DIR, DATA_DIR, OUTPUT_DIR, SUBJECT_LIST,
        PREDICTION, BRAIN_SEG, BRAINMASK_DESCRIPTOR, PVS_DESCRIPTOR, MODELS_PATH,
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

    # ── Preprocessing (shiva brain masking) ──────────────────────────────────
    
    datagrabber = Node(DataGrabber(infields=['subject_id'],
        outfields=['img1']),
        name='datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.template = '*.nii.gz'
    datagrabber.inputs.field_template = {'img1': '%s_*.nii.gz'}
    datagrabber.inputs.template_args = {'img1': [['subject_id']]}
    
    correct_affine_img1 = Node(CorrectAffine(), name="correct_affine_img1")
    correct_affine_img1.inputs.correction_threshold = kwargs['AFFINE_CORREC_THRESHOLD']
    
    main_wf.connect(subject_iterator, 'subject_id', datagrabber, 'subject_id')
    main_wf.connect(datagrabber, 'img1', correct_affine_img1, 'img')
    
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.voxels_tolerance = kwargs['TOLERANCE']
    conform.inputs.orientation = kwargs['ORIENTATION']
    
    main_wf.connect(correct_affine_img1, 'corrected_img', conform, 'img')
    
    mask_to_conform = Node(Resample_from_to(),
                           name="mask_to_conform")
    mask_to_conform.inputs.spline_order = 0

    main_wf.connect(conform, 'resampled', mask_to_conform, 'fixed_image')
    
    # Creating and incorporating the brain mask sub-wf
    masking_wf = gen_masking_wf(**kwargs)
    main_wf.add_nodes([masking_wf])
    
    main_wf.connect(correct_affine_img1, 'corrected_img', masking_wf, 'preconform.img')
    main_wf.connect(conform, 'resampled', masking_wf, 'intensity_norm_with_premask.input_image')
    main_wf.connect(masking_wf, 'proper_brain_mask.segmentation', mask_to_conform, 'moving_image')
    
    binarize_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="binarize_brain_mask")
    binarize_brain_mask.inputs.binarize = True
    binarize_brain_mask.inputs.minVol = 100  # Get rif of potential small clusters
    binarize_brain_mask.inputs.clusterCheck = 'keep_all'  # Keep all clusters above minVol

    main_wf.connect(mask_to_conform, 'resampled_image', binarize_brain_mask, 'img')
    
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    main_wf.connect(conform, 'resampled',
                     crop, 'apply_to')
    main_wf.connect(binarize_brain_mask, 'thresholded',
                     crop, 'roi_mask')
    
    # Apply the cropping to the mask
    mask_to_crop = Node(Resample_from_to(),
                        name='mask_to_crop')
    mask_to_crop.inputs.spline_order = 0  # should be equivalent to NearestNeighbor(?)
    mask_to_crop.inputs.out_name = 'brainmask_cropped.nii.gz'

    main_wf.connect(binarize_brain_mask, 'thresholded', mask_to_crop, 'moving_image')
    main_wf.connect(crop, "cropped", mask_to_crop, 'fixed_image')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 1)
    img1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img1_final_intensity_normalization")
    if 'inverse_t2' in kwargs['ACQUISITIONS']:
        img1_norm.inputs.inverse = kwargs['ACQUISITIONS']['inverse_t2']
    main_wf.connect(crop, 'cropped',
                    img1_norm, 'input_image')
    main_wf.connect(mask_to_crop, 'resampled_image',
                    img1_norm, 'brain_mask')



    # ── PVS prediction ───────────────────────────────────────────────────────
    predict_node = Node(Predict(), name='predict_pvs')
    predict_node.inputs.out_filename = 'pvs_map.nii.gz'
    predict_node.inputs.model = kwargs['MODELS_PATH']
    predict_node.inputs.descriptor = kwargs['BRAINMASK_DESCRIPTOR']
    predict_node.inputs.gpu_number = kwargs['GPU']
    
    main_wf.connect(img1_norm, 'intensity_normalized', predict_node, 't1')
    

    # ── Cluster labelling (threshold + connected-component filtering) ────────
    cluster_labelling = Node(Label_clusters(), name='cluster_labelling_pvs')
    cluster_labelling.inputs.thr_cluster_val = kwargs['THRESHOLD_PVS']
    cluster_labelling.inputs.thr_cluster_size = kwargs['MIN_PVS_SIZE'] - 1
    cluster_labelling.inputs.out_name = 'pvs_labelled_clusters.nii.gz'

    main_wf.connect(predict_node, 'segmentation', cluster_labelling, 'biomarker_raw')
    main_wf.connect(mask_to_crop, 'resampled_image', cluster_labelling, 'brain_seg')

    # ── Register results back to native input space ──────────────────────────
    # Labelled clusters → native space (preserves integer labels)
    clusters_to_native = Node(Labelled_Clusters_Registration(), name='clusters_to_native')
    clusters_to_native.inputs.out_name = 'pvs_clusters_native.nii.gz'

    main_wf.connect(cluster_labelling, 'labelled_biomarkers', clusters_to_native, 'input_image')
    main_wf.connect(datagrabber, 'img1', clusters_to_native, 'target_image')

    # Raw prediction (posterior) → native space (continuous resampling)
    posterior_to_native = Node(Labelled_Clusters_Registration(), name='posterior_to_native')
    posterior_to_native.inputs.out_name = 'pvs_posterior_native.nii.gz'
    posterior_to_native.inputs.input_type = 'pred'

    main_wf.connect(predict_node, 'segmentation', posterior_to_native, 'input_image')
    main_wf.connect(datagrabber, 'img1', posterior_to_native, 'target_image')

    # ── Save outputs (posterior + binary mask in native space) ────────────────
    save_node = Node(
        Function(input_names=['raw_prediction', 'labelled_clusters', 'subject_id', 'output_dir'],
                 output_names=['posterior_path', 'mask_path'],
                 function=save_dora_outputs),
        name='save_outputs')
    save_node.inputs.output_dir = kwargs['OUTPUT_DIR']

    main_wf.connect(posterior_to_native, 'output_image', save_node, 'raw_prediction')
    main_wf.connect(clusters_to_native, 'output_image', save_node, 'labelled_clusters')
    main_wf.connect(subject_iterator, 'subject_id', save_node, 'subject_id')

    return main_wf


# ── CLI ──────────────────────────────────────────────────────────────────────

class Args(argparse.Namespace):
    """Typed namespace for CLI arguments."""
    input_dir: Path
    output_dir: Path
    work_dir: Path
    models_path: Path
    brainmask_descriptor: Path | None
    pvs_descriptor: Path | None
    threshold: float
    min_pvs_size: int
    gpu: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description='DORA: simplified PVS detection (preproc → prediction → output)')

    parser.add_argument('--input', '--input_dir', type=Path, default='/input/',
                        dest='input_dir',
                        help='Directory containing <subject_id>_<modality>.nii.gz')
    parser.add_argument('--output', '--output_dir', type=Path, default='/output/',
                        dest='output_dir',
                        help='Directory for output files')
    parser.add_argument('--work_dir', type=Path, default='/tmp/dora_work',
                        help='Nipype working directory')

    # Model paths
    parser.add_argument('--models_path', type=Path,
                        default='/opt/model/weights',
                        help='Base path to the model files')
    parser.add_argument('--brainmask_descriptor', type=Path, default=None,
                        help='Brainmask model descriptor JSON '
                             '(default: <models_path>/brainmask/model_info.json)')
    parser.add_argument('--pvs_descriptor', type=Path, default=None,
                        help='PVS model descriptor JSON '
                             '(default: <models_path>/T1-PVS/model_info.json)')

    # Processing
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='PVS prediction threshold')
    parser.add_argument('--min_pvs_size', type=int, default=5,
                        help='Minimum PVS cluster size in voxels')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (-1 for CPU)')
    return parser.parse_args(namespace=Args())


def build_kwargs(args: Args):
    """Build the kwargs dict expected by the workflow generators."""

    for arg in args.__dict__:
        if isinstance(args.__dict__[arg], Path):
            args.__dict__[arg] = str(args.__dict__[arg].resolve())
    # Discover subject and modality from the input file(s)
    subject_id, modality, is_t2 = detect_modality(args.input_dir)
    print(f'Subject: {subject_id}, Modality: {modality}, T2-like: {is_t2}')

    # Resolve model descriptor paths
    models_path = Path(args.models_path)
    brainmask_descriptor = (args.brainmask_descriptor
                            or models_path / 'brainmask' / 'model_info.json')
    pvs_descriptor = (args.pvs_descriptor
                      or models_path / 'T1-PVS' / 'model_info.json')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    return {
        # Directories
        'BASE_DIR': str(args.work_dir),
        'DATA_DIR': str(args.input_dir),
        'OUTPUT_DIR': str(args.output_dir),
        'SUBJECT_LIST': [subject_id],

        # Prediction / segmentation
        'PREDICTION': ['PVS'],
        'BRAIN_SEG': 'shiva_gpu',
        'USE_T1': True,

        # Model paths
        'MODELS_PATH': str(models_path),
        'BRAINMASK_DESCRIPTOR': str(brainmask_descriptor),
        'PVS_DESCRIPTOR': str(pvs_descriptor),

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
        'GPU': args.gpu if args.gpu >= 0 else None,
        'AI_THREADS': 8,
        'BATCH_SIZE': 20,
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

        # Acquisitions — T2w detection sets inverse normalization
        'ACQUISITIONS': {
            't1-like': None,
            'flair-like': None,
            'swi-like': None,
            'inverse_t2': is_t2,
        },


        # Misc
        'CUSTOM_LUT': None,
        'ANONYMIZED': False,
        'SAVE_GRAPH': False,
        'SUB_WF': True,
        'DB': None,
        'SWI_FILE_NUM': None,
    }


def main():
    args = parse_args()
    kwargs = build_kwargs(args)
    wf = generate_dora_wf(**kwargs)
    wf.run()


if __name__ == '__main__':
    main()
