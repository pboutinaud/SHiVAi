#!/usr/bin/env python
"""Docker entrypoint for the DORA PVS segmentation challenge.

Discovers the single NIfTI file in the input directory, detects its modality
(T1w / T2w), runs the DORA workflow, and ensures the output directory
contains exactly two files:
    <subject_id>_pvs_posterior.nii.gz   (float64, values in [0, 1])
    <subject_id>_pvs_mask.nii.gz        (uint8, 0 = background, 1 = PVS)

Usage (inside container):
    python /opt/model/run_inference_dora.py --input /input --output /output
"""

import sys
import shutil
from pathlib import Path

from shivai.workflows.dora_workflow import generate_dora_wf, detect_modality


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='DORA PVS segmentation challenge entrypoint')
    parser.add_argument('--input', type=str, default='/input/',
                        dest='input_dir',
                        help='Input directory with a single NIfTI file')
    parser.add_argument('--output', type=str, default='/output/',
                        dest='output_dir',
                        help='Output directory for results')
    args = parser.parse_args()

    # ── Discover subject and modality ────────────────────────────────────────
    subject_id, modality, is_t2 = detect_modality(args.input_dir)
    print(f'Subject: {subject_id}, Modality: {modality}, T2-like: {is_t2}')

    work_dir = '/tmp/dora_work'
    models_path = Path('/opt/model/weights')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # ── Build kwargs and run workflow ────────────────────────────────────────
    kwargs = {
        'BASE_DIR': work_dir,
        'DATA_DIR': args.input_dir,
        'OUTPUT_DIR': args.output_dir,
        'SUBJECT_LIST': [subject_id],

        'PREDICTION': ['PVS'],
        'BRAIN_SEG': 'shiva_gpu',
        'USE_T1': True,

        'MODELS_PATH': str(models_path),
        'BRAINMASK_DESCRIPTOR': str(models_path / 'brainmask' / 'model_info.json'),
        'PVS_DESCRIPTOR': str(models_path / 'T1-PVS' / 'model_info.json'),

        'CONTAINERIZE_NODES': False,
        'CONTAINER_RUNTIME': None,
        'CONTAINER_IMAGE': None,

        'IMAGE_SIZE': (160, 214, 176),
        'RESOLUTION': (1.0, 1.0, 1.0),
        'TOLERANCE': (0.0, 0.0, 0.0),
        'ORIENTATION': 'RAS',
        'AFFINE_CORREC_THRESHOLD': 0.005,
        'PERCENTILE': 99.0,
        'THRESHOLD': 0.5,
        'INTERPOLATION': 'WelchWindowedSinc',

        'THRESHOLD_PVS': 0.5,
        'MIN_PVS_SIZE': 1,

        'GPU': 0,
        'AI_THREADS': 8,
        'BATCH_SIZE': 20,
        'PRED_PLUGIN_ARGS': {},
        'REG_PLUGIN_ARGS': {},

        'PREP_SETTINGS': {
            'input_type': 'standard',
            'file_type': 'nifti',
            'preproc_only': False,
            'prev_qc': None,
            'preproc_res': None,
            'prereg_flair': False,
        },

        'ACQUISITIONS': {
            't1-like': None,
            'flair-like': None,
            'swi-like': None,
            'inverse_t2': is_t2,
        },

        'CUSTOM_LUT': None,
        'ANONYMIZED': False,
        'SAVE_GRAPH': False,
        'SUB_WF': True,
        'DB': None,
        'SWI_FILE_NUM': None,
    }

    wf = generate_dora_wf(**kwargs)
    wf.run()

    # ── Validate and clean output directory ──────────────────────────────────
    expected = {
        f'{subject_id}_pvs_posterior.nii.gz',
        f'{subject_id}_pvs_mask.nii.gz',
    }

    output_path = Path(args.output_dir)
    for name in expected:
        fpath = output_path / name
        if not fpath.is_file():
            print(f'ERROR: expected output file missing: {fpath}', file=sys.stderr)
            sys.exit(1)

    # Remove any extra files/dirs (challenge requires exactly 2 files)
    for entry in output_path.iterdir():
        if entry.name not in expected:
            print(f'Removing unexpected output entry: {entry}', file=sys.stderr)
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    print('Done. Output files:')
    for name in sorted(expected):
        print(f'  {output_path / name}')


if __name__ == '__main__':
    main()
