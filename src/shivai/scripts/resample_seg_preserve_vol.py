#!/usr/bin/env python
"""Resample a labelled cluster (or segmentation) image to a target space,
preserving cluster volumes as much as possible.

Wraps :func:`shivai.postprocessing.clusters.resample_cluster_img`.
"""

import argparse
import numpy as np
import nibabel as nib
from scipy.io import loadmat

from shivai.postprocessing.clusters import resample_cluster_img


def _load_ants_affine(mat_path, inverse=False):
    """Read an ANTs .mat affine and return a 4x4 matrix.

    Reproduces the logic used in
    :class:`shivai.interfaces.image.Labelled_Clusters_Registration`.
    """
    mat = loadmat(mat_path)
    key_name = [k for k in mat if 'AffineTransform_' in k][0]
    raw = mat[key_name]
    fixed = mat['fixed']
    A = raw[:9].reshape((3, 3))
    t = raw[9:12].squeeze()
    c = fixed.squeeze()
    affine = np.eye(4)
    affine[:3, :3] = A
    affine[:3, 3] = t + c - A @ c
    if inverse:
        affine = np.linalg.inv(affine)
    return affine


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Resample a labelled-cluster / segmentation image into the '
            'space of a target image while preserving cluster volumes.'
        ),
    )
    parser.add_argument(
        '-i', '--input-image',
        required=True,
        help='Nifti file containing labelled clusters (integer labels).',
    )
    parser.add_argument(
        '-t', '--target-image',
        required=True,
        help='Nifti file defining the target space for resampling.',
    )
    parser.add_argument(
        '-o', '--output',
        default='resampled_clusters.nii.gz',
        help='Output filename (default: resampled_clusters.nii.gz).',
    )
    parser.add_argument(
        '--transform-affine',
        default=None,
        help='ANTs .mat affine file encoding the linear transform between the two spaces (optional).',
    )
    parser.add_argument(
        '--inverse-affine',
        action='store_true',
        default=False,
        help='Invert the ANTs affine before applying it.',
    )
    parser.add_argument(
        '--input-type',
        choices=['map', 'pred', 'anat'],
        default='map',
        help=(
            'Type of input data. '
            '"map" (default): labelled cluster map (binary mask or one integer '
            'per cluster) — uses smart per-cluster resampling that preserves '
            'cluster volumes. '
            '"pred": continuous prediction map (e.g. posterior probabilities) '
            '— uses multi-threshold level-set smart resampling that preserves '
            'clusters at all threshold levels. '
            '"anat": anatomical or other continuous image — uses standard '
            'spline interpolation (may lose small clusters).'
        ),
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help=(
            'Minimum threshold level for the "pred" mode (default: 0.05). '
            'Ignored for other input types.'
        ),
    )
    parser.add_argument(
        '--threshold-step',
        type=float,
        default=0.05,
        help=(
            'Step between threshold levels for the "pred" mode (default: 0.05). '
            'Ignored for other input types.'
        ),
    )
    parser.add_argument(
        '--n-parallel',
        type=int,
        default=8,
        help='Number of parallel threads for per-cluster resampling (default: 8).',
    )

    args = parser.parse_args()

    cluster_img = nib.load(args.input_image)
    target_img = nib.load(args.target_image)

    transform_affine = None
    if args.transform_affine is not None:
        transform_affine = _load_ants_affine(
            args.transform_affine,
            inverse=args.inverse_affine,
        )

    resampled_img = resample_cluster_img(
        cluster_img,
        target_img,
        input_type=args.input_type,
        transform_affine=transform_affine,
        n_parallel=args.n_parallel,
        threshold=args.threshold,
        threshold_step=args.threshold_step,
    )

    nib.save(resampled_img, args.output)
    print(f'Saved resampled image to {args.output}')


if __name__ == '__main__':
    main()
