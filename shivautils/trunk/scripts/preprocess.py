#!/usr/bin/env python
"""Script for the anatomical preprocessing applied prior to predictions on the
   images."""
import os
import argparse

import nibabel as nb
import nibabel.processing as nip

from shivautils.image import normalization, crop, threshold


DESCRIPTION = """SHIVA preprocessing for deep learning predictors.
                 Perform resampling of a structural NIfTI head image,
                 followed by intensity normalization, and cropping centered on
                 the brain."""


def existing_file(file):
    """Checking if file exist

    Args:
        f (_type_): _description_

    Raises:
        argparse.ArgumentTypeError: _description_

    Returns:
        _type_: _description_
    """
    if not os.path.isfile(file):
        raise argparse.ArgumentTypeError(file + " not found.")
    else:
        return file


def build_args_parser():
    """Create a command line to specify arguments with argparse

    Returns:
        arguments
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='input', type=existing_file,
                        help='Input NIfTI image to preprocess',
                        metavar='path/to/existing/nifti.nii',
                        required=True)

    parser.add_argument('--out', dest='output', type=str,
                        help='Output file path',
                        metavar='path/to/cropped.nii',
                        required=True)

    parser.add_argument('--minimal_dimensions', nargs='+', type=int,
                        default=(256, 256, 256),
                        help='Minimal intermediate dimensions (for maintaining'
                        'the field of view of the image before cropping and'
                        'avoiding index errors).')

    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Treshold for the brain_mask')

    parser.add_argument('--binarize', type=bool, default=False,
                        help='Binarized intensities voxel of brain_mask')

    parser.add_argument('--final_dimensions', nargs='+', type=int,
                        default=(160, 214, 176),
                        help='Final image array size in i, j, k.')

    parser.add_argument('--voxel_size', nargs='+', type=float,
                        default=(1.0, 1.0, 1.0),
                        help='Voxel size of final image')

    return parser


def main():
    """Run preprocessing pipeline."""
    parser = build_args_parser()
    args = parser.parse_args()
    img = nb.loadsave.load(args.input)

    # argument validation
    required_ndim = 3
    if not isinstance(img, nb.nifti1.Nifti1Image):
        raise TypeError("Only NIfTI1 images are supported")
    if len(args.minimal_dimensions) != required_ndim:
        raise TypeError("Instantiate minimal dimensions as : int int int")
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError('The threshold must be beetween 0 and 1')
    if img.ndim != required_ndim:
        raise ValueError("Only 3D images are supported.")
    if len(args.final_dimensions) != required_ndim:
        raise ValueError(f"`dimensions` must have {required_ndim} values")
    if len(args.voxel_size) != required_ndim:
        raise ValueError(f"`voxel_size` must have {required_ndim} values")

    final_dimensions = tuple(args.final_dimensions)
    voxel_size = tuple(args.voxel_size)

    # These are the minimum dimensions to maintain the field of view of the
    # image
    minimal_dimensions = tuple(args.minimal_dimensions)

    # To keep the same field of view as the input image
    # (above minimal dimensions) :
    resampled_dimensions = img.shape * nb.affines.voxel_sizes(img.affine)
    resampled_dimensions = list((round(resampled_dimensions[0]),
                                 round(resampled_dimensions[1]),
                                 round(resampled_dimensions[2])))
    index = list(range(len(resampled_dimensions)))
    for i in index:
        if resampled_dimensions[i] < minimal_dimensions[i]:
            resampled_dimensions[i] = minimal_dimensions[i]

    # conform image to desired voxel sizes and dimensions
    resampled = nip.conform(img, out_shape=resampled_dimensions,
                            voxel_size=voxel_size, order=3,
                            cval=0.0, orientation='RAS',
                            out_class=None)

    # intensity normalization
    img_normalized, report, mode = normalization(resampled, 99)
    nb.loadsave.save(img_normalized, os.path.join(args.output, 'img_normalized.nii.gz'))
    thresholded = threshold(img_normalized,
                            thr=args.threshold,
                            binarize=True)
    nb.loadsave.save(thresholded, os.path.join(args.output, 'brainmask.nii.gz'))
    cropped = crop(roi_mask=thresholded,
                   apply_to=img_normalized,
                   dimensions=final_dimensions,
                   )

    # save the image to the desired path
    nb.loadsave.save(cropped[0],
                     os.path.join(args.output, 'preproc.nii.gz'))


if __name__ == "__main__":
    main()
