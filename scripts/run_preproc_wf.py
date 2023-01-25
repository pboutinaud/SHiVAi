"""Script workflow"""
import os
import argparse
from shivautils.workflows.vrs_preprocessing import genWorkflow

DESCRIPTION = """SHIVA preprocessing for deep learning predictors.
                 Perform resampling of a structural NIfTI head image,
                 followed by intensity normalization, and cropping centered on
                 the brain. A nipype workflow is used to preprocess a lot of
                 image in same time"""


def build_args_parser():
    """Create a command line to specify arguments with argparse

    Returns:
        arguments
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='input',
                        help='Input NIfTI image to preprocess',
                        metavar='path/to/existing/nifti.nii',
                        required=True)

    parser.add_argument('--out', dest='output', type=str,
                        help='Output file path',
                        metavar='path/to/cropped.nii',
                        required=True)

    parser.add_argument('--percentile', type=float, default=99,
                        help='value to threshold above this percentile')

    parser.add_argument('--final_dimensions', nargs='+', type=int,
                        default=(160, 214, 176),
                        help='Final image array size in i, j, k.')

    parser.add_argument('--voxel_size', nargs='+', type=float,
                        default=(1.0, 1.0, 1.0),
                        help='Voxel size of final image')

    return parser


def main():
    """Run a nipype workflow
    """
    parser = build_args_parser()
    args = parser.parse_args()
    # the path from which the images to be preprocessed come
    data_dir = os.path.normpath(args.input)
    # the preprocessed image destination path
    output_dir = args.output
    subject_list = os.listdir(data_dir)
    type(subject_list)

    # This value of percentile is adapted to CMBDOU cohort and GIN
    # preprocessing, to have a percentile adapted to another cohort
    # see 'preproc_wf_adapted_percentile.py' script
    percentile = args.percentile
    voxel_size = tuple(args.voxel_size)
    final_dimensions = tuple(args.final_dimensions)

    args = {'SUBJECT_LIST': subject_list,
            'BASE_DIR': data_dir,
            'percentile': percentile,
            'voxel_size': voxel_size,
            'final_dimensions': final_dimensions}

    wf = genWorkflow(**args)
    wf.base_dir = output_dir
    wf.run(plugin="MultiProc")


if __name__ == "__main__":
    main()
