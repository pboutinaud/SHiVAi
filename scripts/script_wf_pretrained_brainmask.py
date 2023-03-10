"""Script workflow"""
import os
import argparse
import json



from shivautils.workflows.wf_pretrained_brainmask import genWorkflow

DESCRIPTION = """SHIVA preprocessing for deep learning predictors.
                 Perform resampling of a structural NIfTI head image,
                 followed by intensity normalization, and cropping centered on
                 the brain. A nipype workflow is used to preprocess a lot of
                 image in same time"""
                 
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

    parser.add_argument('--in', dest='input',
                        help='JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/slicer_extension_database.json',
                        required=True)

    parser.add_argument('--out', dest='output',
                        type=str,
                        help='Output folder path (nipype working directory)',
                        metavar='path/to/nipype_work_dir',
                        required=True)

    parser.add_argument('--grab', dest='grab_pattern',
                        type=str,
                        help='data grabber pattern, between quotes',
                        metavar='%s/*nii',
                        default='%s/*nii',
                        required=True)

    parser.add_argument('--percentile',
                        type=float,
                        default=99,
                        help='Threshold value expressed as percentile')

    parser.add_argument('--final_dimensions',
                        nargs='+', type=int,
                        default=(160, 214, 176),
                        help='Final image array size in i, j, k.')

    parser.add_argument('--voxel_size', nargs='+',
                        type=float,
                        default=(1.0, 1.0, 1.0),
                        help='Voxel size of final image')
                        
    parser.add_argument('--model',
                        type=existing_file,
                        default=None,
                        help='path to model descriptor')

    parser.add_argument('--simg',
                        type=existing_file,
                        default=None,
                        help='Predictor Singularity image to use')
    
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='GPU to use.')

    return parser


def main():
    """Parameterize and run the nipype preprocessing workflow."""
    parser = build_args_parser()
    args = parser.parse_args()
    # the path from which the images to be preprocessed come
    # the preprocessed image destination path
    with open(args.input, 'r') as json_in:
        subject_dict = json.load(json_in)
    
    out_dir = subject_dict['parameters']['out_dir']
    wfargs = {'SUBJECT_LIST': list(subject_dict['all_files'].keys()),
              'ARGS': subject_dict,
              'BASE_DIR': out_dir}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    data_dir = subject_dict['files_dir']

    wf = genWorkflow(**wfargs)
    wf.base_dir = out_dir
    wf.get_node('dataGrabber').inputs.base_directory = data_dir
    wf.get_node('dataGrabber').inputs.template = args.grab_pattern
    wf.get_node('dataGrabber').inputs.template_args = {'main': [['subject_id']]}
    wf.get_node('conform').inputs.dimensions = (256, 256, 256)
    wf.get_node('conform').inputs.voxel_size = tuple(args.voxel_size)
    wf.get_node('conform').inputs.orientation = 'RAS'
    wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)
    wf.get_node('crop_2').inputs.final_dimensions = tuple(args.final_dimensions)
    wf.get_node('intensity_normalization').inputs.percentile = args.percentile
    wf.get_node('intensity_normalization_2').inputs.percentile = subject_dict['parameters']['percentile']
    # Predictor model descriptor file (JSON)
    wf.get_node('brain_mask_2').inputs.descriptor = args.model
    # singularity container bind mounts (so that folders of 
    # the host appear inside the container)
    # gcc, cuda libs, model directory and source data dir on the host
    wf.get_node('brain_mask_2').inputs.snglrt_bind =  [
        (out_dir, out_dir,'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        ('/homes_unix/boutinaud/ReferenceModels', '/mnt/model', 'ro'),
        ('/homes_unix/yrio/Documents/modele/ReferenceModels/model_info/brainmask', '/homes_unix/yrio/mnt/descriptor', 'ro')]
    wf.get_node('brain_mask_2').inputs.snglrt_working_directory = out_dir
    wf.get_node('brain_mask_2').inputs.descriptor = '/homes_unix/yrio/mnt/descriptor/model_info.json'
    wf.get_node('brain_mask_2').inputs.model = '/mnt/model'
    wf.get_node('brain_mask_2').inputs.snglrt_image = args.simg
    wf.get_node('brain_mask_2').inputs.out_filename = '/mnt/data/brain_mask.nii.gz'

    wf.run(plugin='Linear')


if __name__ == "__main__":
    main()
