#!/usr/bin/env python
# Script workflow in containeur singularity
import os
import argparse
import json
from nipype import Node, Workflow


from shivautils.workflows.dual_preprocessing import genWorkflow
from shivautils.workflows.dual_predict import genWorkflow as genWorkflowPredict
from shivautils.workflows.dual_post_processing import genWorkflow as genWorkflowPost

DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                preprocess a lot of images at the same time."""
                 
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


parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument('--in', dest='input',
                    help='path folder dataset',
                    metavar='path input',
                    required=True)

parser.add_argument('--out', dest='output',
                    type=str,
                    help='Output folder path (nipype working directory)',
                    metavar='path/to/nipype_work_dir',
                    required=True)

parser.add_argument('--percentile',
                    type=float,
                    default=99,
                    help='Threshold value expressed as percentile')

parser.add_argument('--threshold',
                    type=float,
                    default=0.5,
                    help='Value of the treshold to apply to the image')

parser.add_argument('--final_dimensions',
                    nargs='+', type=int,
                    default=(160, 214, 176),
                    help='Final image array size in i, j, k.')

parser.add_argument('--voxels_size', nargs='+',
                    type=float,
                    default=(1.0, 1.0, 1.0),
                    help='Voxel size of final image')
                    
parser.add_argument('--model',
                    default=None,
                    help='path to model descriptor')

parser.add_argument('--gpu',
                    type=int,
                    help='GPU to use.')



args = parser.parse_args()

GRAB_PATTERN = '%s/%s/*.nii*'
subject_directory = args.input
subject_list = os.listdir(subject_directory)

out_dir = args.output
wfargs = {'SUBJECT_LIST': subject_list,
          'DATA_DIR': subject_directory,
          'BASE_DIR': out_dir,
          'BRAINMASK_DESCRIPTOR': os.path.join(args.model, 'brainmask/V0/model_info.json'),
          'WMH_DESCRIPTOR': os.path.join(args.model, 'T1.FLAIR-WMH/V1/model_info.json'),
          'PVS_DESCRIPTOR': os.path.join(args.model, 'T1.FLAIR-PVS/V0/model_info.json'),
          'MODELS_PATH': args.model,
          'CONTAINER': False,
          'ANONYMIZED': False,
          'INTERPOLATION': 'WelchWindowedSinc',
          'THRESHOLD_CLUSTERS': 0.2,
          'PERCENTILE': args.percentile,
          'THRESHOLD': args.threshold,
          'IMAGE_SIZE': tuple(args.final_dimensions),
          'RESOLUTION': tuple(args.voxels_size)}

if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
    os.makedirs(out_dir)
print(f'Working directory set to: {out_dir}')

wf_preprocessing = genWorkflow(**wfargs)

wf_predict = genWorkflowPredict(**wfargs)
wf_post = genWorkflowPost(**wfargs)

wf_preprocessing.base_dir = out_dir
wf_predict.base_dir = out_dir
wf_post.base_dir = out_dir

wf_preprocessing.get_node('dataGrabber').inputs.base_directory = subject_directory
wf_preprocessing.get_node('dataGrabber').inputs.template = GRAB_PATTERN
wf_preprocessing.get_node('dataGrabber').inputs.template_args = {'main': [['subject_id', 'main']],
                                                                 'acc': [['subject_id', 'acc']]}
wf_preprocessing.get_node('conform').inputs.dimensions = (256, 256, 256)
wf_preprocessing.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
wf_preprocessing.get_node('conform').inputs.orientation = 'RAS'
wf_preprocessing.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

wf_preprocessing.config['execution'] = {'remove_unnecessary_outputs': 'False'}
wf_preprocessing.run(plugin='Linear')

wf_predict.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf_preprocessing.name)
wf_predict.get_node('dataGrabber').inputs.template = GRAB_PATTERN
wf_predict.get_node('dataGrabber').inputs.template_args = {'main': [['subject_id', 'subject_id']],
                                                           'acc': [['subject_id']]}
wf_predict.get_node('dataGrabber').inputs.field_template = {'main': '_subject_id_%s/main_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz',
                                                            'acc': '_subject_id_%s/acc_final_intensity_normalization/main_to_acc__Warped_img_normalized.nii.gz'}

wf_predict.run(plugin='Linear')

wf_post.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf_predict.name)
wf_post.get_node('dataGrabber').inputs.template = GRAB_PATTERN
wf_post.get_node('dataGrabber').inputs.template_args = {'segmentation_pvs': [['subject_id']],
                                                        'segmentation_wmh': [['subject_id']],
                                                        'brainmask': [['subject_id']],
                                                        'pre_brainmask': [['subject_id']],
                                                        'T1_cropped': [['subject_id', 'subject_id']],
                                                        'FLAIR_cropped': [['subject_id']],
                                                        'T1_conform': [['subject_id', 'subject_id']],
                                                        'BBOX1': [['subject_id']],
                                                        'BBOX2': [['subject_id']],
                                                        'CDG_IJK': [['subject_id']],
                                                        'sum_preproc_wf': [[]]}
wf_post.get_node('dataGrabber').inputs.field_template = {'segmentation_pvs': '_subject_id_%s/predict_pvs/pvs_map.nii.gz',
                                                         'segmentation_wmh': '_subject_id_%s/predict_wmh/wmh_map.nii.gz',
                                                         'brainmask': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/hard_post_brain_mask/post_brain_mask_thresholded.nii.gz'),
                                                         'pre_brainmask': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/hard_brain_mask/pre_brain_maskresampled_thresholded.nii.gz'),
                                                         'T1_cropped': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/main_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz'),
                                                         'FLAIR_cropped': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/acc_final_intensity_normalization/main_to_acc__Warped_img_normalized.nii.gz'),
                                                         'T1_conform': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/conform/%s_T1_rawresampled.nii.gz'),
                                                         'BBOX1': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/crop/bbox1.txt'),
                                                         'BBOX2': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/crop/bbox2.txt'),
                                                         'CDG_IJK': os.path.join(out_dir, wf_preprocessing.name, '_subject_id_%s/crop/cdg_ijk.txt'),
                                                         'sum_preproc_wf': os.path.join(out_dir, wf_preprocessing.name, 'graph.svg')
}

wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
wf_post.run(plugin='Linear')