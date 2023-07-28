#!/usr/bin/env python
# Script workflow in containeur singularity
import os
import argparse
import json
import sys
sys.path.append('/mnt/devt')
from nipype import config
config.enable_provenance()

from shivautils.workflows.dual_preprocessing import genWorkflow
from shivautils.workflows.dual_predict import genWorkflow as genWorkflowPredict
from shivautils.workflows.dual_post_processing_container import genWorkflow as genWorkflowPost
from shivautils.workflows.SWI_preprocessing import genWorkflow as genWorkflowSWI
from shivautils.workflows.SWI_predict import genWorkflow as genWorkflowPredictSWI
from shivautils.workflows.SWI_post_processing import genWorkflow as genWorkflowPostSWI


DESCRIPTION = """SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
                followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
                preprocess a lot of images at the same time. In the last step the file segmentations with the wmh and 
                pvs models are processed"""


parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument('--in', dest='input',
                    help='Folder path with files, BIDS structure folder path or JSON formatted extract of the Slicer plugin',
                    metavar='path/to/existing/folder/structure',
                    required=True)

parser.add_argument('--out', dest='output',
                    type=str,
                    help='Output folder path (nipype working directory)',
                    metavar='path/to/nipype_work_dir',
                    required=True)

parser.add_argument('--input_type',
                    type=str,
                    help='Way to grab and manage nifti files : standard, BIDS or json',
                    default='standard')


parser.add_argument('--percentile',
                    type=float,
                    default=99,
                    help='Threshold value expressed as percentile')

parser.add_argument('--final_dimensions',
                    nargs='+', type=int,
                    default=(160, 214, 176),
                    help='Final image array size in i, j, k.')

parser.add_argument('--voxels_size', nargs='+',
                    type=float,
                    default=(1.0, 1.0, 1.0),
                    help='Voxel size of final image')

parser.add_argument('--threshold', 
                    type=float,
                    default=0.5,
                    help='Value of the treshold to apply to the image')

parser.add_argument('--threshold_clusters',
                    type=float,
                    default=0.2,
                    help='Threshold to compute clusters metrics')
                    
parser.add_argument('--interpolation', 
                    type=str,
                    default='WelchWindowedSinc',
                    help='final interpolation apply to the main image')

parser.add_argument('--model',
                    default=None,
                    help='path to model descriptor')

parser.add_argument('--synthseg',
                    default=False,
                    help='Optional FreeSurfer segmentation of regions to compute metrics clusters of specific regions')

parser.add_argument('--SWI',
                    default='False',
                    help='If a second workflow for CMB is required')

parser.add_argument('--gpu',
                    type=int,
                    help='GPU to use.')

parser.add_argument('--brainmask_descriptor', 
                    type=str,
                    default='brainmask/V0/model_info.json',
                    help='brainmask descriptor file path')

parser.add_argument('--pvs_descriptor', 
                    type=str,
                    default='T1.FLAIR-PVS/V0/model_info.json',
                    help='pvs descriptor file path')

parser.add_argument('--wmh_descriptor', 
                    type=str,
                    default='T1.FLAIR-WMH/V1/model_info.json',
                    help='wmh descriptor file path')

parser.add_argument('--cmb_descriptor', 
                    type=str,
                    default='SWI-CMB/V0/model_info.json',
                    help='cmb descriptor file path')


GRAB_PATTERN = '%s/%s/*.nii*'
args = parser.parse_args()

SWI = str(args.SWI)
synthseg = args.synthseg

if args.input_type == 'json':
    with open(args.input, 'r') as json_in:
        subject_dict = json.load(json_in)
    
    out_dir = subject_dict['parameters']['out_dir']
    subject_directory = subject_dict["files_dir"]
    subject_list = os.listdir(subject_directory)
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

if args.input_type == 'standard' or args.input_type == 'BIDS':
    subject_directory = args.input
    out_dir = args.output
    subject_list = os.listdir(subject_directory)
    brainmask_descriptor = os.path.join(args.model, args.brainmask_descriptor)
    wmh_descriptor = os.path.join(args.model, args.wmh_descriptor)
    pvs_descriptor = os.path.join(args.model, args.pvs_descriptor)
    if SWI == 'True':
        cmb_descriptor = os.path.join(args.model, args.cmb_descriptor)
    else:
        cmb_descriptor = None
    

wfargs = {'SUBJECT_LIST': subject_list,
          'DATA_DIR': subject_directory,
          'BASE_DIR': out_dir,
          'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
          'WMH_DESCRIPTOR': wmh_descriptor,
          'PVS_DESCRIPTOR': pvs_descriptor,
          'CMB_DESCRIPTOR': cmb_descriptor,
          'CONTAINER': True, 
          'MODELS_PATH': args.model,
          'ANONYMIZED': False,
          'SWI': SWI,
          'INTERPOLATION': args.interpolation,
          'PERCENTILE': args.percentile,
          'THRESHOLD': args.threshold,
          'THRESHOLD_CLUSTERS' : args.threshold_clusters,
          'IMAGE_SIZE': tuple(args.final_dimensions),
          'RESOLUTION': tuple(args.voxels_size)}

if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
    os.makedirs(out_dir)
print(f'Working directory set to: {out_dir}')

wf = genWorkflow(**wfargs)

wf_predict = genWorkflowPredict(**wfargs)
wf_post = genWorkflowPost(**wfargs)

wf.base_dir = out_dir
wf_predict.base_dir = out_dir
wf_post.base_dir = out_dir

if SWI == 'True':
    swi_wf = genWorkflowSWI(**wfargs)
    swi_wf_predict = genWorkflowPredictSWI(**wfargs)
    swi_wf_post = genWorkflowPostSWI(**wfargs)
    swi_wf.base_dir = out_dir
    swi_wf_predict.base_dir = out_dir
    swi_wf_post.base_dir = out_dir


if args.input_type == 'standard' or args.input_type == 'json':    
    wf.get_node('dataGrabber').inputs.base_directory = subject_directory
    wf.get_node('dataGrabber').inputs.template = GRAB_PATTERN
    wf.get_node('dataGrabber').inputs.template_args = {'main': [['subject_id', 'main']],
                                                       'acc': [['subject_id', 'acc']]}
    if SWI == 'True':
        swi_wf.get_node('dataGrabber').inputs.base_directory = args.input
        swi_wf.get_node('dataGrabber').inputs.template = '%s/%s/*.nii*'
        swi_wf.get_node('dataGrabber').inputs.template_args = {'SWI': [['subject_id', 'SWI']]}

        swi_wf.get_node('conform').inputs.dimensions = (256, 256, 256)
        swi_wf.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
        swi_wf.get_node('conform').inputs.orientation = 'RAS'
        swi_wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)
    

wf.get_node('conform').inputs.dimensions = (256, 256, 256)
wf.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
wf.get_node('conform').inputs.orientation = 'RAS'
wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

if args.gpu:
    wf.get_node('pre_brain_mask').inputs.gpu_number = args.gpu
    wf.get_node('post_brain_mask').inputs.gpu_number = args.gpu

if args.input_type == 'BIDS':
    wf.get_node('dataGrabber').inputs.base_directory = args.input
    wf.get_node('dataGrabber').inputs.template = '%s/%s/*.nii*'
    wf.get_node('dataGrabber').inputs.template_args = {'main': [['subject_id', 'subject_id']],
                                                       'acc': [['subject_id', 'subject_id']]}
    wf.get_node('dataGrabber').inputs.field_template = {'main': '%s/anat/%s_T1_raw.nii.gz',
                                                        'acc': '%s/anat/%s_FLAIR_raw.nii.gz'}
    if SWI == 'True':
        swi_wf.get_node('dataGrabber').inputs.base_directory = args.input
        swi_wf.get_node('dataGrabber').inputs.template = '%s/%s/*.nii*'
        swi_wf.get_node('dataGrabber').inputs.template_args = {'SWI': [['subject_id', 'subject_id']]}
        swi_wf.get_node('dataGrabber').inputs.field_template = {'SWI': '%s/anat/%s_SWI_raw.nii.gz'}

        swi_wf.get_node('conform').inputs.dimensions = (256, 256, 256)
        swi_wf.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
        swi_wf.get_node('conform').inputs.orientation = 'RAS'
        swi_wf.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)


wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
wf.run(plugin='Linear')
 
wf_predict.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf.name)
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
                                                        'brainmask': os.path.join(out_dir, wf.name, '_subject_id_%s/hard_post_brain_mask/post_brain_mask_thresholded.nii.gz'),
                                                        'pre_brainmask': os.path.join(out_dir, wf.name, '_subject_id_%s/hard_brain_mask/pre_brain_maskresampled_thresholded.nii.gz'),
                                                        'T1_cropped': os.path.join(out_dir, wf.name, '_subject_id_%s/main_final_intensity_normalization/%s_T1_raw_trans_img_normalized.nii.gz'),
                                                        'FLAIR_cropped': os.path.join(out_dir, wf.name, '_subject_id_%s/acc_final_intensity_normalization/main_to_acc__Warped_img_normalized.nii.gz'),
                                                        'T1_conform': os.path.join(out_dir, wf.name, '_subject_id_%s/conform/%s_T1_rawresampled.nii.gz'),
                                                        'BBOX1': os.path.join(out_dir, wf.name, '_subject_id_%s/crop/bbox1.txt'),
                                                        'BBOX2': os.path.join(out_dir, wf.name, '_subject_id_%s/crop/bbox2.txt'),
                                                        'CDG_IJK': os.path.join(out_dir, wf.name, '_subject_id_%s/crop/cdg_ijk.txt'),
                                                        'sum_preproc_wf': os.path.join(out_dir, wf.name, 'graph.svg')}

wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
wf_post.run(plugin='Linear')

if SWI == 'True':
    swi_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    swi_wf.run(plugin='Linear')

    swi_wf_predict.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, swi_wf.name)
    swi_wf_predict.get_node('dataGrabber').inputs.template = GRAB_PATTERN
    swi_wf_predict.get_node('dataGrabber').inputs.template_args = {'SWI': [['subject_id']]}
    swi_wf_predict.get_node('dataGrabber').inputs.field_template = {'SWI': '_subject_id_%s/final_intensity_normalization/*.nii.gz'}

    swi_wf_predict.run(plugin='Linear')

    swi_wf_post.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, swi_wf_predict.name)
    swi_wf_post.get_node('dataGrabber').inputs.template = GRAB_PATTERN
    swi_wf_post.get_node('dataGrabber').inputs.template_args = {'segmentation_cmb': [['subject_id']],
                                                                'brainmask': [['subject_id']],
                                                                'pre_brainmask': [['subject_id']],
                                                                'SWI_cropped': [['subject_id']],
                                                                'SWI_conform': [['subject_id']],
                                                                'BBOX1': [['subject_id']],
                                                                'BBOX2': [['subject_id']],
                                                                'CDG_IJK': [['subject_id']],
                                                                'sum_preproc_wf': [[]]}
    swi_wf_post.get_node('dataGrabber').inputs.field_template = {'segmentation_cmb': '_subject_id_%s/predict_cmb/cmb_map.nii.gz',
                                                                'brainmask': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/hard_post_brain_mask/post_brain_mask_thresholded.nii.gz'),
                                                                'pre_brainmask': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/hard_brain_mask/pre_brain_maskresampled_thresholded.nii.gz'),
                                                                'SWI_cropped': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/final_intensity_normalization/*.nii.gz'),
                                                                'SWI_conform': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/conform/*.nii.gz'),
                                                                'BBOX1': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/crop/bbox1.txt'),
                                                                'BBOX2': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/crop/bbox2.txt'),
                                                                'CDG_IJK': os.path.join(out_dir, swi_wf.name, '_subject_id_%s/crop/cdg_ijk.txt'),
                                                                'sum_preproc_wf': os.path.join(out_dir, swi_wf.name, 'graph.svg')}

    swi_wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    swi_wf_post.run(plugin='Linear')
