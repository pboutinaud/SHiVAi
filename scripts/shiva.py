#!/usr/bin/env python
"""Workflow script for singularity container"""
from shivautils.workflows.SWI_postprocessing import genWorkflow as genWorkflowPostSWI
from shivautils.workflows.SWI_predict import genWorkflow as genWorkflowPredictSWI
from shivautils.workflows.SWI_preprocessing import genWorkflow as genWorkflowSWI
from shivautils.workflows.dual_post_processing_container import genWorkflow as genWorkflowPost
from shivautils.workflows.dual_predict import genWorkflow as genWorkflowPredict
from shivautils.workflows.dual_preprocessing import genWorkflow
from nipype import config
import os
import argparse
import json
import sys

sys.path.append('/mnt/devt')

config.enable_provenance()


DESCRIPTION = """SHIVA pipeline for deep-learning imaging biomarkers computation. Performs resampling and coregistration 
                of a set of structural NIfTI head image, followed by intensity normalization, and cropping centered on the brain.
                A nipype workflow is used to preprocess a lot of images at the same time.
                The segmentations from the wmh, cmb and pvs models are generated depending on the inputs. A Report is generated.
                
                Input data can be staged in BIDS or a simplified file arborescence, or described with a JSON file (for the 3D Slicer extension)."""


def shivaParser():
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
                        choices=['standard', 'BIDS', 'json'],
                        help="Way to grab and manage nifti files : 'standard', 'BIDS' or 'json'",
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
                        help='final interpolation apply to the t1 image')

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
                        help='Force GPU to use (default is taken from "CUDA_VISIBLE_DEVICES").')

    parser.add_argument('--brainmask_descriptor',
                        type=str,
                        # default='brainmask/V0/model_info.json',
                        help='brainmask descriptor file path',
                        required=True)

    parser.add_argument('--pvs_descriptor',
                        type=str,
                        # default='T1.FLAIR-PVS/V0/model_info.json',
                        help='pvs descriptor file path',
                        required=True)

    parser.add_argument('--wmh_descriptor',
                        type=str,
                        # default='T1.FLAIR-WMH/V1/model_info.json',
                        help='wmh descriptor file path',
                        required=True)

    parser.add_argument('--cmb_descriptor',
                        type=str,
                        # default='SWI-CMB/V0/model_info.json',
                        help='cmb descriptor file path',
                        required=True)
    return parser


def main():

    parser = shivaParser()
    args = parser.parse_args()

    # GRAB_PATTERN = '%s/%s/*.nii*'  # Now directly specified in the workflow generators
    SWI = str(args.SWI)
    synthseg = args.synthseg

    if args.input_type == 'json':  # TODO: Check definition of brainmask_descriptor, wmh_descriptor and pvs_descriptor
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
              'DATA_DIR': subject_directory,  # Default base_directory for the dataGrabber
              'BASE_DIR': out_dir,  # Default base_dir for each workflow
              'INPUT_TYPE': args.input_type,
              'WF_DIRS': {'preproc': 'shiva_preprocessing_dual', 'pred': 'dual_predictor_workflow'},  # Name of the preprocessing wf, used to link wf for now
              'BRAINMASK_DESCRIPTOR': brainmask_descriptor,
              'WMH_DESCRIPTOR': wmh_descriptor,
              'PVS_DESCRIPTOR': pvs_descriptor,
              'CMB_DESCRIPTOR': cmb_descriptor,
              'CONTAINER': True,  # TODO: Change so that we can use this script without singularity
              'MODELS_PATH': args.model,
              'ANONYMIZED': False,  # TODO: Why False though?
              'SWI': SWI,
              'INTERPOLATION': args.interpolation,
              'PERCENTILE': args.percentile,
              'THRESHOLD': args.threshold,
              'THRESHOLD_CLUSTERS': args.threshold_clusters,
              'IMAGE_SIZE': tuple(args.final_dimensions),
              'RESOLUTION': tuple(args.voxels_size),
              'ORIENTATION': 'RAS'}

    if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    print(f'Working directory set to: {out_dir}')

    wf_preproc = genWorkflow(**wfargs)
    wf_predict = genWorkflowPredict(**wfargs)
    wf_post = genWorkflowPost(**wfargs)

    # If necessary to modify defaults:
    # wf_preproc.get_node('conform').inputs.dimensions = (256, 256, 256)
    # wf_preproc.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
    # wf_preproc.get_node('conform').inputs.orientation = 'RAS'
    # wf_preproc.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

    if args.gpu:
        wf_preproc.get_node('pre_brain_mask').inputs.gpu_number = args.gpu
        wf_preproc.get_node('post_brain_mask').inputs.gpu_number = args.gpu

    wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    # wf_predict.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf_preproc.name)
    # wf_post.get_node('dataGrabber').inputs.base_directory = os.path.join(out_dir, wf_predict.name)
    wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}  # TODO: Is there even the possibility that it is True?

    wf_preproc.run(plugin='Linear')
    wf_predict.run(plugin='Linear')
    wf_post.run(plugin='Linear')

    if SWI == 'True':
        wfargs.update({'WF_SWI_DIRS': {'preproc': 'shiva_preprocessing_swi', 'pred': 'SWI_predictor_workflow'}})
        swi_wf_preproc = genWorkflowSWI(**wfargs)
        swi_wf_predict = genWorkflowPredictSWI(**wfargs)
        swi_wf_post = genWorkflowPostSWI(**wfargs)

        # If necessary to modify defaults:
        # swi_wf_preproc.get_node('conform').inputs.dimensions = (256, 256, 256)
        # swi_wf_preproc.get_node('conform').inputs.voxel_size = tuple(args.voxels_size)
        # swi_wf_preproc.get_node('conform').inputs.orientation = 'RAS'
        # swi_wf_preproc.get_node('crop').inputs.final_dimensions = tuple(args.final_dimensions)

        swi_wf_preproc.config['execution'] = {'remove_unnecessary_outputs': 'False'}
        swi_wf_post.config['execution'] = {'remove_unnecessary_outputs': 'False'}

        swi_wf_preproc.run(plugin='Linear')
        swi_wf_predict.run(plugin='Linear')
        swi_wf_post.run(plugin='Linear')


if __name__ == "__main__":
    main()
