#!/usr/bin/env python
"""Nipype workflow for DICOM to NII image conversion, conformation and preparation before deep
   learning, with accessory image coregistation to cropped space (through ANTS),
   and defacing of native and final images. This also handles back-registration from
   conformed-crop to main.
   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.utility import IdentityInterface
from pyplm.interfaces.shiva import Predict
from nipype.interfaces.quickshear import Quickshear

from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))
}


def as_list(input):
    return [input]
    

def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_predictor_preprocessing_dual")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['main', 'acc']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'main': [['subject_id', 'main']],
                                        'acc': [['subject_id', 'acc']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    # dcm2nii file conversion
    dicom2nifti_main = Node(Dcm2nii(), name="dicom2nifti_main")
    dicom2nifti_main.inputs.anonymize = True
    dicom2nifti_main.inputs.collapse_folders = True
    dicom2nifti_main.inputs.convert_all_pars = True
    dicom2nifti_main.inputs.environ = {}
    dicom2nifti_main.inputs.events_in_filename = True
    dicom2nifti_main.inputs.gzip_output = False
    dicom2nifti_main.inputs.id_in_filename = False
    dicom2nifti_main.inputs.nii_output = True
    dicom2nifti_main.inputs.protocol_in_filename = True
    dicom2nifti_main.inputs.reorient_and_crop = True
    dicom2nifti_main.inputs.source_in_filename = False

    workflow.connect(datagrabber, "main", dicom2nifti_main, "source_names")

    dicom2nifti_acc = Node(Dcm2nii(), name="dicom2nifti_acc")
    dicom2nifti_acc.inputs.anonymize = True
    dicom2nifti_acc.inputs.collapse_folders = True
    dicom2nifti_acc.inputs.convert_all_pars = True
    dicom2nifti_acc.inputs.environ = {}
    dicom2nifti_acc.inputs.events_in_filename = True
    dicom2nifti_acc.inputs.gzip_output = False
    dicom2nifti_acc.inputs.id_in_filename = False
    dicom2nifti_acc.inputs.nii_output = True
    dicom2nifti_acc.inputs.protocol_in_filename = True
    dicom2nifti_acc.inputs.reorient_and_crop = False
    dicom2nifti_acc.inputs.source_in_filename = False

    workflow.connect(datagrabber, "acc", dicom2nifti_acc, "source_names")

    # conform main to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = (1.0, 1.0, 1.0)
    conform.inputs.orientation = 'RAS'

    workflow.connect(dicom2nifti_main, 'reoriented_files', conform, 'img')
    
    # preconform main to 1 mm isotropic, freesurfer-style
    preconform = Node(Conform(),
                   name="preconform")
    preconform.inputs.dimensions = (160, 214, 176)
    preconform.inputs.orientation = 'RAS'

    workflow.connect(dicom2nifti_main, 'reoriented_files', preconform, 'img')

    
    # normalize intensities between 0 and 1 for Tensorflow initial brain mask extraction:
    # identify brain to define image cropping region.
    preconf_normalization = Node(Normalization(percentile = 99), name="preconform_intensity_normalization")
    workflow.connect(preconform, 'resampled',
                     preconf_normalization, 'input_image')

    # brain mask from tensorflow
    pre_brain_mask = Node(Predict(), "pre_brain_mask")
    pre_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 1 --partition GPU'}
    pre_brain_mask.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        ('/homes_unix/boutinaud/ReferenceModels', '/mnt/model', 'ro')]
    pre_brain_mask.inputs.model = '/mnt/model'
    pre_brain_mask.inputs.snglrt_enable_nvidia = True
    pre_brain_mask.inputs.descriptor = kwargs['DESCRIPTOR']
    pre_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict.sif'
    pre_brain_mask.inputs.out_filename = '/mnt/data/pre_brain_mask.nii.gz'
    workflow.connect(preconf_normalization, 'intensity_normalized',
                     pre_brain_mask, 't1')
                     
    
    # send mask from preconformed space to 
    # conformed space 256 256 256 , same as anatomical conformed image
    unconform = Node(Conform(),
                name="unpreconform")
    unconform.inputs.dimensions = (256, 256, 256)
    unconform.inputs.voxel_size = (1.0, 1.0, 1.0)
    unconform.inputs.orientation = 'RAS'

    workflow.connect(pre_brain_mask, 'segmentation', unconform, 'img')
          
    # binarize unpreconformed brain mask
    hard_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_brain_mask")
    workflow.connect(unconform, 'resampled', hard_brain_mask, 'img')
    
    # normalize intensities between 0 and 1 for Tensorflow
    post_normalization = Node(Normalization(percentile = 99), name="post_intensity_normalization")
    workflow.connect(conform, 'resampled',
                     post_normalization, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
    	 	         post_normalization, 'brain_mask')
    
    # crop main centered on mask origin
    crop_normalized = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop_normalized")
    workflow.connect(post_normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop_normalized, 'roi_mask')
                     
    # crop raw
    # crop main centered on mask
    crop = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    workflow.connect(hard_brain_mask, 'thresholded',
                     crop, 'roi_mask')
                     
    # brain mask from tensorflow
    post_brain_mask = Node(Predict(), "post_brain_mask")
    post_brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 1 --partition GPU'}
    post_brain_mask.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        ('/homes_unix/boutinaud/ReferenceModels', '/mnt/model', 'ro')]
    post_brain_mask.inputs.model = '/mnt/model'
    post_brain_mask.inputs.snglrt_enable_nvidia = True
    post_brain_mask.inputs.descriptor = kwargs['DESCRIPTOR']
    post_brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict.sif'
    post_brain_mask.inputs.out_filename = '/mnt/data/post_brain_mask.nii.gz'
    workflow.connect(crop_normalized, 'cropped',
                     post_brain_mask, 't1')

    # binarize post brain mask
    hard_post_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_post_brain_mask")
    workflow.connect(post_brain_mask, 'segmentation', hard_post_brain_mask, 'img')

    # compute 6-dof coregistration parameters of accessory scan 
    # to cropped  main image
    coreg = Node(ants.Registration(),
                 name='coregister')
    coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    coreg.inputs.transforms = ['Rigid']
    coreg.inputs.transform_parameters = [(0.1,)]
    coreg.inputs.metric = ['MI']
    coreg.inputs.radius_or_number_of_bins = [64]
    coreg.inputs.interpolation = 'WelchWindowedSinc'
    coreg.inputs.shrink_factors = [[8,4,2,1]]
    coreg.inputs.output_warped_image = True
    coreg.inputs.smoothing_sigmas = [[3,2,1,0]]
    coreg.inputs.num_threads = 8
    coreg.inputs.number_of_iterations = [[1000,500,250,125]]
    coreg.inputs.sampling_strategy = ['Regular']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.output_transform_prefix = "main_to_acc_"
    coreg.inputs.verbose = True
    coreg.inputs.winsorize_lower_quantile = 0.005
    coreg.inputs.winsorize_upper_quantile = 0.995

    workflow.connect(dicom2nifti_acc, 'reoriented_files',
                     coreg, 'moving_image')
    workflow.connect(crop, 'cropped',
                     coreg, 'fixed_image')
    workflow.connect(hard_post_brain_mask, ('thresholded', as_list),
                     coreg, 'fixed_image_masks')
                
    # compute 3-dof (translations) coregistration parameters of cropped to native main
    crop_to_main = Node(ants.Registration(),
                 name='crop_to_main')
    crop_to_main.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    crop_to_main.inputs.transforms = ['Rigid']
    crop_to_main.inputs.restrict_deformation=[[1,0,0,],[1,0,0,],[1,0,0]]
    crop_to_main.inputs.transform_parameters = [(0.1,)]
    crop_to_main.inputs.metric = ['MI']
    crop_to_main.inputs.radius_or_number_of_bins = [64]
    crop_to_main.inputs.shrink_factors = [[8,4,2,1]]
    crop_to_main.inputs.output_warped_image = False
    crop_to_main.inputs.smoothing_sigmas = [[3,2,1,0]]
    crop_to_main.inputs.num_threads = 8
    crop_to_main.inputs.number_of_iterations = [[1000,500,250,125]]
    crop_to_main.inputs.sampling_strategy = ['Regular']
    crop_to_main.inputs.sampling_percentage = [0.25]
    crop_to_main.inputs.output_transform_prefix = "cropped_to_source_"
    crop_to_main.inputs.verbose = True
    crop_to_main.inputs.winsorize_lower_quantile = 0.0
    crop_to_main.inputs.winsorize_upper_quantile = 1.0

    workflow.connect(dicom2nifti_main, 'reoriented_files',
                     crop_to_main, 'fixed_image')
    workflow.connect(crop, 'cropped',
                     crop_to_main, 'moving_image')

    

    # write brain seg on main in native space
    mask_to_main = Node(ants.ApplyTransforms(), name="mask_to_main")
    mask_to_main.inputs.interpolation = 'NearestNeighbor'
    workflow.connect(crop_to_main, 'forward_transforms', mask_to_main, 'transforms' )
    workflow.connect(hard_post_brain_mask, 'thresholded', mask_to_main, 'input_image')
    workflow.connect(dicom2nifti_main, 'reoriented_files', mask_to_main, 'reference_image')
    
    # write mask to acc in native space
    mask_to_acc = Node(ants.ApplyTransforms(), name="mask_to_acc")
    mask_to_acc.inputs.interpolation = 'NearestNeighbor'
    mask_to_acc.inputs.invert_transform_flags = [True]
    
    workflow.connect(coreg, 'forward_transforms',
                     mask_to_acc, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     mask_to_acc, 'input_image')
    workflow.connect(dicom2nifti_acc, 'reoriented_files',
                     mask_to_acc, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    main_norm =  Node(Normalization(percentile = 99), name="main_final_intensity_normalization")
    workflow.connect(crop, 'cropped',
    		         main_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
    	 	         main_norm, 'brain_mask')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 2)
    acc_norm =  Node(Normalization(percentile = 99), name="acc_final_intensity_normalization")
    workflow.connect(coreg, 'warped_image',
    		         acc_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
    	 	         acc_norm, 'brain_mask')
 	         
    # defaced native main image (ENDPOINT 3)
    deface_main = Node(Quickshear(), name='deface_native_main')
    workflow.connect([
        (dicom2nifti_main, deface_main, [('reoriented_files', 'in_file')]),
        (mask_to_main, deface_main, [('output_image', 'mask_file')]),
        ])

    # defaced native acc image (ENDPOINT 4)
    deface_acc = Node(Quickshear(), name='deface_native_acc')
    workflow.connect([
        (dicom2nifti_acc, deface_acc, [('reoriented_files', 'in_file')]),
        (mask_to_acc, deface_acc, [('output_image', 'mask_file')]),
        ])
        
    # defaced cropped main image (ENDPOINT 5)
    deface_cropped_main = Node(Quickshear(), name='deface_final_norm_cropped_main')
    workflow.connect([
        (main_norm, deface_cropped_main, [('intensity_normalized', 'in_file')]),
        (hard_post_brain_mask, deface_cropped_main, [('thresholded', 'mask_file')]),
        ])

    # defaced cropped acc image (ENDPOINT 6)
    deface_cropped_acc = Node(Quickshear(), name='deface_final_norm_cropped_acc')
    workflow.connect([
        (acc_norm, deface_cropped_acc, 
                        [('intensity_normalized', 'in_file')]),
        (hard_post_brain_mask, deface_cropped_acc,
                        [('thresholded', 'mask_file')]),
        ])

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
