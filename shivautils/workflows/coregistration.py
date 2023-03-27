#!/usr/bin/env python
"""Nipype workflow for DICOM to NII image conversion, coregistration to a NIfTI target and intensity normalization, and quickshear.
resulting image is coregistered to T1 and ready for use in predictor. Meant to be used with SWI / T2GRE images.
"""
import os
from nipype.interfaces import ants
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.quickshear import Quickshear

from shivautils.interfaces.image import Normalization, Crop, Conform


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_coregister_normalize_t2gre")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['target', 'source', 'brainmask']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/%s'
    datagrabber.inputs.template_args = {'target': [['subject_id', 'target', '*.nii*']],
                                        'source': [['subject_id', 'source','*.dcm']],
                                        'brainmask': [['subject_id', 'brainmask','*.nii*']]}

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
 
    dicom2nifti_source = Node(Dcm2nii(), name="dicom2nifti_source")
    dicom2nifti_source.inputs.anonymize = True
    dicom2nifti_source.inputs.collapse_folders = True
    dicom2nifti_source.inputs.convert_all_pars = True
    dicom2nifti_source.inputs.environ = {}
    dicom2nifti_source.inputs.events_in_filename = True
    dicom2nifti_source.inputs.gzip_output = True
    dicom2nifti_source.inputs.id_in_filename = True
    dicom2nifti_source.inputs.nii_output = True
    dicom2nifti_source.inputs.protocol_in_filename = True
    dicom2nifti_source.inputs.reorient_and_crop = True
    dicom2nifti_source.inputs.source_in_filename = False

    workflow.connect(datagrabber, "source", dicom2nifti_source, "source_names")

    #####################################################################
 
    # Coregistration
    # compute 6-dof coregistration parameters of accessory scan 
    # to cropped  target image
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
    coreg.inputs.sigma_units = ['mm']
    coreg.inputs.number_of_iterations = [[1000,500,250,125]]
    coreg.inputs.sampling_strategy = ['Regular']
    coreg.inputs.sampling_percentage = [0.5]
    coreg.inputs.output_transform_prefix = "source_to_cropped_main_"
    coreg.inputs.verbose = True
    coreg.inputs.winsorize_lower_quantile = 0.005
    coreg.inputs.winsorize_upper_quantile = 0.995

    # ici les images cible (T1 crop) et brain_mask proviennent du datagrabber (préexistantes) 
    workflow.connect(dicom2nifti_source, 'reoriented_files',
                     coreg, 'moving_image')
    workflow.connect(datagrabber, "target",
                     coreg, 'fixed_image')

    # conform main to 1 mm isotropic, freesurfer-style
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = (1.0, 1.0, 1.0)
    conform.inputs.orientation = 'RAS'
    conform.inputs.order = 3

    workflow.connect(dicom2nifti_source, 'reoriented_files', conform, 'img')

    # write mask to source in native space
    mask_to_source = Node(ants.ApplyTransforms(), name="mask_to_source")
    mask_to_source.inputs.interpolation = 'NearestNeighbor'
    mask_to_source.inputs.invert_transform_flags = [True]

    workflow.connect(coreg, 'forward_transforms',
                     mask_to_source, 'transforms')
    workflow.connect(datagrabber, 'brainmask',
                     mask_to_source, 'input_image')
    workflow.connect(dicom2nifti_source, 'reoriented_files',
                     mask_to_source, 'reference_image')
    
    # write mask to source in native space
    mask_to_conformed_source  = Node(ants.ApplyTransforms(), name="mask_to_conformed_source")
    mask_to_conformed_source.inputs.interpolation = 'NearestNeighbor'
    mask_to_conformed_source.inputs.invert_transform_flags = [True]

    workflow.connect(coreg, 'forward_transforms',
                     mask_to_conformed_source, 'transforms')
    workflow.connect(datagrabber, 'brainmask',
                     mask_to_conformed_source, 'input_image')
    workflow.connect(conform, 'resampled',
                     mask_to_conformed_source, 'reference_image')

    # normalize intensities within mask
    normalization = Node(Normalization(percentile = 99), name="intensity_normalization")
    workflow.connect(conform, 'resampled',
                     normalization, 'input_image')
    workflow.connect(mask_to_conformed_source, 'output_image',
    	 	         normalization, 'brain_mask')


    # crop the mask
    crop_mask = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop_mask")
    workflow.connect(mask_to_conformed_source, 'output_image',
                     crop_mask, 'apply_to')
    workflow.connect(mask_to_conformed_source, 'output_image',
                     crop_mask, 'roi_mask')

    # crop to source coreg
    # compute 3-dof coregistration parameters of cropped to source scan
    # to cropped  target image
    uncrop_reg = Node(ants.Registration(),
                 name='uncrop_reg')
    uncrop_reg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    uncrop_reg.inputs.transforms = ['Rigid']
    uncrop_reg.inputs.restrict_deformation=[[1,0,0,],[1,0,0,],[1,0,0]]
    uncrop_reg.inputs.transform_parameters = [(0.1,)]
    uncrop_reg.inputs.metric = ['MI']
    uncrop_reg.inputs.radius_or_number_of_bins = [64]
    uncrop_reg.inputs.interpolation = 'WelchWindowedSinc'
    uncrop_reg.inputs.shrink_factors = [[8,4,2,1]]
    uncrop_reg.inputs.output_warped_image = True
    uncrop_reg.inputs.smoothing_sigmas = [[3,2,1,0]]
    uncrop_reg.inputs.num_threads = 8
    uncrop_reg.inputs.number_of_iterations = [[1000,500,250,125]]
    uncrop_reg.inputs.sampling_strategy = ['Regular']
    uncrop_reg.inputs.sampling_percentage = [0.25]
    uncrop_reg.inputs.output_transform_prefix = "cropped_to_source_"
    uncrop_reg.inputs.verbose = True
    uncrop_reg.inputs.winsorize_lower_quantile = 0.0
    uncrop_reg.inputs.winsorize_upper_quantile = 1.0

    # see below endpoint 1 for wiring

    ##########################################################################
    # crop normalized source image (ENDPOINT 1)
    # crop main centered on xyz origin (affine matrix)
    crop_normalized = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop_normalized")
    workflow.connect(normalization, 'intensity_normalized',
                     crop_normalized, 'apply_to')
    workflow.connect(mask_to_conformed_source, 'output_image',
                     crop_normalized, 'roi_mask')
    # wiring uncrop coreg
    workflow.connect(dicom2nifti_source, 'reoriented_files',
                     uncrop_reg, 'fixed_image')
    workflow.connect(crop_normalized, "cropped",
                     uncrop_reg, 'moving_image')


    # defaced cropped source image (ENDPOINT 2)
    deface_cropped_normalized = Node(Quickshear(), name='deface_cropped_final')
    workflow.connect([
        (crop_normalized, deface_cropped_normalized, 
                        [('cropped', 'in_file')]),
        (crop_mask, deface_cropped_normalized,
                        [('cropped', 'mask_file')]),
        ])
    
    # defaced non-preprocessed source image (ENDPOINT 3)
    deface_source = Node(Quickshear(), name='deface_source')
    workflow.connect([
        (dicom2nifti_source, deface_source, [('reoriented_files', 'in_file')]),
        (mask_to_source, deface_source, [('output_image', 'mask_file')]),
        ])

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')