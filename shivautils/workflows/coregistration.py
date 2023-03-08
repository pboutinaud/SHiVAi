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

from shivautils.interfaces.image import Normalization


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~'))
}


def as_list(input):
    return [input]
    

def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("shiva_coregister_normalize")
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
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'target': [['subject_id', 'target']],
                                        'source': [['subject_id', 'source']],
                                        'brainmask': [['subject_id', 'brainmask']]}

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
    coreg = Node(ants.AI(),
                 name='coregister')
    coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}
    coreg.inputs.transform = ('Rigid',
                               0.2)
    coreg.inputs.metric = ('Mattes', 128,
                           'Regular', 0.5)
    coreg.inputs.search_factor = (10, 0.1)
    coreg.inputs.num_threads = 8
    coreg.inputs.output_transform = "source_to_target_affine.mat" 
    coreg.inputs.verbose = True

    # ici les images cible (T1 crop) et brain_mask proviennent du datagrabber (préexistantes) 
    workflow.connect(dicom2nifti_source, 'reoriented_files',
                     coreg, 'moving_image')
    workflow.connect(datagrabber, "target",
                     coreg, 'fixed_image')
    workflow.connect(datagrabber, 'brainmask',
                     coreg, 'fixed_image_mask')


    # write coregistered source image to cropped space
    apply_coreg = Node(ants.ApplyTransforms(), name="apply_coreg")
    apply_coreg.inputs.interpolation = 'LanczosWindowedSinc'
    workflow.connect(coreg, ('output_transform', as_list), apply_coreg, 'transforms' )
    workflow.connect(dicom2nifti_source, 'reoriented_files', apply_coreg, 'input_image')

    workflow.connect(datagrabber, 'target', apply_coreg, 'reference_image')

    # write mask to source in native space
    mask_to_source = Node(ants.ApplyTransforms(), name="mask_to_source")
    mask_to_source.inputs.interpolation = 'NearestNeighbor'
    mask_to_source.inputs.invert_transform_flags = [True]
    
    workflow.connect(coreg, ('output_transform', as_list),
                     mask_to_source, 'transforms')
    workflow.connect(datagrabber, 'brainmask',
                     mask_to_source, 'input_image')
    workflow.connect(dicom2nifti_source, 'reoriented_files',
                     mask_to_source, 'reference_image')


    ##########################################################################

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 1)
    coreg_norm =  Node(Normalization(percentile = 99), name="coreg_final_intensity_normalization")
    workflow.connect(apply_coreg, 'output_image',
    		         coreg_norm, 'input_image')
    workflow.connect(datagrabber, 'target',
    	 	         coreg_norm, 'brain_mask')
    

    # defaced cropped acc image (ENDPOINT 2)
    deface_cropped_coreg = Node(Quickshear(), name='deface_coreg_final')
    workflow.connect([
        (coreg_norm, deface_cropped_coreg, 
                        [('intensity_normalized', 'in_file')]),
        (datagrabber, deface_cropped_coreg,
                        [('brainmask', 'mask_file')]),
        ])
    
    # defaced source image (ENDPOINT 3)
    deface_source = Node(Quickshear(), name='deface_source')
    workflow.connect([
        (dicom2nifti_source, deface_source, [('reoriented_files', 'in_file')]),
        (mask_to_source, deface_source, [('output_image', 'mask_file')]),
        ])

    return workflow


if __name__ == '__target__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')
