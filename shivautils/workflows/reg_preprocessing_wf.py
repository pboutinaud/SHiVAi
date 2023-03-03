"""Nipype workflow for DICOM to NII image conversion, conformation and preparation before deep
   learning, with adaptation with preprocessed image T1. 
   """
import os
from nipype.interfaces import ants
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.utility import IdentityInterface
from pyplm.interfaces.shiva import Predict

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
    workflow = Workflow("preprocessing")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                            name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['main', 'acc', 'brainmask']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'
    datagrabber.inputs.template_args = {'main': [['subject_id', 'main']],
                                        'acc': [['subject_id', 'acc']],
                                        'brainmask': [['subject_id', 'brainmask']]}

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



    # Il faut que j'ajoute la même chose également pour importer le brain_mask
    # et faire toute les adaptations nécessaires au niveau du datagrabber

    dicom2nifti_brainmask = Node(Dcm2nii(), name="dicom2nifti_brainmask")
    dicom2nifti_brainmask.inputs.anonymize = True
    dicom2nifti_brainmask.inputs.collapse_folders = True
    dicom2nifti_brainmask.inputs.convert_all_pars = True
    dicom2nifti_brainmask.inputs.environ = {}
    dicom2nifti_brainmask.inputs.events_in_filename = True
    dicom2nifti_brainmask.inputs.gzip_output = False
    dicom2nifti_brainmask.inputs.id_in_filename = False
    dicom2nifti_brainmask.inputs.nii_output = True
    dicom2nifti_brainmask.inputs.protocol_in_filename = True
    dicom2nifti_brainmask.inputs.reorient_and_crop = False
    dicom2nifti_brainmask.inputs.source_in_filename = False

    workflow.connect(datagrabber, "brainmask", dicom2nifti_brainmask, "source_names")


    #####################################################################


    # compute 6-dof coregistration parameters of accessory scan 
    # to cropped  main image
    coreg = Node(ants.AI(),
                 name='coregister')
    coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4'}
    coreg.inputs.transform = ('Rigid',
                               0.2)
    coreg.inputs.metric = ('Mattes', 128,
                           'Regular', 0.5)
    coreg.inputs.search_factor = (10, 0.1)
    coreg.inputs.num_threads = 4
    coreg.inputs.output_transform = "acc_to_main_affine.mat" 
    coreg.inputs.verbose = True
    # ici les images crop (T1), accessoire (swi) et brain_mask proviennent du datagrabber 
    workflow.connect(dicom2nifti_acc, 'reoriented_files',
                     coreg, 'moving_image')
    workflow.connect(dicom2nifti_main, 'reoriented_files',
                     coreg, 'fixed_image')
    workflow.connect(dicom2nifti_brainmask, 'reoriented_files',
                     coreg, 'fixed_image_mask')
    

    ###########################################################################



    # write coregistered acc image to cropped space
    apply_coreg = Node(ants.ApplyTransforms(), name="apply_coreg")
    apply_coreg.inputs.interpolation = 'LanczosWindowedSinc'
    workflow.connect(coreg, ('output_transform', as_list), apply_coreg, 'transforms' )
    workflow.connect(dicom2nifti_acc, 'reoriented_files', apply_coreg, 'input_image')
    # ici crop vient du datagrabber également
    workflow.connect(dicom2nifti_main, 'reoriented_files', apply_coreg, 'reference_image')



    ##########################################################################



    # Intensity normalize coregistered image for tensorflow (ENDPOINT 2)
    acc_norm =  Node(Normalization(percentile = 99), name="acc_final_intensity_normalization")
    workflow.connect(apply_coreg, 'output_image',
    		         acc_norm, 'input_image')
    workflow.connect(dicom2nifti_brainmask, 'reoriented_files',
    	 	         acc_norm, 'brain_mask')



    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='SLURM')