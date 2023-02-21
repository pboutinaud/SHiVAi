"""Nipype workflow for image conformation and preparation before deep
   learning, with accessory image coregistation (through ANTS).
   
   Equivalent to wf_pretrained_brainmask.py with two input images.
   """
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces import ants
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.utility import IdentityInterface
from pyplm.interfaces.shiva import Predict

from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


dummy_args = {'SUBJECT_LIST': ['test_subject'],
              'BASE_DIR': os.path.normpath(
                            os.path.expanduser('/bigdata/pyherve/test'))}


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

    # crop main centered on xyz origin (affine matrix)
    crop = Node(Crop(final_dimensions=(160, 214, 176)),
                name="crop")
    workflow.connect(conform, 'resampled',
                     crop, 'apply_to')
    
    # normalize intenisties between 0 and 1 for Tensorflow
    prenormalization = Node(Normalization(percentile = 99), name="pre_intensity_normalization")
    workflow.connect(crop, 'cropped',
                     prenormalization, 'input_image')

    # brain mask from tensorflow
    brain_mask = Node(Predict(), "brain_mask")
    brain_mask.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 1 --partition GPU'}
    brain_mask.inputs.snglrt_bind =  [
        (kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),
        ('`pwd`','/mnt/data','rw'),
        ('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),
        ('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),
        ('/homes_unix/boutinaud/ReferenceModels', '/mnt/model', 'ro')]
    brain_mask.inputs.model = '/mnt/model'
    brain_mask.inputs.snglrt_enable_nvidia = True
    brain_mask.inputs.descriptor = "/bigdata/pyherve/test/model_info.json"
    brain_mask.inputs.snglrt_image = '/bigdata/yrio/singularity/predict.sif'
    brain_mask.inputs.out_filename = '/mnt/data/brain_mask.nii.gz'
    workflow.connect(prenormalization, 'intensity_normalized',
                     brain_mask, 't1')

    # binarize brain mask
    hard_brain_mask = Node(Threshold(threshold=0.5, binarize=True), name="hard_brain_mask")
    workflow.connect(brain_mask, 'segmentation', hard_brain_mask, 'img')

    # non-uniformity correct within brain mask
    nuc_main = Node(ants.N4BiasFieldCorrection(), name="nuc_main")
    workflow.connect(crop, 'cropped',
                     nuc_main, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
                     nuc_main, 'mask_image')

    # renormalize intensities of cropped image for tensorflow, 
    # this time within brain mask (ENDPOINT 1)
    normalization = Node(Normalization(percentile = 99), name="post_intensity_normalization")
    workflow.connect(nuc_main, 'output_image',
    		         normalization, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
    	 	         normalization, 'brain_mask')

    # compute 6-dof coregistration parameters of accessory scan 
    # to cropped nuc normalized main image
    coreg = Node(ants.AI(),
                 name='coregister')
    coreg.inputs.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 4'}
    coreg.inputs.transform = ('Rigid',
                              0.1)
    coreg.inputs.metric = ('Mattes', 128,
                           'Regular', 1)
    coreg.inputs.search_factor = (10, 0.1)
    coreg.inputs.num_threads = 4
    coreg.inputs.verbose = True
    workflow.connect(dicom2nifti_acc, 'reoriented_files',
                     coreg, 'moving_image')
    workflow.connect(normalization,
                     'intensity_normalized',
                     coreg,
                     'fixed_image')
    workflow.connect(hard_brain_mask,
                     'thresholded',
                     coreg,
                     'fixed_image_mask')

    # write coregistered image
    apply_coreg = Node(ants.ApplyTransforms(), name="apply_coreg")
    apply_coreg.inputs.interpolation = 'LanczosWindowedSinc'
    workflow.connect(coreg, ('output_transform', as_list), apply_coreg, 'transforms' )
    workflow.connect(dicom2nifti_acc, 'reoriented_files', apply_coreg, 'input_image')
    workflow.connect(normalization, 'intensity_normalized', apply_coreg, 'reference_image')

    # NUC coregistered image within brain mask
    nuc_acc = Node(ants.N4BiasFieldCorrection(), name="nuc_acc")
    workflow.connect(apply_coreg, 'output_image',
                     nuc_acc, "input_image")
    workflow.connect(hard_brain_mask, 'thresholded',
                     nuc_acc, 'mask_image')

    # Intensity normalize coregisterd image for tensorflow (ENDPOINT 2)
    coreg_norm =  Node(Normalization(percentile = 99), name="coreg_intensity_normalization")
    workflow.connect(nuc_acc, 'output_image',
    		         coreg_norm, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
    	 	         coreg_norm, 'brain_mask')
    
    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear',plugin_args={'sbatch_args':'-p all'})
