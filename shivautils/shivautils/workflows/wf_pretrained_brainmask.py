"""Nipype workflow for image conformation and preparation before deep
   learning"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface

from shivautils.interfaces.image import Threshold, Normalization, Conform, Crop, Apply_mask


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow_brainmask_pretrained(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("preprocessing")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subjectList = Node(IdentityInterface(
        fields=['subject_id'], mandatory_inputs=True), name="subjectList")
    subjectList.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['main']),
                       name='dataGrabber')
    datagrabber.inputs.base_directory = kwargs['BASE_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*nii*'
    datagrabber.inputs.template_args = {'main': [['subject_id', 'main']]}

    workflow.connect(subjectList, 'subject_id', datagrabber, 'subject_id')

    conform = Node(Conform(dimensions=(256, 256, 256),
                           voxel_size=kwargs['voxel_size'],
                           orientation='RAS'),
                   name="conform")

    workflow.connect(datagrabber, 'main', conform, 'img')

    normalization = Node(Normalization(percentile=kwargs['percentile']), name="intensity_normalization")
    workflow.connect(conform, 'resampled', normalization, 'input_image')

    brain_mask = Node(Threshold(threshold=0.4, binarize=True), name="brain_mask")
    workflow.connect(normalization, 'intensity_normalized', brain_mask, 'img')

    crop = Node(Crop(final_dimensions=kwargs['final_dimensions']),
                name="crop")
    workflow.connect(brain_mask, 'thresholded',
                     crop, 'roi_mask')
    workflow.connect(normalization, 'intensity_normalized',
                     crop, 'apply_to')
    apply_mask = Node(Apply_mask(model=kwargs['model_brainmask']),
    		          name="apply_mask")
    workflow.connect(crop, 'cropped',
    		         apply_mask, 'apply_to')
    crop_2 = Node(Crop(final_dimensions=kwargs['final_dimensions']),
    		      name="crop_2")
    workflow.connect(brain_mask, 'thresholded',
    		         crop_2, 'roi_mask')
    workflow.connect(conform, 'resampled', 
    		         crop_2, 'apply_to') 
    workflow.connect(crop, 'cdg_ijk',
    		         crop_2, 'cdg_ijk')
    normalization_2 = Node(Normalization(percentile=kwargs['percentile']), name="intensity_normalization_2")
    workflow.connect(crop_2, 'cropped',
    		         normalization_2, 'input_image')
    workflow.connect(apply_mask, 'brain_mask',
    	 	         normalization_2, 'brain_mask')

    datasink = Node(DataSink(), name='sinker')
    datasink.inputs.base_directory = workflow.base_dir
    workflow.connect(normalization, 'report', datasink, '@savedfile')

    return workflow
