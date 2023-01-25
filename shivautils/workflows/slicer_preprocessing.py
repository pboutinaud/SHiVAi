"""Nipype workflow for image conformation and preparation before deep
   learning"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface
from pyplm.interfaces.shiva import Predict

from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop)


dummy_args = {'FILE_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("slicer_predictor_preprocessing")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    fileList = Node(IdentityInterface(
                            fields=['filepath'],
                            mandatory_inputs=True),
                            name="fileList")
    fileList.iterables = ('filepath', kwargs['FILE_LIST'])

    


    conform = Node(Conform(),
                   name="conform")
    workflow.connect(fileList, 'filepath', conform, 'img')


    normalization = Node(Normalization(), name="intensity_normalization")
    workflow.connect(conform, 'resampled', normalization, 'input_image')

    brain_mask = Node(Threshold(threshold=0.4, binarize=True), name="brain_mask")
    workflow.connect(normalization, 'intensity_normalized', brain_mask, 'img')


    crop = Node(Crop(),
                name="crop")
    workflow.connect(brain_mask, 'thresholded',
                     crop, 'roi_mask')
    workflow.connect(normalization, 'intensity_normalized',
                     crop, 'apply_to')

    brain_mask_2 = Node(Predict(), "brain_mask_2")
    workflow.connect(crop, 'cropped',
                     brain_mask_2, 't1')

    hard_brain_mask = Node(Threshold(threshold=0.2, binarize=True), name="hard_brain_mask")
    workflow.connect(brain_mask_2, 'segmentation', hard_brain_mask, 'img')
    
    
    crop_2 = Node(Crop(),
    		      name="crop_2")
    workflow.connect(hard_brain_mask, 'thresholded',
    		         crop_2, 'roi_mask')
    workflow.connect(conform, 'resampled', 
    		         crop_2, 'apply_to') 
    workflow.connect(crop, 'cdg_ijk',
    		         crop_2, 'cdg_ijk')
    
    
    normalization_2 = Node(Normalization(), name="intensity_normalization_2")
    workflow.connect(crop_2, 'cropped',
    		         normalization_2, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
    	 	         normalization_2, 'brain_mask')

    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = workflow.base_dir
    workflow.connect(normalization, 'report', datasink, '@savedfile')

    return workflow
