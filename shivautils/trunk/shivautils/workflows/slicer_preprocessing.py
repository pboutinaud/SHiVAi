"""Nipype workflow for image conformation and preparation before deep
   learning"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
from pyplm.interfaces.shiva import Predict

from shivautils.interfaces.image import (Threshold, Normalization,
                            Conform, Crop, DataGrabberSlicer)


dummy_args = {'FILES_LIST': ['BIOMIST::SUBJECT_LIST'],
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
                            fields=['fileList'],
                            mandatory_inputs=True),
                            name="fileList")

    fileList.iterables = ('fileList', kwargs['FILES_LIST'])

    # file selection
    datagrabberslicer = Node(DataGrabberSlicer(args=kwargs['ARGS'], subject='subject_id'),
                             name='dataGrabber')

    workflow.connect(fileList, 'fileList', datagrabberslicer, 'subject')

    conform = Node(Conform(),
                   name="conform")
    workflow.connect(datagrabberslicer, 'raw', conform, 'img')

    conformSeg = Node(Conform(order=0),
                      name="conform_seg")
    workflow.connect(datagrabberslicer, 'seg', conformSeg, 'img')


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
    
    cropSeg = Node(Crop(),
                   name="crop_seg")
    workflow.connect(brain_mask, 'thresholded',
                     cropSeg, 'roi_mask')
    workflow.connect(conformSeg, 'resampled',
                     cropSeg, 'apply_to')

    brain_mask_2 = Node(Predict(), "brain_mask_2")
    workflow.connect(crop, 'cropped',
                     brain_mask_2, 't1')

    hard_brain_mask = Node(Threshold(threshold=0.2, binarize=True), name="hard_brain_mask")
    workflow.connect(brain_mask_2, 'segmentation', hard_brain_mask, 'img')
    
    
    crop_2 = Node(Crop(),
    		      name="crop_2")
    workflow.connect(brain_mask, 'thresholded',
    		         crop_2, 'roi_mask')
    workflow.connect(conform, 'resampled', 
    		         crop_2, 'apply_to') 
    
    
    normalization_2 = Node(Normalization(), name="intensity_normalization_2")
    workflow.connect(crop_2, 'cropped',
    		         normalization_2, 'input_image')
    workflow.connect(hard_brain_mask, 'thresholded',
    	 	         normalization_2, 'brain_mask')

    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = workflow.base_dir
    workflow.connect(normalization, 'report', datasink, '@savedfile')

    return workflow
