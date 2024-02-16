#!/usr/bin/env python
"""
    Nipype workflow copied and adapted from preprocessing_synthseg.py, swapping the 
    synthseg node by a datagrabber that get the precomputed synthseg parc
"""
import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from shivautils.workflows.preprocessing_premasked import genWorkflow as gen_premasked_wf
from shivautils.interfaces.image import Parc_from_Synthseg, Segmentation_Cleaner, Resample_from_to


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg precomputed segmenation
    It is initialized with gen_premasked_wf, which already contains most
    connections and QCs.

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_synthseg'

    workflow = gen_premasked_wf(**kwargs, )

    datagrabber = workflow.get_node('datagrabber')

    synthseg = Node(DataGrabber(infields=['subject_id'],
                                outfields=['segmentation', 'qc', 'volumes']),
                    name='synthseg')
    synthseg.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results', 'shiva_preproc', 'synthseg')
    synthseg.inputs.raise_on_empty = True
    synthseg.inputs.sort_filelist = True
    synthseg.inputs.template = '%s/%s/*.nii*'
    synthseg.inputs.field_template = {'segmentation': '%s/synthseg_parc.nii*'}  # add 'qc' and 'volumes' here if needed
    synthseg.inputs.template_args = {'segmentation': [['subject_id']]}

    # Correct small "islands" mislabelled by Synthseg
    seg_cleaning = Node(Segmentation_Cleaner(),
                        name='seg_cleaning')
    workflow.connect(synthseg, 'segmentation', seg_cleaning, 'input_seg')

    # conform segmentation to 256x256x256 size (already 1mm3 resolution)
    conform_mask = workflow.get_node('conform_mask')
    workflow.disconnect(datagrabber, 'img1', conform_mask, 'img')  # Changing the connection
    workflow.connect(seg_cleaning, 'ouput_seg', conform_mask, 'img')

    # Putting the synthseg parc in cropped space
    crop = workflow.get_node('crop')
    seg_to_crop = Node(Resample_from_to(),
                       name='seg_to_crop')
    seg_to_crop.inputs.spline_order = 0  # should be equivalent to NearestNeighbor(?)
    seg_to_crop.inputs.out_name = 'synthseg_cropped.nii.gz'
    workflow.connect(conform_mask, 'resampled', seg_to_crop, 'moving_image')
    workflow.connect(crop, "cropped", seg_to_crop, 'fixed_image')

    # Creates our custom segmentation with WM parcellation and lobar distinctions
    custom_parc = Node(Parc_from_Synthseg(), name='custom_parc')
    workflow.connect(seg_to_crop, 'resampled_image', custom_parc, 'brain_seg')

    return workflow
