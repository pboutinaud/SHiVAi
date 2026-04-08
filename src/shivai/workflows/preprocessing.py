#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing, with conformation and preparation before AI segmentation.
    Defacing of native and final images. This also handles back-registration from
    conformed-crop to T1 or SWI ('img1').

    Required external input connections:
        datagrabber.subject_id
    
    External output connections:
        mask_to_crop.resampled_image
        img1_final_intensity_normalization.intensity_normalized
        img2_final_intensity_normalization.intensity_normalized
        mask_to_crop.resampled_image
        crop.bbox1_file
        crop.bbox2_file
        crop.cdg_ijk_file
        preproc_qc_workflow.qc_crop_box.crop_brain_img
        preproc_qc_workflow.qc_overlay_brainmask.overlayed_brainmask
        preproc_qc_workflow.qc_overlay_brainmask_swi.overlayed_brainmask
        preproc_qc_workflow.qc_coreg_FLAIR_T1.qc_coreg
        preproc_qc_workflow.qc_metrics.csv_qc_metrics
        
"""
import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.quickshear import Quickshear
from shivai.interfaces.image import (Threshold, Normalization, CorrectAffine,
                                     Conform, Crop, Resample_from_to)
from shivai.interfaces.shiva import Quickshear_Contained
from shivai.utils.container_config import configure_container_node
from shivai.workflows.qc_preproc import gen_qc_wf


dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    "conformed" = (256, 256, 256) 1x1x1mm3
    "preconformed" = (160, 214, 176) = pred model dim (no specific resolution in mm3)
    "unpreconformed" = "preconformed" sent in "conformed" space
    Returns:
        workflow
    """
    wf_name = 'shiva_preprocessing'
    if 'wf_name' in kwargs.keys():
        wf_name = kwargs['wf_name']
    workflow = Workflow(wf_name)
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on

    # file selection
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1', 'img2', 'img3', 'seg']),
        name='datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'

    correct_affine_img1 = Node(CorrectAffine(), name="correct_affine_img1")
    correct_affine_img1.inputs.correction_threshold = kwargs['AFFINE_CORREC_THRESHOLD']
    workflow.connect(datagrabber, 'img1', correct_affine_img1, 'img')

    # conform img1 to 1 mm isotropic, freesurfer-style (unless a tolerance marfin is given)
    conform = Node(Conform(),
                   name="conform")
    conform.inputs.dimensions = (256, 256, 256)
    conform.inputs.voxel_size = kwargs['RESOLUTION']
    conform.inputs.voxels_tolerance = kwargs['TOLERANCE']
    conform.inputs.orientation = kwargs['ORIENTATION']
    # conform.inputs.correction_threshold = kwargs['AFFINE_CORREC_THRESHOLD']
    # conform.inputs.adaptive_dim = True  # adapt dimensions to keep FOV if tolerance kept some voxel sizes

    workflow.connect(correct_affine_img1, 'corrected_img', conform, 'img')

    # conform mask to 256 256 256, same as anatomical conformed image (works even on cropped input like shiva mask)

    correct_affine_seg = Node(CorrectAffine(), name="correct_affine_seg")
    correct_affine_seg.inputs.correction_threshold = kwargs['AFFINE_CORREC_THRESHOLD']
    workflow.connect(datagrabber, 'seg', correct_affine_seg, 'img')

    mask_to_conform = Node(Resample_from_to(),
                           name="mask_to_conform")
    mask_to_conform.inputs.spline_order = 0

    workflow.connect(correct_affine_seg, 'corrected_img', mask_to_conform, 'moving_image')
    workflow.connect(conform, 'resampled', mask_to_conform, 'fixed_image')

    # binarize and clean conformed brain mask
    binarize_brain_mask = Node(Threshold(threshold=kwargs['THRESHOLD']), name="binarize_brain_mask")
    binarize_brain_mask.inputs.binarize = True
    binarize_brain_mask.inputs.minVol = 100  # Get rif of potential small clusters
    binarize_brain_mask.inputs.clusterCheck = 'all'  # Keep all cluster above minVol

    workflow.connect(mask_to_conform, 'resampled_image', binarize_brain_mask, 'img')

    # Defacing the conformed image
    container_runtime = kwargs.get('CONTAINER_RUNTIME')
    if kwargs['CONTAINERIZE_NODES']:
        defacing_img1 = Node(Quickshear_Contained(), name="defacing_img1")
        bind_list = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '`pwd`', 'rw')]
        configure_container_node(defacing_img1, container_runtime, kwargs['CONTAINER_IMAGE'], bind_list, gpu=False)
    else:
        defacing_img1 = Node(Quickshear(),
                             name='defacing_img1')

    workflow.connect(conform, 'resampled', defacing_img1, 'in_file')
    workflow.connect(binarize_brain_mask, 'thresholded', defacing_img1, 'mask_file')

    # crop img1 centered on mask
    crop = Node(Crop(final_dimensions=kwargs['IMAGE_SIZE']),
                name="crop")
    workflow.connect(defacing_img1, 'out_file',
                     crop, 'apply_to')
    workflow.connect(binarize_brain_mask, 'thresholded',
                     crop, 'roi_mask')

    # Apply the cropping to the mask
    mask_to_crop = Node(Resample_from_to(),
                        name='mask_to_crop')
    mask_to_crop.inputs.spline_order = 0  # should be equivalent to NearestNeighbor(?)
    mask_to_crop.inputs.out_name = 'brainmask_cropped.nii.gz'

    workflow.connect(binarize_brain_mask, 'thresholded', mask_to_crop, 'moving_image')
    workflow.connect(crop, "cropped", mask_to_crop, 'fixed_image')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 1)
    img1_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img1_final_intensity_normalization")
    if 'inverse_t2' in kwargs['ACQUISITIONS']:
        img1_norm.inputs.inverse = kwargs['ACQUISITIONS']['inverse_t2']
    workflow.connect(crop, 'cropped',
                     img1_norm, 'input_image')
    workflow.connect(mask_to_crop, 'resampled_image',
                     img1_norm, 'brain_mask')

    # Adding the QC sub-workflow
    # Initializing the wf
    qc_wf = gen_qc_wf('preproc_qc_workflow')
    workflow.add_nodes([qc_wf])
    # Connect QC nodes
    workflow.connect(defacing_img1, 'out_file', qc_wf, 'qc_crop_box.brain_img')
    workflow.connect(binarize_brain_mask, 'thresholded', qc_wf, 'qc_crop_box.brainmask')
    workflow.connect(crop, 'bbox1', qc_wf, 'qc_crop_box.bbox1')
    workflow.connect(crop, 'bbox2', qc_wf, 'qc_crop_box.bbox2')
    workflow.connect(crop, 'cdg_ijk', qc_wf, 'qc_crop_box.slice_coord')
    workflow.connect(mask_to_crop, 'resampled_image', qc_wf, 'qc_overlay_brainmask.brainmask')
    workflow.connect(img1_norm, 'intensity_normalized', qc_wf, 'qc_overlay_brainmask.img_ref')
    workflow.connect(img1_norm, 'intensity_normalized', qc_wf, 'save_hist_final.img_normalized')
    workflow.connect(mask_to_crop, 'resampled_image', qc_wf, 'qc_metrics.brain_mask')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
