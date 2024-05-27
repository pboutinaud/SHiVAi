#!/usr/bin/env python
"""
Workflow computing the different quality control steps of the preprocessing
"""

from nipype.pipeline.engine import Node, Workflow
from shivautils.interfaces.post import QC_metrics
from shivautils.interfaces.image import Isocontour, Save_Histogram, Mask_and_Crop_QC, Brainmask_Overlay


def gen_qc_wf(workflow_name) -> Workflow:
    """
    Quality control workflow for the preprocessing of the 'main' image (t1 or swi)

    Required external connections:
        # qc_crop_box
    ____.connect(____, 'defacing_img1.out_file', ____, 'qc_crop_box.brain_img')
    ____.connect(____, 'hard_brain_mask.thresholded', ____, 'qc_crop_box.brainmask')
    ____.connect(____, 'crop.bbox1', ____, 'qc_crop_box.bbox1')
    ____.connect(____, 'crop.bbox2', ____, 'qc_crop_box.bbox2')
    ____.connect(____, 'crop.cdg_ijk', ____, 'qc_crop_box.slice_coord')
        #qc_overlay_brainmask
    ____.connect(____, 'mask_to_crop.resampled_image', ____, 'qc_overlay_brainmask.brainmask')
    ____.connect(____, 'img1_final_intensity_normalization.intensity_normalized', ____, 'qc_overlay_brainmask.img_ref')
        # save_hist_final
    ____.connect(____, 'img1_final_intensity_normalization.intensity_normalized', ____, 'save_hist_final.img_normalized')
        # qc_metrics
    ____.connect(____, 'mask_to_crop.resampled_image', ____, 'qc_metrics.brain_mask')


    if with flair:
        # qc_coreg_FLAIR_T1
    ____.connect(____, 'img2_final_intensity_normalization.intensity_normalized', ____, 'qc_coreg_FLAIR_T1.path_image')
    ____.connect(____, 'img1_final_intensity_normalization.intensity_normalized', ____, 'qc_coreg_FLAIR_T1.path_ref_image')
    ____.connect(____, 'mask_to_crop.resampled_image', ____, 'qc_coreg_FLAIR_T1.path_brainmask')
        # qc_metrics
    ____.connect(____, 'img2_final_intensity_normalization.mode', ____, 'qc_metrics.flair_norm_peak')
    ____.connect(____, 'flair_to_t1.forward_transforms', ____, 'qc_metrics.flair_reg_mat')

    if with swi and with t1:
        # qc_overlay_brainmask_swi
    ____.connect(____, 'mask_to_crop.resampled_image', ____, 'qc_overlay_brainmask_swi.brainmask')
    ____.connect(____, 'swi_intensity_normalisation.intensity_normalized', ____, 'qc_overlay_brainmask_swi.img_ref')
        # qc_metrics
    ____.connect(____, 'swi_intensity_normalisation.mode', ____, 'qc_metrics.swi_norm_peak')
    ____.connect(____, 'swi_to_t1.forward_transforms', ____, 'qc_metrics.swi_reg_mat')

    """
    workflow = Workflow(workflow_name)

    qc_crop_box = Node(Mask_and_Crop_QC(),
                       name='qc_crop_box')

    qc_overlay_brainmask = Node(Brainmask_Overlay(),
                                name='qc_overlay_brainmask')
    qc_overlay_brainmask.inputs.outname = 'qc_overlay_brainmask.png'

    save_hist_final = Node(Save_Histogram(),
                           name='save_hist_final')

    qc_metrics = Node(QC_metrics(),
                      name='qc_metrics')

    workflow.connect(save_hist_final, 'peak', qc_metrics, 'main_norm_peak')
    # Needs brain mask size, histogram pics

    # Manually adding the nodes are not all are connected from within the wf definition
    workflow.add_nodes([qc_crop_box,
                        qc_overlay_brainmask])
    return workflow


def qc_wf_add_flair(workflow: Workflow) -> Workflow:
    # if with_flair:
    qc_coreg_FLAIR_T1 = Node(Isocontour(),
                             name='qc_coreg_FLAIR_T1')
    qc_coreg_FLAIR_T1.inputs.nb_of_slices = 12  # Should be enough

    workflow.add_nodes([qc_coreg_FLAIR_T1])

    return workflow


def qc_wf_add_swi(workflow) -> Workflow:
    # if with_swi and with_t1:
    qc_overlay_brainmask_swi = Node(Brainmask_Overlay(),
                                    name='qc_overlay_brainmask_swi')
    qc_overlay_brainmask_swi.inputs.outname = 'qc_overlay_brainmask.png'

    workflow.add_nodes([qc_overlay_brainmask_swi])

    return workflow
