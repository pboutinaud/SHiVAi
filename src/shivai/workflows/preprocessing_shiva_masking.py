"""
Imports the default preprocessing workflow and add the shiva_mask_wf to create a brain mask from img1
"""


from nipype.pipeline.engine import Workflow
from shivai.workflows.preprocessing import genWorkflow as gen_preproc_wf
from shivai.workflows.shiva_mask_wf import genWorkflow as gen_masking_wf


def genWorkflow(**kwargs) -> Workflow:
    # Initilazing the wf
    if 'wf_name' not in kwargs['wf_name']:
        kwargs['wf_name'] = 'shiva_preprocessing_brainmask'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_brainmask'
    workflow = gen_preproc_wf(**kwargs)

    # Changing the thresholding parameters with stronger cleaning steps
    binarize_brain_mask = workflow.get_node('binarize_brain_mask')
    binarize_brain_mask.inputs.open = 3  # morphological opening of clusters using a ball of radius 3
    binarize_brain_mask.inputs.minVol = 30000  # Get rif of potential small clusters
    binarize_brain_mask.inputs.clusterCheck = 'size'  # Select biggest cluster

    # Creating and incorporating the brain mask sub-wf
    masking_wf = gen_masking_wf(**kwargs)
    workflow.add_nodes([masking_wf])

    # Changing the connections to adapt to the sub-wf
    datagrabber = workflow.get_node('datagrabber')
    conform = workflow.get_node('conform')
    mask_to_conform = workflow.get_node('mask_to_conform')
    workflow.disconnect(datagrabber, 'seg', mask_to_conform, 'moving_image')
    # Since we use the conformed image as input for proper_brain_mask, no need for corrected_affine
    workflow.disconnect(conform, 'corrected_affine', mask_to_conform, 'corrected_affine')
    workflow.disconnect(conform, 'original_affine', mask_to_conform, 'original_affine')
    workflow.connect(datagrabber, 'img1', masking_wf, 'preconform.img')
    workflow.connect(conform, 'resampled', masking_wf, 'intensity_norm_with_premask.input_image')
    workflow.connect(masking_wf, 'proper_brain_mask.segmentation', mask_to_conform, 'moving_image')

    return workflow
