"""
Preprocessing workflow using a custom brain parcellation given by the user
"""
from nipype.pipeline.engine import Node, Workflow

from shivai.workflows.preprocessing import genWorkflow as gen_preproc_wf

from shivai.interfaces.shiva import SynthSeg, SynthsegSingularity

from shivai.interfaces.image import Parc_from_Synthseg, Segmentation_Cleaner, Resample_from_to


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using a custom brain segmentation.
    Very similar to the synthseg preproc wf, but without the specifics of synthseg.

    External output connections:
        seg_to_crop.resampled_image

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_seg'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_custom_seg'

    # Initilazing the wf
    workflow = gen_preproc_wf(**kwargs)

    # Putting the custom parc in cropped space
    seg_to_crop = Node(Resample_from_to(),
                       name='seg_to_crop')
    seg_to_crop.inputs.spline_order = 0
    seg_to_crop.inputs.out_name = 'custom_seg_cropped.nii.gz'

    # Rewiring the workflow with the new nodes
    mask_to_conform = workflow.get_node('mask_to_conform')
    crop = workflow.get_node('crop')
    workflow.connect(mask_to_conform, 'resampled_image', seg_to_crop, 'moving_image')
    workflow.connect(crop, "cropped", seg_to_crop, 'fixed_image')

    # ENDPOINT for custom seg: seg_to_crop.resampled_image

    return workflow
