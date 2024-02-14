#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing for nifti file using Synthseg.
    As such, it requires either that Synthseg is installed on the machine
    (with freesurfer for example) or that you provide an Appainer image with
    Synthseg. Synthseg must be able to be called with the command "mri_synthseg".

    The workflow is derived from the "preprocessing_premasked" workflow.
"""

from nipype.pipeline.engine import Node, Workflow
from shivautils.workflows.preprocessing_premasked import genWorkflow as gen_premasked_wf
from shivautils.interfaces.shiva import SynthSeg, SynthsegSingularity

from shivautils.interfaces.image import Parc_from_Synthseg


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg
    It is initialized with gen_premasked_wf, which already contains most
    connections and QCs.

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_synthseg'

    workflow = gen_premasked_wf(**kwargs)

    datagrabber = workflow.get_node('datagrabber')

    if kwargs['CONTAINERIZE_NODES']:
        synthseg = Node(SynthsegSingularity(),
                        name='synthseg')
        synthseg.inputs.snglrt_bind = [
            (kwargs['DATA_DIR'], kwargs['DATA_DIR'], 'ro'),
            # (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),]
        synthseg.inputs.snglrt_image = kwargs['SYNTHSEG_IMAGE']
        synthseg.inputs.out_filename = '/mnt/data/synthseg_parc.nii.gz'
        if not kwargs['SYNTHSEG_ON_CPU']:
            synthseg.inputs.snglrt_enable_nvidia = True
    else:
        synthseg = Node(SynthSeg(),
                        name='synthseg')
    if kwargs['SYNTHSEG_ON_CPU']:
        synthseg.inputs.cpu = True
        synthseg.inputs.threads = kwargs['SYNTHSEG_ON_CPU']
        synthseg.plugin_args = {'sbatch_args': f'--nodes 1 --cpus-per-task {kwargs['SYNTHSEG_ON_CPU']}'}
    else:
        synthseg.plugin_args = kwargs['PRED_PLUGIN_ARGS']

    workflow.connect(datagrabber, 'img1', synthseg, 'input')

    # conform segmentation to 256x256x256 size (already 1mm3 resolution)
    conform_mask = workflow.get_node('conform_mask')
    workflow.disconnect(datagrabber, "img1", conform_mask, 'img')  # Changing the connection
    workflow.connect(synthseg, 'segmentation', conform_mask, 'img')

    mask_to_crop = workflow.get_node('mask_to_crop')
    mask_to_crop.inputs.out_name = 'synthseg_cropped.nii.gz'
    # mask_to_crop.resampled_image contains the parcelization just before the binarization

    # Creates our custom segmentation with WM parcellation and lobar distinctions
    custom_parc = Node(Parc_from_Synthseg(), name='custom_parc')
    workflow.connect(mask_to_crop, 'resampled_image', custom_parc, 'brain_seg')

    return workflow
