#!/usr/bin/env python
"""
    Nipype workflow for image preprocessing for nifti file using Synthseg.
    As such, it requires either that Synthseg is installed on the machine
    (with freesurfer for example) or that you provide an Appainer image with
    Synthseg. Synthseg must be able to be called with the command "mri_synthseg".

"""

from nipype.pipeline.engine import Node, Workflow

from shivautils.workflows.preprocessing import genWorkflow as gen_preproc_wf

from shivautils.interfaces.shiva import SynthSeg, SynthsegSingularity

from shivautils.interfaces.image import Parc_from_Synthseg, Segmentation_Cleaner, Resample_from_to


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for image preprocessing using Synthseg

    Returns:
        workflow
    """

    if 'wf_name' not in kwargs.keys():
        kwargs['wf_name'] = 'shiva_preprocessing_synthseg'
    else:
        kwargs['wf_name'] = kwargs['wf_name'] + '_synthseg'

    # Initilazing the wf
    workflow = gen_preproc_wf(**kwargs)

    # Preparing the rewiring of the workflow with the new nodes
    datagrabber = workflow.get_node('datagrabber')
    mask_to_conform = workflow.get_node('mask_to_conform')
    crop = workflow.get_node('crop')
    workflow.disconnect(datagrabber, 'seg', mask_to_conform, 'moving_image')

    # Creating the specific Synthseg nodes
    # First he synthseg node
    if kwargs['CONTAINERIZE_NODES']:
        synthseg = Node(SynthsegSingularity(),
                        name='synthseg')
        synthseg.inputs.snglrt_bind = [
            (kwargs['DATA_DIR'], kwargs['DATA_DIR'], 'ro'),
            # (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),]
        synthseg.inputs.snglrt_image = kwargs['SYNTHSEG_IMAGE']
        synthseg.inputs.out_filename = '/mnt/data/synthseg_parc.nii.gz'
        synthseg.inputs.vol = '/mnt/data/volumes.csv'
        if not kwargs['SYNTHSEG_ON_CPU']:
            synthseg.inputs.snglrt_enable_nvidia = True
    else:
        synthseg = Node(SynthSeg(),
                        name='synthseg')
    if kwargs['SYNTHSEG_ON_CPU']:
        synthseg.inputs.cpu = True
        synthseg.inputs.threads = kwargs['SYNTHSEG_ON_CPU']
        synthseg.plugin_args = {'sbatch_args': f'--nodes 1 --cpus-per-task {kwargs["SYNTHSEG_ON_CPU"]}'}
    else:
        synthseg.plugin_args = kwargs['PRED_PLUGIN_ARGS']

    # Then the correction small "islands" mislabelled by Synthseg
    seg_cleaning = Node(Segmentation_Cleaner(),
                        name='seg_cleaning')

    # Putting the synthseg parc in cropped space
    seg_to_crop = Node(Resample_from_to(),
                       name='seg_to_crop')
    seg_to_crop.inputs.spline_order = 0
    seg_to_crop.inputs.out_name = 'synthseg_cropped.nii.gz'

    # Creates the shiva custom parcellation with WM parcellation and lobar distinctions
    custom_parc = Node(Parc_from_Synthseg(), name='custom_parc')

    # All the connections and rewiring
    workflow.connect(datagrabber, 'img1', synthseg, 'input')
    workflow.connect(synthseg, 'segmentation', seg_cleaning, 'input_seg')
    workflow.connect(seg_cleaning, 'ouput_seg', mask_to_conform, 'moving_image')
    workflow.connect(mask_to_conform, 'resampled_image', seg_to_crop, 'moving_image')
    workflow.connect(crop, "cropped", seg_to_crop, 'fixed_image')
    workflow.connect(seg_to_crop, 'resampled_image', custom_parc, 'brain_seg')

    # ENDPOINT: custom_parc.brain_parc
    return workflow
