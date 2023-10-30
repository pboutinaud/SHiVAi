#!/usr/bin/env python
"""Nipype workflow for DICOM to NII image conversion, conformation and preparation before deep
   learning, with accessory image coregistation to cropped space (through ANTS),
   and defacing of native and final images. This also handles back-registration from
   conformed-crop to t1.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces import ants
from nipype.interfaces.utility import Function

from shivautils.interfaces.image import Normalization
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc

dummy_args = {"SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
              "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
              "DESCRIPTOR": os.path.normpath(os.path.join(os.path.expanduser('~'), '.swomed', 'default_config.ini'))
              }


def as_list(input):
    return [input]


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow for T1 + FLAIR based on T1-only workflow

    Returns:
        workflow
    """
    # Import single img preproc workflow to build uppon
    workflow = genWorkflowPreproc(**kwargs)

    # file selection
    datagrabber = workflow.get_node('datagrabber')
    datagrabber.inputs.outfields = ['img1', 'img2']

    # compute 6-dof coregistration parameters of accessory scan
    # to cropped t1 image
    coreg = Node(ants.Registration(),
                 name='coregister')
    coreg.plugin_args = {'sbatch_args': '--nodes 1 --cpus-per-task 8'}  # TODO: would it work with other schedulers?
    coreg.inputs.transforms = ['Rigid']
    coreg.inputs.transform_parameters = [(0.1,)]
    coreg.inputs.metric = ['MI']
    coreg.inputs.radius_or_number_of_bins = [64]
    coreg.inputs.interpolation = 'WelchWindowedSinc'
    coreg.inputs.shrink_factors = [[8, 4, 2, 1]]
    coreg.inputs.output_warped_image = True
    coreg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    coreg.inputs.num_threads = 8
    coreg.inputs.number_of_iterations = [[1000, 500, 250, 125]]
    coreg.inputs.sampling_strategy = ['Regular']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.output_transform_prefix = "t1_to_flair_"
    coreg.inputs.verbose = True
    coreg.inputs.winsorize_lower_quantile = 0.005
    coreg.inputs.winsorize_upper_quantile = 0.995

    crop = workflow.get_node('crop')
    hard_post_brain_mask = workflow.get_node('hard_post_brain_mask')
    workflow.connect(datagrabber, "img2",
                     coreg, 'moving_image')
    workflow.connect(crop, 'cropped',
                     coreg, 'fixed_image')
    workflow.connect(hard_post_brain_mask, ('thresholded', as_list),
                     coreg, 'fixed_image_masks')

    # write mask to flair in native space
    mask_to_img2 = Node(ants.ApplyTransforms(), name="mask_to_img2")
    mask_to_img2.inputs.interpolation = 'NearestNeighbor'
    mask_to_img2.inputs.invert_transform_flags = [True]

    workflow.connect(coreg, 'forward_transforms',
                     mask_to_img2, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     mask_to_img2, 'input_image')
    workflow.connect(datagrabber, 'img2',
                     mask_to_img2, 'reference_image')

    # Intensity normalize coregistered image for tensorflow (ENDPOINT 2)
    img2_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img2_final_intensity_normalization")
    workflow.connect(coreg, 'warped_image',
                     img2_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     img2_norm, 'brain_mask')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
