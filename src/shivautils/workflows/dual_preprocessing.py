#!/usr/bin/env python
"""Add accessory image (flair) co-registration to cropped space (through ANTS),
   and defacing of native and final images. This also handles back-registration from
   conformed-crop to t1.
   """
import os

from nipype.pipeline.engine import Node, Workflow, JoinNode
from nipype.interfaces import ants
from nipype.interfaces.utility import Function

from shivautils.interfaces.image import Normalization
from shivautils.workflows.preprocessing import genWorkflow as genWorkflowPreproc
from shivautils.workflows.preprocessing_premasked import genWorkflow as genWorkflow_preproc_masked


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
    # Import single img preproc workflow to build upon
    if kwargs['BRAIN_SEG'] is not None:
        workflow = genWorkflow_preproc_masked(**kwargs)
    else:
        workflow = genWorkflowPreproc(**kwargs)

    # file selection
    datagrabber = workflow.get_node('datagrabber')
    datagrabber.inputs.outfields = ['img1', 'img2']

    # compute 6-dof coregistration parameters of accessory scan
    # to cropped t1 image
    flair_to_t1 = Node(ants.Registration(),
                       name='flair_to_t1')
    flair_to_t1.plugin_args = kwargs['REG_PLUGIN_ARGS']
    flair_to_t1.inputs.transforms = ['Rigid']
    flair_to_t1.inputs.transform_parameters = [(0.1,)]
    flair_to_t1.inputs.metric = ['MI']
    flair_to_t1.inputs.radius_or_number_of_bins = [64]
    flair_to_t1.inputs.interpolation = kwargs['INTERPOLATION']
    flair_to_t1.inputs.shrink_factors = [[8, 4, 2, 1]]
    flair_to_t1.inputs.output_warped_image = True
    flair_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    flair_to_t1.inputs.num_threads = 8
    flair_to_t1.inputs.number_of_iterations = [[1000, 500, 250, 125]]
    flair_to_t1.inputs.sampling_strategy = ['Regular']
    flair_to_t1.inputs.sampling_percentage = [0.25]
    flair_to_t1.inputs.output_transform_prefix = "t1_to_flair_"
    flair_to_t1.inputs.verbose = True
    flair_to_t1.inputs.winsorize_lower_quantile = 0.005
    flair_to_t1.inputs.winsorize_upper_quantile = 0.995

    crop = workflow.get_node('crop')
    hard_post_brain_mask = workflow.get_node('hard_post_brain_mask')
    workflow.connect(datagrabber, "img2",
                     flair_to_t1, 'moving_image')
    workflow.connect(crop, 'cropped',
                     flair_to_t1, 'fixed_image')
    workflow.connect(hard_post_brain_mask, ('thresholded', as_list),
                     flair_to_t1, 'fixed_image_masks')

    # write mask to flair in native space
    mask_to_img2 = Node(ants.ApplyTransforms(), name="mask_to_img2")
    mask_to_img2.inputs.interpolation = 'NearestNeighbor'
    mask_to_img2.inputs.invert_transform_flags = [True]

    workflow.connect(flair_to_t1, 'forward_transforms',
                     mask_to_img2, 'transforms')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     mask_to_img2, 'input_image')
    workflow.connect(datagrabber, 'img2',
                     mask_to_img2, 'reference_image')

    # Intensity normalize co-registered image for tensorflow (ENDPOINT 2)
    img2_norm = Node(Normalization(percentile=kwargs['PERCENTILE']), name="img2_final_intensity_normalization")
    workflow.connect(flair_to_t1, 'warped_image',
                     img2_norm, 'input_image')
    workflow.connect(hard_post_brain_mask, 'thresholded',
                     img2_norm, 'brain_mask')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
