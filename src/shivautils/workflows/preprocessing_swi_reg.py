#!/usr/bin/env python
"""
Plugin workflow to add coregistration steps from SWI to T1 when doing CMB
segmentation while a T1 is available from another segmentation
"""
from nipype.interfaces import ants
from nipype.pipeline.engine import Node, Workflow


def genWorkflow_swi_pluggin(in_workflow, **kwargs) -> Workflow:
    """
    Plugin workflow to add coregistration steps from SWI to T1 when doing CMB
    segmentation while a T1 is available from another segmentation

    Returns:
        workflow
    """
    # compute 6-dof coregistration parameters of t1 croped image
    # to raw swi
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
    coreg.inputs.output_transform_prefix = "t1_to_swi_"
    coreg.inputs.verbose = True
    coreg.inputs.winsorize_lower_quantile = 0.005
    coreg.inputs.winsorize_upper_quantile = 0.995

    seg_to_native = Node(ants.ApplyTransforms(), name="seg_to_native")
