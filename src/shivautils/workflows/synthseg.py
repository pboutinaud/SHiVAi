"""SynthSeg nipype workflow"""
import os

from nipype.pipeline.engine import Node, Workflow
from shivautils.interfaces.shiva import SynthSeg, SynthsegSingularity

dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow that takes an MRI as input and perform the
    brain segmentation using mri_synthseg, then produce the different maps and
    masks used to compute metrics

    Requires a connection to an external datagrabber:
    main_wf.connect(preproc_wf.datagrabber, "img1", synth_seg_wf.synth_seg, "input")

    Returns:
        workflow
    """
    workflow = Workflow('Synthseg_workflow')
    workflow.base_dir = kwargs['BASE_DIR']

    if kwargs['CONTAINERIZE_NODES']:
        synth_seg = Node(SynthsegSingularity(), name="synth_seg")
        synth_seg.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),
            (kwargs['MODELS_PATH'], kwargs['MODELS_PATH'], 'ro')]
        synth_seg.inputs.out_filename = '/mnt/data/pvs_map.nii.gz'
        synth_seg.inputs.snglrt_enable_nvidia = True
        synth_seg.inputs.snglrt_image = kwargs['CONTAINER_IMAGE']

    synth_seg = Node(SynthSeg(), name="synth_seg")
    synth_seg.plugin_args = kwargs['PRED_PLUGIN_ARGS']
    synth_seg.inputs.cpu = False
    synth_seg.inputs.robust = True
    synth_seg.inputs.parc = True
    synth_seg.inputs.out_filename = 'seg.nii.gz'

    return workflow
