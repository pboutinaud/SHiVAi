"""SynthSeg nipype workflow"""
import os

from nipype.pipeline.engine import Node, Workflow
from shivautils.interfaces.shiva import SynthSeg, SynthsegSingularity
from shivautils.interfaces.image import MaskRegions

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

    # Creating the brain segmentation using synthseg
    if kwargs['CONTAINERIZE_NODES']:
        synth_seg = Node(SynthsegSingularity(), name="synth_seg")
        synth_seg.inputs.snglrt_bind = [
            (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
            ('`pwd`', '/mnt/data', 'rw'),]
        synth_seg.inputs.out_filename = '/mnt/data/seg.nii.gz'
        synth_seg.inputs.snglrt_enable_nvidia = True
        synth_seg.inputs.snglrt_image = kwargs['SYNTHSEG_IMAGE']
    else:
        synth_seg = Node(SynthSeg(), name="synth_seg")
        synth_seg.inputs.out_filename = 'seg.nii.gz'
    synth_seg.plugin_args = kwargs['PRED_PLUGIN_ARGS']
    synth_seg.inputs.cpu = False
    synth_seg.inputs.robust = True
    synth_seg.inputs.parc = True

    mask_latventricles_regions = Node(MaskRegions(), name='mask_latventricles_regions')
    mask_latventricles_regions.inputs.list_labels_regions = [4, 5, 43, 44]
    workflow.connect(synth_seg, 'segmentation', mask_latventricles_regions, 'img')

    bg_mask = Node(BGMask(), name="bg_mask")
    workflow.connect(synth_seg, "segmentation", bg_mask, "segmented_regions")

    # Creating a distance map for each ventricle mask
    make_distance_latventricles_map = Node(MakeDistanceMap(), name="make_distance_latventricles_map")
    make_distance_latventricles_map.inputs.out_file = 'distance_map.nii.gz'
    workflow.connect(mask_latventricles_regions, 'mask_regions', make_distance_latventricles_map, "in_file")

    wmh_quantification_latventricles = Node(QuantificationWMHLatVentricles(), name='wmh_quantification_latventricles')
    wmh_quantification_latventricles.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']
    workflow.connect(datagrabber, 'segmentation_wmh', wmh_quantification_latventricles, 'wmh')
    workflow.connect(make_distance_latventricles_map, 'out_file', wmh_quantification_latventricles, 'latventricles_distance_maps')
    workflow.connect(subject_list, 'subject_id', wmh_quantification_latventricles, 'subject_id')

    pvs_quantification_bg = Node(PVSQuantificationBG(), name="pvs_quantification_bg")
    pvs_quantification_bg.inputs.threshold_clusters = kwargs['THRESHOLD_CLUSTERS']
    workflow.connect(datagrabber, 'segmentation_pvs', pvs_quantification_bg, "img")
    workflow.connect(bg_mask, 'bg_mask', pvs_quantification_bg, "bg_mask")

    return workflow
