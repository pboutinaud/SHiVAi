"""
If the workflow is run with synthseg, it needs shivai_node.inputs.brain_seg = 'synthseg_precomp'
and the synthseg workflow called beforehand needs the same base dir and synthseg.inputs.out_filename = '/mnt/data/synthseg_parc.nii.gz'
"""

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles
from shivautils.interfaces.shiva import Shivai_Singularity
import os

dummy_args = {
    "SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
    "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
    "SHIVAI_CONFIG": __file__,
    "SHIVAI_IMAGE":  __file__,
}


def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow("shivai_singularity_wf")
    workflow.base_dir = kwargs['BASE_DIR']

    shivai_node = Node(Shivai_Singularity(),
                       name='shivai_node')
    # Singularity settings
    config_dir = os.path.dirname(kwargs['SHIVAI_CONFIG'])
    shivai_node.inputs.snglrt_bind = [
        (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
        (config_dir, config_dir, 'rw'),]
    shivai_node.inputs.snglrt_image = kwargs['SHIVAI_IMAGE']
    shivai_node.inputs.snglrt_enable_nvidia = True
    # Mandatory inputs:
    shivai_node.inputs.in_dir = kwargs['BASE_DIR']
    shivai_node.inputs.out_dir = kwargs['BASE_DIR']
    shivai_node.inputs.input_type = 'standard'
    shivai_node.inputs.sub_names = kwargs['SUBJECT_LIST']
    shivai_node.inputs.prediction = 'PVS'  # placeholder default
    shivai_node.inputs.brain_seg = 'shiva'  # placeholder default
    shivai_node.inputs.config = kwargs['SHIVAI_CONFIG']

    subject_list_out = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list_out")
    subject_list_out.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    list_output = Node(SelectFiles(),
                       name='list_output')
    list_output.inputs.template = {  # Can be expanded later
        'pvs_census': 'segmentations/pvs_segmentation/{subject_id}/pvs_census.csv',
        'wmh_census': 'segmentations/wmh_segmentation/{subject_id}/wmh_census.csv',
        'cmb_census': 'segmentations/cmb_segmentation_swi-space/{subject_id}/cmb_swi-space_census.csv',
        'lacuna_census': 'segmentations/lac_segmentation/{subject_id}/lac_census.csv',
        'pvs_stats': 'segmentations/pvs_segmentation/{subject_id}/pvs_stats_wide.csv',
        'wmh_stats': 'segmentations/wmh_segmentation/{subject_id}/wmh_stats_wide.csv',
        'cmb_stats': 'segmentations/cmb_segmentation_swi-space/{subject_id}/cmb_swi-space_stats_wide.csv',
        'lacuna_stats': 'segmentations/lac_segmentation/{subject_id}/lac_stats_wide.csv',
        'pvs_labelled_map': 'segmentations/pvs_segmentation/{subject_id}/labelled_pvs.nii.gz',
        'wmh_labelled_map': 'segmentations/wmh_segmentation/{subject_id}/labelled_wmh.nii.gz',
        'cmb_labelled_map': 'segmentations/cmb_segmentation_swi-space/{subject_id}/labelled_cmb.nii.gz',
        'lacuna_labelled_map': 'segmentations/lac_segmentation/{subject_id}/labelled_lac.nii.gz',
        'summary_report': 'report/{subject_id}/Shiva_report.pdf'
    }

    workflow.connect(shivai_node, 'result_dir', list_output, 'base_directory')
    workflow.connect(subject_list_out, 'subject_id', list_output, 'subject_id')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
