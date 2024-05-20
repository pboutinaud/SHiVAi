"""
If the workflow is run with synthseg, it needs shivai_node.inputs.brain_seg = 'synthseg_precomp'
and the synthseg workflow called beforehand needs the same base dir and synthseg.inputs.out_filename = '/mnt/data/synthseg_parc.nii.gz'
"""

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from shivautils.interfaces.shiva import Shivai_Singularity, SynthsegSingularity
import os
import yaml

dummy_args = {
    "SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
    "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
    "SHIVAI_CONFIG": __file__,
}
# args = {'BASE_DIR': '/scratch/nozais/test_shiva/MRI_anat', 'SUBJECT_LIST':['1C016BE'], 'SHIVAI_CONFIG': '/scratch/nozais/test_shiva/config_debug.yml', 'SHIVAI_IMAGE': '/scratch/nozais/test_shiva/shiva_0.3.11bis.sif'}


def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow("ss_shivai_singularity_wf")
    workflow.base_dir = kwargs['BASE_DIR']

    # Load config to get the different file path to bind and the singularity images
    with open(kwargs['SHIVAI_CONFIG'], 'r') as file:
        config = yaml.safe_load(file)

    subject_list_in = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list_in")
    subject_list_in.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    synthseg_node = Node(SynthsegSingularity(),
                         name='synthseg_node')

    shivai_node = Node(Shivai_Singularity(),
                       name='shivai_node')
    # Singularity settings
    config_dir = os.path.dirname(kwargs['SHIVAI_CONFIG'])
    bind_list = [
        (config['model_path'], '/mnt/model', 'ro'),
        (kwargs['BASE_DIR'], kwargs['BASE_DIR'], 'rw'),
    ]
    if os.path.abspath(config_dir) != os.path.abspath(kwargs['BASE_DIR']):
        bind_list.append((config_dir, config_dir, 'rw'))
    shivai_node.inputs.snglrt_bind = bind_list
    shivai_node.inputs.snglrt_image = config['apptainer_image']
    shivai_node.inputs.snglrt_enable_nvidia = True
    # Mandatory inputs:
    shivai_node.inputs.in_dir = kwargs['BASE_DIR']
    shivai_node.inputs.out_dir = kwargs['BASE_DIR']
    shivai_node.inputs.config = kwargs['SHIVAI_CONFIG']
    # shivai_node.inputs.input_type = 'standard'
    # shivai_node.inputs.sub_names = kwargs['SUBJECT_LIST']
    # shivai_node.inputs.prediction = 'PVS'
    # shivai_node.inputs.brain_seg = 'shiva'

    workflow.connect(subject_list_in, 'subject_id', shivai_node, 'sub_name')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
