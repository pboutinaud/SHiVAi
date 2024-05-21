"""
If the workflow is run with synthseg, it needs shivai_node.inputs.brain_seg = 'synthseg_precomp'
and the synthseg workflow called beforehand needs the same base dir and synthseg.inputs.out_filename = '/mnt/data/synthseg_parc.nii.gz'
"""

from nipype import Node, Workflow, IdentityInterface, DataGrabber
from shivautils.interfaces.shiva import Shivai_Singularity
import os
import yaml

dummy_args = {
    "SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
    "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
    "SHIVAI_CONFIG": __file__,
}
# args = {'BASE_DIR': '/scratch/nozais/test_shiva/MRI_anat', 'SUBJECT_LIST':['1C016BE'], 'SHIVAI_CONFIG': '/scratch/nozais/test_shiva/config_debug.yml', 'SHIVAI_IMAGE': '/scratch/nozais/test_shiva/shiva_0.3.11bis.sif'}


def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow("shivai_singularity_wf")
    workflow.base_dir = kwargs['BASE_DIR']

    # Load config to get the different file path to bind and the singularity images
    if os.path.splitext(kwargs['SHIVAI_CONFIG'])[1] in ['.yaml', '.yml']:
        with open(kwargs['SHIVAI_CONFIG'], 'r') as file:
            config = yaml.safe_load(file)
    else:  # dummy args
        config = {'model_path': kwargs['BASE_DIR'], 'apptainer_image': kwargs['SHIVAI_CONFIG']}

    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['t1_image', 'flair_image', 'swi_image']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.*'

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
    shivai_node.inputs.sub_name = 'dummy'
    shivai_node.inputs.input_type = 'swomed'
    # shivai_node.inputs.prediction = 'PVS'
    # shivai_node.inputs.brain_seg = 'shiva'

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
    workflow.connect(datagrabber, 't1_image', shivai_node, 't1_image')
    workflow.connect(datagrabber, 'flair_image', shivai_node, 'flair_image')
    workflow.connect(datagrabber, 'swi_image', shivai_node, 'swi_image')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
