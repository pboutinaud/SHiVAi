"""
Shivai workflow for swomed
"""

from nipype import Node, Workflow, IdentityInterface, DataGrabber
from shivai.interfaces.shiva import Shivai_Singularity
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
        config = {'model_path': workflow.base_dir, 'apptainer_image': kwargs['SHIVAI_CONFIG']}

    subject_list = Node(IdentityInterface(
        fields=['subject_id'],
        mandatory_inputs=True),
        name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['t1_image', 'flair_image', 'swi_image']),
                       name='datagrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*'

    shivai_node = Node(Shivai_Singularity(),
                       name='shivai_node')
    # Singularity settings
    config_dir = os.path.dirname(kwargs['SHIVAI_CONFIG'])
    bind_list = [
        (config['model_path'], '/mnt/model', 'ro'),
        (workflow.base_dir, workflow.base_dir, 'rw'),
    ]
    # Pluging the descriptor files when given by swomed
    for descriptor in ['brainmask_descriptor',
                       'wmh_descriptor',
                       'pvs_descriptor',
                       'pvs2_descriptor',
                       'cmb_descriptor',
                       'lac_descriptor']:
        if f'BIOMIST::{descriptor.upper()}' in kwargs:
            setattr(shivai_node.inputs, descriptor, kwargs[f'BIOMIST::{descriptor.upper()}'])
    if os.path.abspath(config_dir) != os.path.abspath(workflow.base_dir):
        bind_list.append((config_dir, config_dir, 'rw'))
    shivai_node.inputs.snglrt_bind = bind_list
    shivai_node.inputs.snglrt_image = config['apptainer_image']
    shivai_node.inputs.snglrt_enable_nvidia = True
    # Mandatory inputs:
    shivai_node.inputs.in_dir = workflow.base_dir
    shivai_node.inputs.out_dir = workflow.base_dir
    shivai_node.inputs.config = kwargs['SHIVAI_CONFIG']
    shivai_node.inputs.input_type = 'swomed'
    # shivai_node.inputs.file_type = 'dicom'
    # shivai_node.inputs.swi_file_num = 1  # second echo
    # shivai_node.inputs.prediction = 'PVS'
    # shivai_node.inputs.brain_seg = 'shiva'

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
    workflow.connect(datagrabber, 't1_image', shivai_node, 't1_image_dcm')  # swap for t1_image_nii with nifti input
    workflow.connect(datagrabber, 'flair_image', shivai_node, 'flair_image_dcm')  # flair_image_nii
    workflow.connect(datagrabber, 'swi_image', shivai_node, 'swi_image_dcm')  # swi_image_nii
    workflow.connect(subject_list, 'subject_id', shivai_node, 'sub_name')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
