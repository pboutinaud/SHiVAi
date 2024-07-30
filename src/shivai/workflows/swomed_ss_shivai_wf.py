"""
Shivai worflow for swomed with a dcm2nii first step and synthseg 
"""

from nipype import Node, Workflow, IdentityInterface, DataGrabber
from nipype.interfaces.dcm2nii import Dcm2niix
from shivai.interfaces.shiva import Shivai_Singularity, SynthsegSingularity
from shivai.utils.misc import file_selector
import os
import yaml

dummy_args = {
    "SUBJECT_LIST": ['BIOMIST::SUBJECT_LIST'],
    "BASE_DIR": os.path.normpath(os.path.expanduser('~')),
    "SHIVAI_CONFIG": __file__,
}
# args = {'BASE_DIR': '/scratch/nozais/test_shiva/MRI_anat', 'SUBJECT_LIST':['1C016BE'], 'SHIVAI_CONFIG': '/scratch/nozais/test_shiva/config_debug.yml', 'SHIVAI_IMAGE': '/scratch/nozais/test_shiva/shiva_0.3.11bis.sif'}


def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow("synthseg_shivai_singularity_wf")
    workflow.base_dir = kwargs['BASE_DIR']

    # Load config to get the different file path to bind and the singularity images
    if os.path.splitext(kwargs['SHIVAI_CONFIG'])[1] in ['.yaml', '.yml']:
        with open(kwargs['SHIVAI_CONFIG'], 'r') as file:
            config = yaml.safe_load(file)
    else:  # dummy args
        config = {'model_path': kwargs['BASE_DIR'],
                  'apptainer_image': kwargs['SHIVAI_CONFIG'],
                  'synthseg_image': kwargs['SHIVAI_CONFIG'],
                  'parameters': {'swi_echo': 1}}

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
    datagrabber.inputs.template = '%s/%s/'
    datagrabber.inputs.template_args = {'t1_image': [['subject_id', 't1']],
                                        'flair_image': [['subject_id', 'flair']],
                                        'swi_image': [['subject_id', 'swi']]}

    dcm2nii_t1_node = Node(Dcm2niix(), name='dcm2nii_t1_node')
    dcm2nii_t1_node.inputs.anon_bids = True
    dcm2nii_t1_node.inputs.out_filename = 'converted_%p'
    dcm2nii_flair_node = Node(Dcm2niix(), name='dcm2nii_flair_node')
    dcm2nii_flair_node.inputs.anon_bids = True
    dcm2nii_flair_node.inputs.out_filename = 'converted_%p'
    dcm2nii_swi_node = Node(Dcm2niix(), name='dcm2nii_swi_node')
    dcm2nii_swi_node.inputs.anon_bids = True
    dcm2nii_swi_node.inputs.out_filename = 'converted_%p'

    synthseg_node = Node(SynthsegSingularity(),
                         name='synthseg_node')
    synthseg_node.inputs.snglrt_bind = [(workflow.base_dir, workflow.base_dir, 'rw')]
    synthseg_node.inputs.snglrt_image = config['synthseg_image']
    synthseg_node.inputs.snglrt_enable_nvidia = True
    synthseg_node.inputs.out_filename = 'synthseg_parc.nii.gz'
    synthseg_node.inputs.vol = 'volumes.csv'

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
    # shivai_node.inputs.brain_seg = 'synthseg'

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')
    workflow.connect(datagrabber, 't1_image', dcm2nii_t1_node, 'source_dir')
    workflow.connect(datagrabber, 'flair_image', dcm2nii_flair_node, 'source_dir')
    workflow.connect(datagrabber, 'swi_image', dcm2nii_swi_node, 'source_dir')
    workflow.connect(dcm2nii_t1_node, 'converted_files', synthseg_node, 'input')
    workflow.connect(subject_list, 'subject_id', shivai_node, 'sub_name')
    workflow.connect(dcm2nii_t1_node, 'converted_files', shivai_node, 't1_image_nii')
    workflow.connect(dcm2nii_flair_node, 'converted_files', shivai_node, 'flair_image_nii')
    workflow.connect(dcm2nii_swi_node, ('converted_files', file_selector, config['parameters']['swi_echo']), shivai_node, 'swi_image_nii')
    workflow.connect(synthseg_node, 'segmentation', shivai_node, 'synthseg_parc')
    workflow.connect(synthseg_node, 'volumes', shivai_node, 'synthseg_vol')

    return workflow


if __name__ == '__main__':
    wf = genWorkflow(**dummy_args)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.run(plugin='Linear')
