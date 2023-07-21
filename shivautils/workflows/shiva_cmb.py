#!/usr/bin/env python
"""SHIVA project Cerebral Micro Bleeds 3D U-Net prediction pipeline.

@author: Pierre-Yves Herve
@contact: pyherve@fealinx.com
"""
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber
from pyplm.interfaces.shiva import Predict
from nipype.interfaces.ants import ApplyTransforms
import os


# dummy args are a set of dummy parameters. The real kwargs come
# from teamcenter.
# SUBJECT_LIST and BASE_DIR come from the dynamic parameters
dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~')),
              'DESCRIPTOR': os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))}


def genWorkflow(**kwargs):
    """Singularity DICOM to NIfTI conversion workflow."""
    shiva = Workflow('shiva_cmb_swi_prediction')
    shiva.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subjectList = Node(IdentityInterface(
        fields=['subject_id'], mandatory_inputs=True), name="subjectList")
    subjectList.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    dataGrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['swi', 'native_img', 'native_tfm']),
                       name='dataGrabber')
    dataGrabber.inputs.base_directory = kwargs['BASE_DIR']
    dataGrabber.inputs.raise_on_empty = True
    dataGrabber.inputs.sort_filelist = True
    dataGrabber.inputs.template = '%s/%s/*.%s*'
    dataGrabber.inputs.template_args = {'swi': [['subject_id', 'swi', 'nii']],'native_img': [['subject_id', 'native_img', 'nii']],'native_tfm': [['subject_id', 'native_tfm', 'mat']]}

    shiva.connect(subjectList, 'subject_id', dataGrabber, 'subject_id')

    # SHIVA cerebral microbleed predictions
    cmb = Node(Predict(), name="cmb")
    cmb.inputs.snglrt_enable_nvidia = True
    cmb.inputs.snglrt_bind = [(kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),('`pwd`','/mnt/data','rw'),('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),('/bigdata/resources/gcc-10.1.0', '/mnt/gcc', 'ro'),('/bigdata/yrio/Documents/modele/ReferenceModels','/mnt/model', 'ro')]
    cmb.inputs.verbose = True
    cmb.inputs.snglrt_image = '/homes_unix/yrio/singularity/predict_2.sif'
    cmb.inputs.model = '/mnt/model'
    cmb.inputs.descriptor = kwargs['DESCRIPTOR']
    cmb.inputs.out_filename = '/mnt/data/cmbmap.nii.gz'
    shiva.connect(dataGrabber, "swi", cmb, "swi")

     # warp back on swi native space
    seg_to_native = Node(ApplyTransforms(), name="seg_to_native")
    seg_to_native.inputs.interpolation = 'NearestNeighbor'
    seg_to_native.inputs.invert_transform_flags = [True]

    shiva.connect(dataGrabber, 'native_tfm',
                   seg_to_native, 'transforms')
    shiva.connect(cmb, 'segmentation',
                seg_to_native, 'input_image')
    shiva.connect(dataGrabber, 'native_img',
                seg_to_native, 'reference_image')
    return shiva
