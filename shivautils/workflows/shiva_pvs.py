"""SHIVA project perivascular spaces 3D U-Net prediction pipeline.

@author: Pierre-Yves Herve
@contact: pyherve@fealinx.com
"""
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber
from pyplm.interfaces.shiva import Predict 
import os


# dummy args are a set of dummy parameters. The real kwargs come
# from teamcenter.
# SUBJECT_LIST and BASE_DIR come from the dynamic parameters
dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~')),
              'DESCRIPTOR': os.path.normpath(os.path.join(os.path.expanduser('~'),'.swomed', 'default_config.ini'))}


def genWorkflow(**kwargs):
    """Singularity DICOM to NIfTI conversion workflow."""
    shiva = Workflow('shiva_pvs_t1_prediction')
    shiva.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subjectList = Node(IdentityInterface(
                            fields=['subject_id'],
                            mandatory_inputs=True),
                       name="subjectList")
    subjectList.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    dataGrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['t1']),
                       name='dataGrabber')
    dataGrabber.inputs.base_directory = kwargs['BASE_DIR']
    dataGrabber.inputs.raise_on_empty = True
    dataGrabber.inputs.sort_filelist = True
    dataGrabber.inputs.template = '%s/%s/*.nii*'
    dataGrabber.inputs.template_args = {'t1': [['subject_id', 't1']]}

    shiva.connect(subjectList, 'subject_id', dataGrabber, 'subject_id')

    # SHIVA VRS predictions
    pvs = Node(Predict(), name="pvs")
    pvs.inputs.snglrt_enable_nvidia = True
    pvs.inputs.snglrt_bind = [(kwargs['BASE_DIR'],kwargs['BASE_DIR'],'rw'),('`pwd`','/mnt/data','rw'),('/bigdata/resources/cudas/cuda-11.2','/mnt/cuda','ro'),('/bigdata/resources/gcc-10.1.0','/mnt/gcc', 'ro'),('/bigdata/yrio/Documents/modele/ReferenceModels','/mnt/model','ro')]
    pvs.inputs.verbose = True
    pvs.inputs.snglrt_image = '/homes_unix/yrio/singularity/predict_2.sif'
    pvs.inputs.model = '/mnt/model'
    pvs.inputs.descriptor = kwargs['DESCRIPTOR']
    pvs.inputs.out_filename = '/mnt/data/pvsmap_t1.nii.gz'
    shiva.connect(dataGrabber, "t1", pvs, "t1")
    return shiva
