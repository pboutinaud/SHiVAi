#!/usr/bin/env python
"""
Standalone workflow that must be installed in the synthseg.sif image

"""

import argparse
import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.base import (traits, TraitedSpec,
                                    CommandLineInputSpec,
                                    CommandLine,)


class SynthSegInputSpec(CommandLineInputSpec):
    """Input arguments structure for Freesurfer synthseg."""

    input = traits.File(argstr='--i %s',
                        desc='The structural image of the subject (use an image input file, not a file list or folder).',
                        exists=True)

    out_filename = traits.Str('synthseg_parc.nii.gz', argstr='--o %s',
                              desc='Output file path.')

    threads = traits.Int(10, argstr='--threads %d',
                         desc='Number of threads',
                         usedefault=True)

    robust = traits.Bool(True, argstr='--robust',
                         desc='Perform robust computations for noisy images.',
                         usedefault=True)

    parc = traits.Bool(True, argstr='--parc', desc='Perform parcellation',
                       mandatory=False,
                       usedefault=True)

    cpu = traits.Bool(False, argstr='--cpu', mandatory=False,
                      desc='Use CPU instead of GPU for computations')

    vol = traits.Str('volumes.csv', argstr='--vol %s', mandatory=False,
                     desc='Path to a CSV file where volumes for all segmented regions will be saved.')

    qc = traits.Str('qc.csv', argstr='--qc %s',
                    desc='Path to a CSV file where QC scores will be saved.', mandatory=False)


class SynthSegOutputSpec(TraitedSpec):
    """Freesurfer synthseg output ports."""
    segmentation = traits.File(desc='The segmentation regions image',
                               exists=True)

    qc = traits.File(desc='The quality control csv file',
                     exists=False)

    volumes = traits.File(desc='The volumetry results csv file',
                          exists=False)


class SynthSeg(CommandLine):
    """Segment brain regions with Freesurfer synthseg."""

    input_spec = SynthSegInputSpec
    output_spec = SynthSegOutputSpec
    _cmd = 'mri_synthseg'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        outputs["qc"] = os.path.abspath(os.path.split(str(self.inputs.qc))[1])
        outputs["volumes"] = os.path.abspath(os.path.split(str(self.inputs.vol))[1])
        return outputs


def set_wf_shapers(predictions):
    """
    Set with_t1, with_flair, and with_swi with the corresponding value depending on the
    segmentations (predictions) that will be done.
    The tree boolean variables are used to shape the main and postproc workflows
    (e.g. if doing PVS and CMB, the wf will use T1 and SWI)
    """
    # Setting up the different cases to build the workflows (should clarify things up)
    if any(pred in predictions for pred in ['PVS', 'PVS2', 'WMH', 'LAC']):  # all which requires T1
        with_t1 = True
    else:
        with_t1 = False
    if any(pred in predictions for pred in ['PVS2', 'WMH', 'LAC']):
        with_flair = True
    else:
        with_flair = False
    if 'CMB' in predictions:
        with_swi = True
    else:
        with_swi = False
    return with_t1, with_flair, with_swi


def synthsegParser():
    return


def main():
    parser = synthsegParser()
    args = parser.parse_args()

    if args.input_type == 'standard' or args.input_type == 'BIDS':
        subject_directory = args.input
        out_dir = args.output

    wfargs = {
        'DATA_DIR': subject_directory,
        'BASE_DIR': out_dir,
        'PREDICTION': args.prediction,
        'SUBJECT_LIST': args.sub_list,
        'SYNTHSEG_ON_CPU': args.synthseg_cpu,
    }

    with_t1, with_flair, with_swi = set_wf_shapers(wfargs['PREDICTION'])

    synthseg_wf = Workflow('synthseg_wf')
    synthseg_wf.base_dir = wfargs['BASE_DIR']

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', wfargs['SUBJECT_LIST'])

    # Initialize the datagrabber
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1']),
        name='datagrabber')
    datagrabber.inputs.base_directory = wfargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'

    synthseg = Node(SynthSeg(),
                    name='synthseg')
    synthseg.inputs.cpu = wfargs['SYNTHSEG_ON_CPU']

    synthseg_wf.connect(subject_iterator, 'subject_id', datagrabber, 'subject_id')
    synthseg_wf.connect(datagrabber, 'img1', synthseg, 'input')

    if with_t1:
        if args.replace_t1:
            acq = [('img1', args.replace_t1)]
        else:
            acq = [('img1', 't1')]
    elif with_swi and not with_t1:
        if args.replace_swi:
            acq = [('img1', args.replace_swi)]
        else:
            acq = [('img1', 'swi')]

    # TODO: replace this with the actual code
    synthseg_wf = update_wf_grabber(synthseg_wf, args.input_type, acq)

    # TODO: Sink
