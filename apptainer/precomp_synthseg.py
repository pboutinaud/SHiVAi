#!/usr/bin/env python
"""
Standalone workflow that must be installed in the synthseg.sif image.
As it is standalone and thus does not import functions or class from shivai,
any change made in shivai must be manually re-implemented here too.
"""

import argparse
import os
import json
import yaml
import shutil
from nipype import config
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec,
                                    CommandLine,)


class SynthSegInputSpec(BaseInterfaceInputSpec):  # CommandLineInputSpec ?
    """Input arguments structure for Freesurfer synthseg."""

    input = traits.File(argstr='--i %s',
                        desc='The structural image of the subject (use an image input file, not a file list or folder).',
                        exists=True)

    out_filename = traits.Str('synthseg_parc.nii.gz', argstr='--o %s',
                              usedefault=True,
                              desc='Output file path.')

    threads = traits.Int(argstr='--threads %d',
                         desc='Number of threads')

    robust = traits.Bool(True, argstr='--robust',
                         desc='Perform robust computations for noisy images.',
                         usedefault=True)

    parc = traits.Bool(True, argstr='--parc', desc='Perform parcellation',
                       mandatory=False,
                       usedefault=True)

    cpu = traits.Bool(False, argstr='--cpu', mandatory=False,
                      desc='Use CPU instead of GPU for computations')

    vol = traits.Str('volumes.csv', argstr='--vol %s', usedefault=True,
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
    DESCRIPTION = """
    Small workflow running SynthSeg on input images. It is made to be used before running shiva with
    the --synthseg_precomp argument (meaning that shiva will look for the Synthseg parcellation in the
    output directory instead of computing it). This is necessary when running a fully contained shiva
    processing, as it can't run the syntheg container from the shiva container.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--in', dest='in_dir',
                        help='Folder path with files, BIDS structure folder path or JSON formatted extract of the Slicer plugin',
                        metavar='path/to/existing/folder/structure',
                        required=True)

    parser.add_argument('--out', dest='out_dir',
                        type=str,
                        help='Output folder path (nipype working directory)',
                        metavar='path/to/nipype_work_dir',
                        required=True)

    parser.add_argument('--input_type',
                        choices=['standard', 'BIDS', 'json'],
                        help="Way to grab and manage nifti files : 'standard', 'BIDS' or 'json'",
                        default='standard')

    sub_lists_args = parser.add_mutually_exclusive_group()
    sub_lists_args.add_argument('--sub_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to be processed. The IDs must be '
                                      'the same as the ones given in the input folder. In the file, the IDs can be separated '
                                      'by a whitespace, a new line, or any of the following characters [ "," ";" "|" ] '
                                      '(or a combination of those). If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--sub_names',
                                nargs='+',
                                required=False,
                                help=('List of participant IDs to be processed. With this option, the IDs are given directly '
                                      'in the command line, separated by a white-space, and must be the same as the ones given '
                                      'in the input folder. If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    sub_lists_args.add_argument('--exclusion_list',
                                type=str,
                                required=False,
                                help=('Text file containing the list of participant IDs to NOT be processed. This option can be '
                                      'used when processing all the data in the input folder except for a few (because they have '
                                      'faulty data for exemple).\n'
                                      'In the file, the syntax is the same as for --sub_list\n.'
                                      'If none of --sub_list, --sub_names, or --exclusion_list '
                                      'are used, all the participants in the input folder will be processed'))

    parser.add_argument('--prediction',
                        choices=['PVS', 'PVS2', 'WMH', 'CMB', 'LAC', 'all'],
                        nargs='+',
                        help=("Choice of the type of prediction (i.e. segmentation) you want to compute.\n"
                              "A combination of multiple predictions (separated by a white space) can be given.\n"
                              "- 'PVS' for the segmentation of perivascular spaces using only T1 scans\n"
                              "- 'PVS2' for the segmentation of perivascular spaces using both T1 and FLAIR scans\n"
                              "- 'WMH' for the segmentation of white matter hyperintensities (requires both T1 and FLAIR scans)\n"
                              "- 'CMB' for the segmentation of cerebral microbleeds (requires SWI scans)\n"
                              "- 'LAC' for the segmentation of cerebral lacunas (requires both T1 and FLAIR scans)\n"
                              "- 'all' for doing 'PVS2', 'WMH', and 'CMB' segmentation (requires T1, FLAIR, and SWI scans)"),
                        default=['PVS'])

    parser.add_argument('--replace_t1',
                        type=str,
                        metavar='img_type',
                        help=('Image type to be used instead of T1w for PVS (and PVS2), WMH, and Lacuna segmentations.\n'
                              '(Note that part of the labels may keep the "t1" notation instead of the image type you '
                              'specified)'))

    parser.add_argument('--replace_swi',
                        type=str,
                        metavar='img_type',
                        help=('Image type to be used instead of SWI for CMB segmentations.\n'
                              '(Note that part of the labels may keep the "swi" notation instead of the image type you '
                              'specified)'))

    parser.add_argument('--use_t1',
                        action='store_true',
                        help=('Can be used when predicting CMBs only (so only expecting SWI acquisitions) while T1 acquisitions '
                              'are available. This enable the CMB preprocessing steps using t1 for the brain parcelization. '
                              'This option can also be used with "replace_t1" to use another type of acquisition.'))

    parser.add_argument('--synthseg_cpu',
                        action='store_true',
                        help='If selected, will run Synthseg using CPUs instead of GPUs')

    parser.add_argument('--threads',
                        default=8,
                        type=int,
                        help='Number of threads to create for parallel computation when using --synthseg_cpu (default is 8).')

    parser.add_argument('--run_plugin',
                        default='Linear',
                        help=('Type of plugin used by Nipype to run the workflow.\n'
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))

    parser.add_argument('--run_plugin_args',  # hidden feature: you can also give a json string '{"arg1": val1, ...}'
                        type=str,
                        help=('Configuration file (.yml) for the plugin used by Nipype to run the workflow.\n'
                              'It will be imported as a dictionary and given plugin_args '
                              '(see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html '
                              'for more details )'))

    file_management = parser.add_mutually_exclusive_group()

    file_management.add_argument('--keep_all',
                                 action='store_true',
                                 help='Keep all intermediary file')

    file_management.add_argument('--debug',
                                 action='store_true',
                                 help='Like --keep_all plus stop on first crash')

    file_management.add_argument('--remove_intermediates',
                                 action='store_true',
                                 help=('Remove the folder containing all the intermediary steps, keeping only the "results" folder.\n'
                                       'Obvioulsy not compatible with debugging or re-running the workflow.'))
    return parser


def parse_sub_list_file(filename):
    list_path = os.path.abspath(filename)
    sub_list = []
    sep_chars = [' ', ';', '|']
    if not os.path.exists(list_path):
        raise FileNotFoundError(f'The participant list file was not found at the given location: {list_path}')
    with open(list_path) as f:
        lines = f.readlines()
    for line in lines:
        line_s = line.strip('\n')
        # replacing potential separators with commas
        for sep in sep_chars:
            if sep in line_s:
                line_s = line_s.replace(sep, ',')
        subs = line_s.split(',')
        sub_list += [s.strip() for s in subs if s]
    return sub_list


def main():
    parser = synthsegParser()
    args = parser.parse_args()

    # Parse the prediction
    if 'all' in args.prediction:
        if 'PVS' in args.prediction:
            args.prediction = ['PVS', 'WMH', 'CMB', 'LAC']
        else:
            args.prediction = ['PVS2', 'WMH', 'CMB', 'LAC']
    if not isinstance(args.prediction, list):  # When only one input
        args.prediction = [args.prediction]

    # Checks and parsing of subjects
    subject_list = os.listdir(args.in_dir)
    if args.sub_list is None and args.sub_names is None:
        if args.exclusion_list:
            args.exclusion_list = parse_sub_list_file(args.exclusion_list)
            subject_list = sorted(list(set(subject_list) - set(args.exclusion_list)))
        args.sub_list = subject_list
    else:
        if args.sub_list:
            args.sub_list = parse_sub_list_file(args.sub_list)
        elif args.sub_names:
            args.sub_list = args.sub_names
        subs_not_in_dir = set(args.sub_list) - set(subject_list)
        if len(subs_not_in_dir) == len(args.sub_list):
            raise parser.error('None of the participant IDs given in the sub_list file was found in the input directory.\n'
                               f'Participant IDs given: {args.sub_list}\n'
                               f'Participant available: {subject_list}')
        elif len(subs_not_in_dir) > 0:
            raise parser.error(f'Some participants where not found in the input directory: {sorted(list(subs_not_in_dir))}')

    # Parse data structure type
    if args.input_type == 'standard' or args.input_type == 'BIDS':
        subject_directory = args.in_dir
        out_dir = args.out_dir
    elif args.input_type == 'json':
        raise NotImplementedError(
            'Sorry, using the json input format has not been implemented here. '
            'Please use "standard" or "BIDS".'
        )

    # Parse the plugin arguments
    if args.run_plugin_args:
        if os.path.isfile(args.run_plugin_args):
            with open(args.run_plugin_args, 'r') as file:
                yaml_content = yaml.safe_load(file)
            args.run_plugin_args = yaml_content
        else:
            try:
                args.run_plugin_args = json.loads(args.run_plugin_args)
            except json.JSONDecodeError:
                raise ValueError('The "--run_plugin_args" argument was not recognised as a file path (file not existing) '
                                 f'nor a json string (bad formatting possibly). Input string: {args.run_plugin_args}')

    else:
        args.run_plugin_args = {}

    with_t1, with_flair, with_swi = set_wf_shapers(args.prediction)

    if args.use_t1:  # Override default with_t1 value
        with_t1 = True

    if with_t1:
        if args.replace_t1:
            acq = args.replace_t1
        else:
            acq = 't1'
    elif with_swi and not with_t1:
        if args.replace_swi:
            acq = args.replace_swi
        else:
            acq = 'swi'

    synthseg_wf = Workflow('synthseg_wf')
    synthseg_wf.base_dir = out_dir

    # Start by initializing the iterable
    subject_iterator = Node(
        IdentityInterface(
            fields=['subject_id'],
            mandatory_inputs=True),
        name="subject_iterator")
    subject_iterator.iterables = ('subject_id', args.sub_list)

    # Initialize the datagrabber
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['img1']),
        name='datagrabber')
    datagrabber.inputs.base_directory = subject_directory
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'
    if args.input_type in ['standard', 'json']:
        datagrabber.inputs.field_template = {'img1': f'%s/{acq}/*.nii*'}
        datagrabber.inputs.template_args = {'img1': [['subject_id']]}
    elif args.input_type == 'BIDS':
        datagrabber.inputs.field_template = {'img1': f'%s/anat/%s_{acq.upper()}*.nii*'}
        datagrabber.inputs.template_args = {'img1': [['subject_id', 'subject_id']]}

    pred_plugin_args = {'sbatch_args': f'--nodes 1 --cpus-per-task {args.threads} --gpus 1'}
    pred_plugin_args.update(args.run_plugin_args)
    synthseg = Node(SynthSeg(),
                    name='synthseg')
    synthseg.inputs.cpu = args.synthseg_cpu
    synthseg.inputs.threads = args.threads
    synthseg.plugin_args = pred_plugin_args

    # Initializing the data sinks
    sink_node_subjects = Node(DataSink(), name='sink_node_subjects')
    sink_node_subjects.inputs.base_directory = os.path.join(out_dir, 'results')
    sink_node_subjects.inputs.substitutions = [
        ('_subject_id_', ''),
    ]

    synthseg_wf.connect(subject_iterator, 'subject_id', datagrabber, 'subject_id')
    synthseg_wf.connect(datagrabber, 'img1', synthseg, 'input')
    synthseg_wf.connect(synthseg, 'segmentation', sink_node_subjects, 'shiva_preproc.synthseg')
    synthseg_wf.connect(synthseg, 'volumes', sink_node_subjects, 'shiva_preproc.synthseg.@vol')

    if args.keep_all or args.debug:
        config.enable_provenance()
        synthseg_wf.config['execution'] = {'remove_unnecessary_outputs': 'False'}
    if args.debug:
        synthseg_wf.config['execution']['stop_on_first_crash'] = 'True'
    synthseg_wf.run(plugin=args.run_plugin, plugin_args=args.run_plugin_args)

    # Remove intermediate files if asked
    if args.remove_intermediates:
        workflow_dir = os.path.join(out_dir, 'synthseg_wf')
        shutil.rmtree(workflow_dir)


if __name__ == "__main__":
    main()
