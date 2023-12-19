"""Interfaces for SHIVA project deep learning segmentation and prediction tools."""
import os

from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec)

from shivautils.interfaces.singularity import (CommandLine, SingularityCommandLine,
                                               SingularityInputSpec, CommandLineInputSpec)
from nipype.interfaces.ants.registration import RegistrationInputSpec, RegistrationOutputSpec, Registration


class PredictInputSpec(BaseInterfaceInputSpec):
    """Predict input specification."""
    models = traits.List(traits.File(exists=True),
                         argstr='-m %s',
                         desc='Model files.',
                         mandatory=False,
                         )

    t1 = traits.File(argstr='--t1 %s',
                     desc='The T1W image of the subject.',
                     mandatory=False,
                     exists=True)

    flair = traits.File(argstr='--flair %s',
                        desc='The FLAIR image of the subject.',
                        mandatory=False,
                        exists=True)

    swi = traits.File(argstr='--swi %s',
                      desc='The SWI image of the subject.',
                      mandatory=False,
                      exists=True)

    t2 = traits.File(argstr='--t2 %s',
                     desc='The T2 image of the subject.',
                     mandatory=False,
                     exists=True)

    model = traits.Directory('/mnt/model',
                             argstr='--model %s',
                             exists=False,
                             desc='Folder containing hdf5 model files.',
                             usedefault=True
                             )

    descriptor = traits.File(argstr='-descriptor %s',
                             exists=True,
                             desc='File information about models for validation',
                             mandatory=True)

    gpu_number = traits.Int(argstr='--gpu %d',
                            desc='GPU to use if several GPUs are available.',
                            mandatory=False)

    verbose = traits.Bool(True,
                          argstr='--verbose',
                          desc='Verbose output',
                          mandatory=False)

    out_filename = traits.Str('map.nii.gz',
                              argstr='-o %s',
                              desc='Output filename.',
                              usedefault=True)


class PredictSingularityInputSpec(SingularityInputSpec, PredictInputSpec):
    """PredictVRS input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """


class PredictOutputSpec(TraitedSpec):
    segmentation = traits.File(desc='The segmentation image',
                               exists=True)


class Predict(CommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net.
    """
    input_spec = PredictInputSpec
    output_spec = PredictOutputSpec
    _cmd = 'shiva_predict'  # shivautils.scripts.predict:main

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        return outputs


class PredictSingularity(SingularityCommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net with tensorflow-gpu in a container (apptainer/singularity).
    """
    input_spec = PredictSingularityInputSpec
    output_spec = PredictOutputSpec
    _cmd = 'shiva_predict'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        return outputs


class SynthSegInputSpec(CommandLineInputSpec):
    """Input arguments structure for Freesurfer synthseg."""

    input = traits.File(argstr='--i %s',
                        desc='The structural image of the subject (use an image input file, not a file list or folder).',
                        exists=True)

    out_filename = traits.Str('segmented.nii.gz', argstr='--o %s',
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

    vol = traits.Str('volumes.csv', argstr='--vol %s', mandatory='False',
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


class SynthsegSingularityInputSpec(SingularityInputSpec, SynthSegInputSpec):
    """Synthseg input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """


class SynthsegSingularity(SingularityCommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net with tensorflow-gpu in a container (apptainer/singularity).
    """
    input_spec = SynthsegSingularityInputSpec
    output_spec = SynthSegOutputSpec
    _cmd = 'mri_synthseg'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        outputs["qc"] = os.path.abspath(os.path.split(str(self.inputs.qc))[1])
        outputs["volumes"] = os.path.abspath(os.path.split(str(self.inputs.vol))[1])
        return outputs


class AntsRegistrationSingularityInputSpec(SingularityInputSpec, RegistrationInputSpec):
    """antsRegistration input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class AntsRegistrationSingularity(Registration, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(AntsRegistrationSingularity, self).__init__()

    input_spec = AntsRegistrationSingularityInputSpec
    output_spec = RegistrationOutputSpec
    _cmd = Registration._cmd
