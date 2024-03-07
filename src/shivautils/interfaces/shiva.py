"""Interfaces for SHIVA project deep learning segmentation and prediction tools."""
import os

from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec,
                                    CommandLineInputSpec,
                                    CommandLine,)

from shivautils.interfaces.singularity import (SingularityCommandLine,
                                               SingularityInputSpec)

from nipype.interfaces.ants.registration import (RegistrationInputSpec,
                                                 RegistrationOutputSpec,
                                                 Registration)
from nipype.interfaces.ants.resampling import (ApplyTransforms,
                                               ApplyTransformsInputSpec,
                                               ApplyTransformsOutputSpec)

from nipype.interfaces.quickshear import (Quickshear,
                                          QuickshearInputSpec,
                                          QuickshearOutputSpec)


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

    out_filename = traits.Str('synthseg_parc.nii.gz',
                              argstr='--o %s',
                              usedefault=True,
                              desc='Output file path.')

    threads = traits.Int(argstr='--threads %d',
                         desc='Number of threads (only used when using CPUs)')

    robust = traits.Bool(True, argstr='--robust',
                         desc='Perform robust computations for noisy images.',
                         usedefault=True)

    parc = traits.Bool(True, argstr='--parc', desc='Perform parcellation',
                       mandatory=False,
                       usedefault=True)

    cpu = traits.Bool(False, argstr='--cpu', mandatory=False,
                      usedefault=True,
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


class AntsRegistration_Singularity_InputSpec(SingularityInputSpec, RegistrationInputSpec):
    """antsRegistration input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class AntsRegistration_Singularity(Registration, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(AntsRegistration_Singularity, self).__init__()

    input_spec = AntsRegistration_Singularity_InputSpec
    output_spec = RegistrationOutputSpec
    _cmd = Registration._cmd


class AntsApplyTransforms_Singularity_InputSpec(SingularityInputSpec, ApplyTransformsInputSpec):
    """antsApplyTransforms input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class AntsApplyTransforms_Singularity(ApplyTransforms, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(AntsApplyTransforms_Singularity, self).__init__()

    input_spec = AntsApplyTransforms_Singularity_InputSpec
    output_spec = ApplyTransformsOutputSpec
    _cmd = ApplyTransforms._cmd


class Quickshear_Singularity_InputSpec(SingularityInputSpec, QuickshearInputSpec):
    """Quickshear input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class Quickshear_Singularity(Quickshear, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(Quickshear_Singularity, self).__init__()

    input_spec = Quickshear_Singularity_InputSpec
    output_spec = QuickshearOutputSpec
    _cmd = Quickshear._cmd


class Shivai_InputSpec(CommandLineInputSpec):
    """Input arguments structure for the shiva command"""

    in_dir = traits.File(argstr='--in %s',
                         desc='Directory containing the input data for all participants.',
                         exists=True,
                         mandatory=True)

    out_dir = traits.File(argstr='--out %s',
                          desc='Directory where the results will be saved (in and "results" sub-directory).',
                          exists=False,
                          mandatory=True)

    input_type = traits.Enum('standard', 'BIDS',
                             argstr='--input_type %s',
                             desc='Input data structure',
                             mandatory=True)

    db_name = traits.Str(argstr='--db_name %s',
                         desc='Name of the dataset (e.g. "UKBB").',
                         mandatory=False)

    sub_names = traits.List(argstr='--sub_names %s',
                            desc='List of the participants ID to be processed from the input directory.',
                            mandatory=True)

    prediction = traits.Enum("PVS", "PVS2", "WMH", "CMB", "LAC", "all",
                             argstr="--prediction %s",
                             desc='Prediction to run ("PVS", "PVS2", "WMH", "CMB", "LAC", "all")',
                             mandatory=True)

    replace_t1 = traits.Str(argstr='--replace_t1 %s',
                            desc='Data type that will replace the "t1" image.',
                            mandatory=False)

    replace_flair = traits.Str(argstr='--replace_flair %s',
                               desc='Data type that will replace the "flair" image (e.g. "t2s").',
                               mandatory=False)

    replace_swi = traits.Str(argstr='--replace_swi %s',
                             desc='Data type that will replace the "swi" image (e.g. "t2gre").',
                             mandatory=False)

    use_t1 = traits.Bool(argstr='--use_t1',
                         desc='Used to perform segmentation on T1 image when running CMB prediction alone (otherwise will use SWI for the seg).',
                         mandatory=False)

    brain_seg = traits.Enum("shiva", "shiva_gpu", "synthseg", "synthseg_cpu", "synthseg_precomp", "premasked", "custom",
                            argstr='--brain_seg %s',
                            desc='Type of segmentation to run. Chose among: "shiva", "shiva_gpu", "synthseg", "synthseg_cpu", "synthseg_precomp", "premasked", "custom"',
                            mandatory=True)

    synthseg_threads = traits.Int(argstr='--synthseg_threads %d',
                                  desc='Number of thread to run with "synthseg_cpu".',
                                  mandatory=False)

    custom_LUT = traits.File(argstr='--custom_LUT %s',
                             desc='Look-up table file to pair with the custom segmentation when used ("custom" brain_seg)',
                             exists=True,
                             mandatory=False)

    anonymize = traits.Bool(argstr='--anonymize',
                            desc='Anonymize the report',
                            mandatory=False)

    synthseg_precomp = traits.Bool(argstr='--synthseg_precomp',
                                   desc='Specify that the synthseg segmentation was run beforehand (typically using precomp_synthseg) and is stored in the results directory',
                                   mandatory=False)

    keep_all = traits.Bool(argstr='--keep_all',
                           desc='Keep all intermediary file and provenance files',
                           mandatory=False)
    debug = traits.Bool(argstr='--debug',
                        desc='Like --keep_all plus stop on first crash',
                        mandatory=False)
    remove_intermediates = traits.Bool(argstr='--remove_intermediates',
                                       desc='Removes the folder containing all the intermediary steps, keeping only the "results" folder.',
                                       mandatory=False)

    config = traits.File(argstr='--config %s',
                         desc='Configuration file (.yml) containing the information and parameters for the pipeline and AI models',
                         mandatory=True)


class Shivai_OutputSpec(TraitedSpec):
    """Shivai ports."""
    result_dir = traits.File(desc='Folder where the results are stored',
                             exists=True)


class Shivai(CommandLine):
    """Runs the shiva workflow (and thus must be called through the Shivai_Singularity inteface, not this one)."""

    input_spec = Shivai_InputSpec
    output_spec = Shivai_OutputSpec
    _cmd = 'shiva'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["result_dir"] = os.path.join(os.path.abspath(str(self.inputs.out_dir)), 'results')
        return outputs


class Shivai_Singularity_InputSpec(Shivai_InputSpec, QuickshearInputSpec):
    """Quickshear input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class Shivai_Singularity(Shivai, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(Shivai_Singularity, self).__init__()

    input_spec = Shivai_Singularity_InputSpec
    output_spec = Shivai_OutputSpec
    _cmd = Shivai._cmd + ' --containerized_all'
