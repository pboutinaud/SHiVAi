"""Interfaces for SHIVA project deep learning segmentation and prediction tools."""
import os
from string import Template
import numpy as np

from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath)
from nipype.interfaces.spm.base import (SPMCommand, SPMCommandInputSpec)
from nipype.utils.filemanip import ensure_list, simplify_list
from shivautils.interfaces.singularity_interface import CommandLine, SingularityCommandLine, SingularityInputSpec, CommandLineInputSpec


# workflow specific class
class CustomIntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True,
                          mandatory=True)

    brain_mask = traits.File(exists=True,
                             mandatory=True, desc='')

    out_file = traits.File('intensity_normalized_image.nii', exists=False, desc='Output file name.',
                           usedefault=True, mandatory=True)

    output_dir = traits.Str('./', mandatory=True, desc='Path to output directory.',
                            usedefault=True)

    spm_path = traits.Directory('/srv/shares/softs/spm12-full',
                                exists=True, desc='SPM12 folder',
                                mandatory=True,
                                usedefault=True)


class CustomIntensityNormalizationOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True)


class CustomIntensityNormalization(BaseInterface):
    """Denoising algorithm for T1/T2flair images in MATLAB."""

    input_spec = CustomIntensityNormalizationInputSpec
    output_spec = CustomIntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_file=self.inputs.in_file,
                 brain_mask=self.inputs.brain_mask,
                 output_dir=self.inputs.output_dir,
                 out_file=self.inputs.out_file,
                 spm_path=self.inputs.spm_path)

        # From V2 function with gzip at the end (32 bits volumes are big!)
        script = Template("""
                addpath('$spm_path')

                fprintf('%s \\n', 'About to normalize your image between 0 and 1.')

                filename = split('$in_file', '/');
                filename = string(filename(length(filename)));
                filename = split(filename, '.');
                filename = string(filename(1));

                % open and read the volume
                vol0 = spm_vol('$in_file');
                pipo = spm_read_vols(vol0);

                % open and read the mask
                vol1 = spm_vol('$brain_mask');
                mask = spm_read_vols(vol1);

                % rescale between 0 and the 99th percentile 
                b = find(mask(:) > 0);
                tab = pipo;
                scale_factor = prctile(tab(b),99); % 99th percentile
                min_tab=0;
                max_tab=1.3; 
                tab= tab/scale_factor;

                % everything below 0 equals to 0
                % and above 1.3 equals to 1.3
                tmp = find(tab < min_tab);
                tab(tmp) = min_tab;
                tmp = find(tab> max_tab);
                tab(tmp) = max_tab;

                % normalizing between [0 1]
                tab = (tab - min(tab(:))) / ( max(tab(:)) - min(tab(:)));

                % writing final normalized volume
                V3=vol0;
                V3.dt = [16 0];
                file_output = strcat('$output_dir', '/', '$out_file');
                V3.fname = char(file_output);
                spm_write_vol(V3, tab);
                system(append("gzip", " ", file_output))
                exit;
                """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file + '.gz')
        return outputs


class SPMApplyDeformationInput(SPMCommandInputSpec):
    in_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        field="out{1}.pull.fnames",
        desc="Files on which deformation is applied",
    )

    deformation_field = File(
        exists=True,
        mandatory=True,
        field="comp{1}.def",
        desc="SPM deformation file"
    )
    target = File(
        exists=True, 
        mandatory=True,
        field="comp{2}.id.space",
        desc="File defining target space"
    )

    interpolation = traits.Range(
        low=0, high=7, field="out{1}.pull.interp", desc="degree of b-spline used for interpolation"
    )


class SPMApplyDeformationOutput(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True), desc="Transformed files")


class SPMApplyDeformation(SPMCommand):
    """
    Since there is a bug in nipype ApplyInverseDeformation interface (Issue #3326)
    that has not been incorporated yet as of version 1.6.1, this is a custom interface
    to apply deformation field.

    Also, note that this is for applying def field to a single or multiple 3D image,
    and not for 4D image, as in the nipype version.

    Examples
    --------
    >>> from workflows.wf_utils.interface import SPMApplyDeformation
    >>> inv = SPMApplyDeformation()
    >>> inv.inputs.in_files = 'template_ROI.nii'
    >>> inv.inputs.deformation_field = 'iy_structural.nii'
    >>> inv.inputs.target = 'structural.nii'
    >>> inv.run() # doctest: +SKIP
    """
    input_spec = SPMApplyDeformationInput
    output_spec = SPMApplyDeformationOutput
    _jobtype = "util"
    _jobname = "defs"

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == "in_files":
            return np.array(ensure_list(val), dtype=object)
        if opt == "target":
            return np.array([simplify_list(val)], dtype=object)
        if opt == "deformation_field":
            return np.array([simplify_list(val)], dtype=object)
        return val

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_files"] = []
        for filename in self.inputs.in_files:
            _, fname = os.path.split(filename)
            outputs["out_files"].append(os.path.realpath("w%s" % fname))
        return outputs


class PredictSingularityInputSpec(SingularityInputSpec):
    """PredictVRS input specification.

    Inherits from Singularity command line fields.
    """
    models = traits.List(traits.File(exists=True),
                         argstr='-m %s',
                         desc='Model files in h5 format.',
                         mandatory=False,
                        )

    t1 = traits.File(argstr='--t1 %s',
                     desc='The T1W image of the subject.',
                     exists=True)

    flair = traits.File(argstr='--flair %s',
                        desc='The FLAIR image of the subject.',
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
                             desc='File info about model and validation',
                             mandatory=True)
    
    gpu_number = traits.Int(argstr='--gpu %d',
                            desc='GPU to use if several GPUs are available.',
                            mandatory=False)

    verbose = traits.Bool(True,
                          argstr='--verbose',
                          desc='Verbose output',
                          mandatory=False)

    out_filename = traits.Str('/mnt/data/map.nii.gz',
                              argstr='-o %s',
                              desc='Output filename.',
                              usedefault=True)


class PredictSingularityOutputSpec(TraitedSpec):
    segmentation = traits.File(desc='The segmentation image',
                                   exists=True)


class PredictSingularity(SingularityCommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net.
    """
    input_spec = PredictSingularityInputSpec
    output_spec = PredictSingularityOutputSpec
    _cmd = 'predict.py'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        return outputs


class PredictInputSpec(BaseInterfaceInputSpec):
    """Predict input specification.

    Inherits from Singularity command line fields.
    """
    models = traits.List(traits.File(exists=True),
                         argstr='-m %s',
                         desc='Model files in h5 format.',
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
                             desc='File info about model and validation',
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


class PredictOutputSpec(TraitedSpec):
    segmentation = traits.File(desc='segmentation image',
                                   exists=True)


class Predict(CommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net.
    """
    input_spec = PredictInputSpec
    output_spec = PredictOutputSpec
    _cmd = 'predict.py'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        return outputs
    


class SynthSegInputSpec(CommandLineInputSpec):
    """Segmentation of brain regions

    Inherits from Singularity command line fields.
    """
    i = traits.File(argstr='--i %s',
                     desc='The T1W image of the subject.',
                     exists=True)
    
    out_filename = traits.Str(argstr='--o %s',
                              desc='path file output')




class SynthSegOutputSpec(TraitedSpec):
    segmentation = traits.File(desc='The segmentation regions image',
                                   exists=True)


class SynthSeg(CommandLine):
    """Run predict to segment regions from reformated structural images.

    Uses Freesurfer Command Line
    """
    input_spec = SynthSegInputSpec
    output_spec = SynthSegOutputSpec
    _cmd = 'mri_synthseg --threads 20 --parc  --robust'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.split(str(self.inputs.out_filename))[1])
        return outputs
    