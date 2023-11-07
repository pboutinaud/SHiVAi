"""
Contains custom interfaces wrapping scripts/functions used by the nipype workflows.

@author: atsuchida
@modified by iastafeva (added niimath)
"""
import os
import os.path as op

import numpy as np
import nibabel as nib
import pandas as pd

from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec,
                                    CommandLine, CommandLineInputSpec,
                                    InputMultiPath, OutputMultiPath)
from nipype.interfaces.spm.base import (SPMCommand, SPMCommandInputSpec)

from nipype.interfaces.matlab import MatlabCommand
from nipype.utils.filemanip import ensure_list, simplify_list

from string import Template
from shivautils.stats import transf_from_affine


class CustomIntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    brain_mask = File(exists=True, mandatory=True)
    out_file = File()
    svn_dir = traits.Str(mandatory=True, desc='Path to your SVN directory.')
    output_dir = traits.Str(mandatory=True, desc='Path to output directory.')


class CustomIntensityNormalizationOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class CustomIntensityNormalization(BaseInterface):
    """ Denoising algorithm for T1/T2flair images in MATLAB """

    input_spec = CustomIntensityNormalizationInputSpec
    output_spec = CustomIntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_file=self.inputs.in_file,
                 svn_dir=self.inputs.svn_dir,
                 brain_mask=self.inputs.brain_mask,
                 output_dir=self.inputs.output_dir,
                 out_file=self.inputs.out_file)
        script = Template("""
                addpath('$svn_dir/workflows/wf_utils/')
                intensity_normalization('$in_file','$brain_mask','$output_dir','$out_file')
                exit;
                """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class MaskOverlayQCplotInputSpec(CommandLineInputSpec):
    bg_im_file = traits.Str(mandatory=True,
                            desc='Background ref image file (typically T1 brain)',
                            argstr='%s',
                            position=1)
    mask_file = traits.Str(mandatory=True,
                           desc='Brain mask',
                           argstr='%s',
                           position=2)
    transparency = traits.Enum(0, 1,
                               argstr='%d',
                               desc='Set transparency (0: solid, 1:transparent)',
                               mandatory=True,
                               position=3)
    out_file = traits.Str('mask_overlay.png',
                          mandatory=False,
                          desc='Output png filename',
                          argstr='%s',
                          position=4)
    bg_max = traits.Float(argstr="%.3f",
                          mandatory=False,
                          desc='Optionally specifies the bg img intensity range as a percentile',
                          position=5)


class MaskOverlayQCplotOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class MakeDistanceMapInputSpec(CommandLineInputSpec):

    # niimath ventricle_mask  -binv -edt output
    in_file = traits.Str(mandatory=True,
                         desc='Object segmentation mask (isotropic)',
                         argstr='%s',
                         position=1)

    out_file = traits.Str('distance_map.nii.gz',
                          mandatory=True,
                          desc='Output filename for ventricle distance maps',
                          argstr='-binv -edt %s',
                          position=2)


class MakeDistanceMapOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class MakeDistanceMap(CommandLine):
    """Create distance maps using ventricles binarized maps (niimaths)."""

    _cmd = 'niimath'

    input_spec = MakeDistanceMapInputSpec
    output_spec = MakeDistanceMapOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class SynthSegSegmentationInputSpec(CommandLineInputSpec):

    # like: os.system("mri_synthseg --i /data/path --o /save/dir --vol /save/dir --qc /save/dir")

    im_file = traits.Str(mandatory=True,
                         desc='Path to MRI image / path to folder with MRI images',
                         argstr='%s',
                         position=1)

    out_file = traits.Str('SynthSegSegmentation_111.nii.gz',

                          mandatory=True,
                          desc='Output SynthSeg map',
                          argstr='%s',
                          position=2)

    out_vol = traits.Str('vol.csv',

                         mandatory=True,
                         desc='Output SynthSeg map',
                         argstr='%s',
                         position=3)

    out_qc = traits.Str('qc.csv',
                        mandatory=True,
                        desc='Output SynthSeg map',
                        argstr='%s',
                        position=4)


class SynthSegSegmentationOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    out_qc = File(exists=True)
    out_vol = File(exists=True)


class SynthSegSegmentationMap(CommandLine):
    """
    Segmentation of MRI images using SynthSeg.
    """
    import shivautils.interfaces as wf_interfaces
    p = op.dirname(wf_interfaces.__file__)
    _cmd = op.join(p, 'SynthSegSegmentation.sh')
    input_spec = SynthSegSegmentationInputSpec
    output_spec = SynthSegSegmentationOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_vol'] = op.abspath(self.inputs.out_vol)
        outputs['out_qc'] = op.abspath(self.inputs.out_qc)

        return outputs


class CoregQCInputSpec(CommandLineInputSpec):

    """Create coregistrations qc endpoints:

     1) create coreg isocontour image
     2) compute the cost function of FLIRT coregistration

     coreg_QC.sh <COREGISTERED_IMG> <REF(T1)_BRAIN> <REF(T1)_BRAIN_MASK> <OUTPUT_BASENAME>

     creates <OUTPUT_BASENAME>.png with isocontours image and <OUTPUT_BASENAME>.txt
     with costfunction value
    """

    in_file = traits.Str(mandatory=True,
                         desc='Path to coregistered image',
                         argstr='%s',
                         position=1)

    ref_file = traits.Str(
        mandatory=True,
        desc='Path to reference image (T1w)',
        argstr='%s',
        position=2)

    ref_mat = traits.Str(mandatory=True,
                         desc='path to a reference matrix',
                         argstr='%s',
                         position=3)

    out_txt = traits.Str('cost_function.txt',
                         mandatory=False,
                         desc='Output txt filename',
                         argstr='%s',
                         position=4)

    out_png = traits.Str('isocontour_image.png',
                         mandatory=False,
                         desc='Output  png filename',
                         argstr='%s',
                         position=5)


class coregQCOutputSpec(TraitedSpec):

    out_txt = File(exists=True)
    out_png = File(exists=True)


class coregQC(CommandLine):
    """
    Coregistration's QC using FLIRT.
    """
    import shivautils.interfaces as wf_interfaces
    p = op.dirname(wf_interfaces.__file__)
    _cmd = op.join(p, 'coreg_QC.sh')
    input_spec = CoregQCInputSpec
    output_spec = coregQCOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_txt'] = op.abspath(self.inputs.out_txt)
        outputs['out_png'] = op.abspath(self.inputs.out_png)
        return outputs


class MaskOverlayQCplot(CommandLine):
    """
    Creates multi-slice axial plot with Slicer showing mask overlaied on
    background image.
    """
    import shivautils.interfaces as wf_interfaces
    p = op.dirname(wf_interfaces.__file__)
    _cmd = op.join(p, 'mask_overlay_QC_images.sh')
    input_spec = MaskOverlayQCplotInputSpec
    output_spec = MaskOverlayQCplotOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
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
            outputs["out_files"].append(op.realpath("w%s" % fname))
        return outputs


class QC_metrics_Input(BaseInterfaceInputSpec):

    brain_mask = traits.File(
        exists=True,
        mandatory=True,
        desc="Brain mask nifti file"
    )

    main_norm_peak = traits.Float(
        mandatory=True,
        desc="Peak from the histogram of the 'main' image (t1 or swi)"
    )

    flair_norm_peak = traits.Float(
        None,
        usedefault=True,
        mandatory=False,
        desc="Peak from the histogram of the flair image"
    )

    swi_norm_peak = traits.Float(
        None,
        usedefault=True,
        mandatory=False,
        desc="Peak from the histogram of the swi image"
    )

    flair_reg_mat = traits.List(
        traits.File(exists=True),
        default=[],
        usedefault=True,
        mandatory=False,
        desc="List of one affine matrix file (.mat) of registration between flair and t1"
    )

    swi_reg_mat = traits.File(
        traits.File(exists=True),
        default=[],
        usedefault=True,
        mandatory=False,
        desc="List of one affine matrix file (.mat) of registration between swi and t1"
    )


class QC_metrics_Output(TraitedSpec):
    csv_qc_metrics = traits.File(exists=True,
                                 desc="csv containing the subject's qc metrics")


class QC_metrics(BaseInterface):
    """
    Compute and gather the quality control metrics for the corrent subject
    This can be used to automatically detect subject with bad preprocessing
    """
    input_spec = QC_metrics_Input
    output_spec = QC_metrics_Output

    def _run_interface(self, runtime):
        qc_dict = {}
        brain_vol = nib.load(self.input_spec.brain_mask).get_fdata().astype(bool)
        brain_size = brain_vol.sum()
        qc_dict['brain_mask_size'] = brain_size
        qc_dict['main_im_hist_peak'] = self.input_spec.main_norm_peak

        if self.input_spec.flair_reg_mat:
            rotation_flair_reg, translation_flair_reg = transf_from_affine(self.input_spec.flair_reg_mat[0])
            qc_dict['rotation_flair_reg'] = rotation_flair_reg
            qc_dict['translation_flair_reg'] = translation_flair_reg

        if self.input_spec.swi_reg_mat:
            rotation_swi_reg, translation_swi_reg = transf_from_affine(self.input_spec.swi_reg_mat[0])
            qc_dict['rotation_swi_reg'] = rotation_swi_reg
            qc_dict['translation_swi_reg'] = translation_swi_reg

        qc_df = pd.DataFrame(qc_dict)
        qc_file = 'qc_metrics.csv'
        qc_df.to_csv(qc_file)

        setattr(self, 'csv_qc_metrics', os.path.abspath(qc_file))
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['csv_qc_metrics'] = getattr(self, 'csv_qc_metrics')
        return outputs
