"""
Contains custom interfaces wrapping scripts/functions used by the nipype workflows.

@author: atsuchida
@modified by iastafeva (added niimath)
"""
import os
import os.path as op
import json

import numpy as np
import nibabel as nib
import pandas as pd
from weasyprint import HTML, CSS
import seaborn as sns
import matplotlib.pyplot as plt

from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec,
                                    CommandLine, CommandLineInputSpec,
                                    InputMultiPath, OutputMultiPath, isdefined)
from nipype.interfaces.spm.base import (SPMCommand, SPMCommandInputSpec)

from nipype.interfaces.matlab import MatlabCommand
from nipype.utils.filemanip import ensure_list, simplify_list

from string import Template
from shivautils.stats import transf_from_affine
from shivautils.postprocessing.report import make_report
from shivautils.stats import swarmplot_from_census
from shivautils.postprocessing import __file__ as postproc_init


class CustomIntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    brain_mask = File(exists=True, mandatory=True)
    out_file = File()
    svn_dir = traits.Str(mandatory=True, desc='Path to your SVN directory.')
    output_dir = traits.Str(mandatory=True, desc='Path to output directory.')


class CustomIntensityNormalizationOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class CustomIntensityNormalization(BaseInterface):
    """ De-noising algorithm for T1/T2flair images in MATLAB """

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

    """Create co-registrations qc endpoints:

     1) create coreg isocontour image
     2) compute the cost function of FLIRT coregistration

     coreg_QC.sh <COREGISTERED_IMG> <REF(T1)_BRAIN> <REF(T1)_BRAIN_MASK> <OUTPUT_BASENAME>

     creates <OUTPUT_BASENAME>.png with isocontours image and <OUTPUT_BASENAME>.txt
     with cost function value
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
    Creates multi-slice axial plot with Slicer showing mask overlaid on
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


class SummaryReportInputSpec(BaseInterfaceInputSpec):
    """Make summary report file in pdf format"""

    anonymized = traits.Bool(False, exists=True,
                             amandatory=False,
                             desc='Anonymized Subject ID')

    subject_id = traits.Str(desc="id for each subject")

    pvs_metrics_csv = traits.File(desc='csv file with pvs stats',
                                  mandatory=False)
    wmh_metrics_csv = traits.File(desc='csv file with wmh stats',
                                  mandatory=False)
    cmb_metrics_csv = traits.File(desc='csv file with cmb stats',
                                  mandatory=False)
    pvs_census_csv = traits.File(desc='csv file compiling each pvs size (and region)',
                                 mandatory=False)
    wmh_census_csv = traits.File(desc='csv file compiling each wmh size (and region)',
                                 mandatory=False)
    cmb_census_csv = traits.File(desc='csv file compiling each cmb size (and region)',
                                 mandatory=False)
    pred_list = traits.List(traits.Str,
                            desc='List of the different predictions computed ("PVS", "WMH" or "CMB")')
    brainmask = traits.File(exists=True,
                            desc='Nifti file of the brain mask in raw space')
    crop_brain_img = traits.File(desc='PNG file of the crop box, the first brain mask on the brain')

    isocontour_slides_FLAIR_T1 = traits.File(mandatory=False,
                                             exists=True,
                                             desc='PNG file of the FLAIR isocontour on T1 (QC of coregistration)')

    overlayed_brainmask_1 = traits.File(desc='PNG file of the final brain mask on first acquisition (T1 or SWI)')

    overlayed_brainmask_2 = traits.File(mandatory=False,
                                        exists=True,
                                        desc='PNG file of the final brain mask on second independent acquisition (SWI)')

    wf_graph = traits.File(mandatory=False,
                           exists=True,
                           desc='SVG file of the workflow graph')

    percentile = traits.Float(99.0,
                              desc='Percentile used during intensity normalisation')
    threshold = traits.Float(0.5,
                             desc='Threshold used to binarise brain masks')
    image_size = traits.Tuple(traits.Int, traits.Int, traits.Int,
                              default=(160, 214, 176),
                              usedefault=True,
                              desc='Dimensions of the cropped image')
    resolution = traits.Tuple(float, float, float,
                              desc='Resampled voxel size of the final image')
    thr_cluster_val = traits.Float(0.2,
                                   desc='Threshold used to binarise the predictions')
    min_seg_size = traits.Dict(key_trait=traits.Str, value_trait=traits.Int,
                               desc='Dictionary holding the minimal size set to filter segmented biomarkers')


class SummaryReportOutputSpec(TraitedSpec):
    """Output class

    Args:
        summary_report (html): summary report for each subject
        summary_report (pdf): summary report for each subject
    """
    summary_report = traits.Any(exists=True,
                                desc='summary html report')

    summary = traits.Any(exists=True,
                         desc='summary pdf report')


class SummaryReport(BaseInterface):
    """Make a summary report of preprocessing and prediction"""
    input_spec = SummaryReportInputSpec
    output_spec = SummaryReportOutputSpec

    def _run_interface(self, runtime):
        """
        Build the report for the whole workflow. It contains segmentation statistics and
        quality control figures.

        """
        if self.inputs.anonymized:  # TODO
            subject_id = None
        else:
            subject_id = self.inputs.subject_id

        brain_vol = nib.load(self.inputs.brainmask).get_fdata().astype(bool).sum()
        pred_metrics_dict = {}  # Will contain the stats dataframe for each biomarker
        pred_census_im_dict = {}  # Will contain the path to the swarm plot for each biomarker
        pred_list = self.inputs.pred_list
        if 'PVS' in pred_list:
            pred_metrics_dict['PVS'] = pd.read_csv(self.inputs.pvs_metrics_csv, index_col=0)
            pred_census_im_dict['PVS'] = swarmplot_from_census(self.inputs.pvs_census_csv, 'PVS')
        if 'WMH' in pred_list:
            pred_metrics_dict['WMH'] = pd.read_csv(self.inputs.wmh_metrics_csv, index_col=0)
            pred_census_im_dict['WMH'] = swarmplot_from_census(self.inputs.wmh_census_csv, 'WMH')
        if 'CMB' in pred_list:
            pred_metrics_dict['CMB'] = pd.read_csv(self.inputs.cmb_metrics_csv, index_col=0)
            pred_census_im_dict['CMB'] = swarmplot_from_census(self.inputs.cmb_census_csv, 'CMB')

        # set optional inputs to None if undefined
        if isdefined(self.inputs.isocontour_slides_FLAIR_T1):
            isocontour_slides_FLAIR_T1 = self.inputs.isocontour_slides_FLAIR_T1
        else:
            isocontour_slides_FLAIR_T1 = None
        if isdefined(self.inputs.overlayed_brainmask_2):
            overlayed_brainmask_2 = self.inputs.overlayed_brainmask_2
        else:
            overlayed_brainmask_2 = None
        if isdefined(self.inputs.wf_graph):
            wf_graph = self.inputs.wf_graph
        else:
            wf_graph = None
        # process
        summary_report = make_report(
            pred_metrics_dict=pred_metrics_dict,
            pred_census_im_dict=pred_census_im_dict,
            brain_vol=brain_vol,
            thr_cluster_val=self.inputs.thr_cluster_val,
            min_seg_size=self.inputs.min_seg_size,
            bounding_crop_path=self.inputs.crop_brain_img,
            overlayed_brainmask_1=self.inputs.overlayed_brainmask_1,
            overlayed_brainmask_2=overlayed_brainmask_2,
            isocontour_slides_FLAIR_T1=isocontour_slides_FLAIR_T1,
            subject_id=subject_id,
            image_size=self.inputs.image_size,
            resolution=self.inputs.resolution,
            percentile=self.inputs.percentile,
            threshold=self.inputs.threshold,
            wf_graph=wf_graph
        )

        with open('summary_report.html', 'w', encoding='utf-8') as fid:
            fid.write(summary_report)

        # Convert the HTML file to PDF
        postproc_dir = os.path.dirname(postproc_init)
        css = os.path.join(postproc_dir, 'report_styling.css')
        HTML('summary_report.html').write_pdf('summary.pdf',
                                              presentational_hints=True,
                                              stylesheets=[CSS(css)])

        setattr(self, 'summary_report', os.path.abspath('summary_report.html'))
        setattr(self, 'summary', os.path.abspath('summary.pdf'))

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['summary_report'] = getattr(self, 'summary_report')
        outputs['summary'] = getattr(self, 'summary')

        return outputs


class Join_Prediction_metrics_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    csv_files = traits.List(traits.File(exists=True),
                            desc='List if csv files containing metrics for individual participants',)

    subject_id = traits.List(desc="id for each subject")


class Join_Prediction_metrics_OutputSpec(TraitedSpec):
    """Output class

    Args:
        prediction_metrics_csv (csv): csv file with metrics about each prediction
    """
    prediction_metrics_csv = traits.File(exists=True,
                                         desc='csv file with metrics about each prediction')


class Join_Prediction_metrics(BaseInterface):
    """Get metrics about each prediction file"""
    input_spec = Join_Prediction_metrics_InputSpec
    output_spec = Join_Prediction_metrics_OutputSpec

    def _run_interface(self, runtime):
        """Run join of all cluster metrics in
        one csv file

        """
        path_csv_files = self.inputs.csv_files
        subject_id = self.inputs.subject_id

        csv_list = []
        for csv_file, sub_id in zip(path_csv_files, subject_id):
            sub_df = pd.read_csv(csv_file, index_col=0)
            sub_df.insert(0, 'sub_id', [sub_id]*sub_df.shape[0])
            csv_list.append(sub_df)
        all_sub_metrics = pd.concat(csv_list)
        all_sub_metrics.to_csv('prediction_metrics.csv')

        setattr(self, 'prediction_metrics_csv', os.path.abspath("prediction_metrics.csv"))
        return runtime

    def _list_outputs(self):
        """File in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['prediction_metrics_csv'] = getattr(self, 'prediction_metrics_csv')
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
        0.0,
        usedefault=True,
        mandatory=False,
        desc="Peak from the histogram of the flair image"
    )

    swi_norm_peak = traits.Float(
        0.0,
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

    swi_reg_mat = traits.List(
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
    Compute and gather the quality control metrics for the current subject
    This can be used to automatically detect subject with bad preprocessing
    """
    input_spec = QC_metrics_Input
    output_spec = QC_metrics_Output

    def _run_interface(self, runtime):
        qc_dict = {}
        brain_vol = nib.load(self.inputs.brain_mask).get_fdata().astype(bool)
        brain_size = brain_vol.sum()
        qc_dict['brain_mask_size'] = [brain_size]
        qc_dict['main_norm_peak'] = [self.inputs.main_norm_peak]

        if self.inputs.flair_reg_mat:
            rotation_flair_reg, translation_flair_reg = transf_from_affine(self.inputs.flair_reg_mat[0])
            qc_dict['rotation_flair_reg'] = [rotation_flair_reg]
            qc_dict['translation_flair_reg'] = [translation_flair_reg]

        if self.inputs.flair_norm_peak:
            qc_dict['flair_norm_peak'] = [self.inputs.flair_norm_peak]

        if self.inputs.swi_reg_mat:
            rotation_swi_reg, translation_swi_reg = transf_from_affine(self.inputs.swi_reg_mat[0])
            qc_dict['rotation_swi_reg'] = [rotation_swi_reg]
            qc_dict['translation_swi_reg'] = [translation_swi_reg]

        if self.inputs.swi_norm_peak:
            qc_dict['swi_norm_peak'] = [self.inputs.swi_norm_peak]

        qc_df = pd.DataFrame(qc_dict)
        qc_file = 'qc_metrics.csv'
        qc_df.to_csv(qc_file)

        setattr(self, 'csv_qc_metrics', os.path.abspath(qc_file))
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['csv_qc_metrics'] = getattr(self, 'csv_qc_metrics')
        return outputs


class Join_QC_metrics_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    csv_files = traits.List(traits.File(exists=True),
                            desc='List if csv files containing qc metrics for individual participants',)

    subject_id = traits.List(desc="id for each subject")

    population_csv_file = traits.File(mandatory=False,
                                      exists=True,
                                      desc='optional csv file from previous analysis to help sort-out outliers')


class Join_QC_metrics_OutputSpec(TraitedSpec):
    """Output class
    """
    qc_metrics_csv = traits.File(exists=True,
                                 desc='csv file with metrics about each qc')

    bad_qc_subs = traits.File(exists=True,
                              desc='json file containing the subjects with bad qc and their bad metrics')

    qc_plot_svg = traits.File(exists=True,
                              desc='svg file displaying the qc values for each metric and each subject')

    csv_pop_file = traits.File(exists=False,
                               desc='optional csv file with the new metrics concatenated to the population metrics')

    pop_bad_subjects_file = traits.File(exists=False,
                                        desc='optional json file with the subjects from the population csv input with outlier metrics')


class Join_QC_metrics(BaseInterface):
    """Get metrics about each subject's SQ, join them in a csv file and 
    check if there are some outliers
    """
    input_spec = Join_QC_metrics_InputSpec
    output_spec = Join_QC_metrics_OutputSpec

    def _run_interface(self, runtime):
        """Join of all qc metrics in
        one csv file

        """
        path_csv_files = self.inputs.csv_files
        subject_id = self.inputs.subject_id
        if isdefined(self.inputs.population_csv_file):
            population_csv_file = self.inputs.population_csv_file
        else:
            population_csv_file = None

        csv_list = []
        for csv_file, sub_id in zip(path_csv_files, subject_id):
            sub_df = pd.read_csv(csv_file, index_col=0)
            sub_df.insert(0, 'sub_id', [sub_id]*sub_df.shape[0])
            csv_list.append(sub_df)
        all_sub_metrics = pd.concat(csv_list)

        csv_out_file = 'qc_metrics.csv'
        all_sub_metrics.to_csv(csv_out_file)
        pop_subs = []
        if population_csv_file is not None:
            pop_metrics = pd.read_csv(population_csv_file, index_col=0)
            pop_subs = pop_metrics['sub_id'].tolist()
            all_sub_metrics = pd.concat([all_sub_metrics, pop_metrics], join='inner')
            csv_pop_file = 'qc_metrics_concat.csv'
            all_sub_metrics.to_csv(csv_pop_file)

        all_sub_metrics.set_index('sub_id', inplace=True)

        bad_subjects = {}  # Will contain the subjects with outlier qc metrics
        pop_bad_subjects = {}  # Will contain the subjects with outlier qc metrics from the population csv
        qc_plot_svg = 'qc_metrics_plot.svg'
        plot_per_row = 4
        n_rows = 1 + (len(all_sub_metrics.columns) - 1)//plot_per_row
        n_cols = min(len(all_sub_metrics.columns), plot_per_row)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        fig.tight_layout(pad=3)
        if len(all_sub_metrics) >= 10:  # We need at least a few subject to detect outliers
            # Plot box plot of each metrics with labeled outliers
            # Using the box plot way (1.5 times the inter-quartile distance)
            flierprops = {
                'markerfacecolor': (1, 0, 0),
                'markersize': 5,
                'markeredgewidth': 0,
                'linewidth': 0}
            q1_all = all_sub_metrics.quantile(0.25)
            q3_all = all_sub_metrics.quantile(0.75)
            min_thr_all = q1_all - 1.5*(q3_all - q1_all)
            max_thr_all = q3_all + 1.5*(q3_all - q1_all)
            for metric, ax in zip(all_sub_metrics.columns, np.nditer(axes, flags=['refs_ok'], op_flags=['readwrite'])):
                ax = ax.item()
                metric_vals = all_sub_metrics[metric]
                sns.boxplot(data=metric_vals, flierprops=flierprops, ax=ax)
                for id, val in metric_vals.items():
                    min_thr = min_thr_all[metric]
                    max_thr = max_thr_all[metric]
                    if val < min_thr or val > max_thr:
                        # plot the name of the subject next to the outlier point
                        ax.annotate(id, xy=(0, val), xytext=(5, -2), textcoords='offset points')
                        if id in pop_subs:  # Old bad subject
                            if id not in pop_bad_subjects.keys():
                                pop_bad_subjects[id] = [metric]
                            else:
                                pop_bad_subjects[id].append(metric)
                        else:  # New bad subjects
                            if id not in bad_subjects.keys():
                                bad_subjects[id] = [metric]
                            else:
                                bad_subjects[id].append(metric)

        else:
            # Simple swarm plot if few subjects for a quick check
            # Labels may overlap but outliers should stand out
            for metric, ax in zip(all_sub_metrics.columns, np.nditer(axes, flags=['refs_ok'], op_flags=['readwrite'])):
                ax = ax.item()
                metric_vals = all_sub_metrics[metric]
                sns.swarmplot(data=metric_vals, ax=ax)
                for id, val in metric_vals.items():
                    ax.annotate(id, xy=(0, val), xytext=(5, -2), textcoords='offset points')
        plt.savefig(qc_plot_svg, format='svg')
        plt.close(fig)

        # Checking if the histogram peaks of normalized images are between 0 and 1
        for metric in all_sub_metrics:
            if '_norm_peak' in metric:
                metric_series = all_sub_metrics[metric]
                bad_norm = metric_series.index[(metric_series > 1) | (metric_series < 0)]
                for id in bad_norm:
                    if id not in bad_subjects.keys():
                        bad_subjects[id] = [metric]
                    else:
                        bad_subjects[id].append(metric)  # May create duplicates, but it doesn't really matter

        # Save the bad subject and bad metrics

        bad_subjects_file = 'failed_qc.json'
        with open(bad_subjects_file, 'w') as fp:
            json.dump(bad_subjects, fp, sort_keys=True, indent=4)
        if pop_bad_subjects:
            pop_bad_subjects_file = 'previous_run_failed_qc.json'
            with open(pop_bad_subjects_file, 'w') as fp:
                json.dump(pop_bad_subjects, fp, sort_keys=True, indent=4)

        setattr(self, 'qc_metrics_csv', os.path.abspath(csv_out_file))
        setattr(self, 'bad_qc_subs', os.path.abspath(bad_subjects_file))
        setattr(self, 'qc_plot_svg', os.path.abspath(qc_plot_svg))
        if population_csv_file is not None:
            setattr(self, 'csv_pop_file', os.path.abspath(csv_pop_file))
            setattr(self, 'pop_bad_subjects_file', os.path.abspath(pop_bad_subjects_file))

        return runtime

    def _list_outputs(self):
        """File in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['qc_metrics_csv'] = getattr(self, 'qc_metrics_csv')
        outputs['bad_qc_subs'] = getattr(self, 'bad_qc_subs')
        outputs['qc_plot_svg'] = getattr(self, 'qc_plot_svg')
        if hasattr(self, 'csv_pop_file'):
            outputs['csv_pop_file'] = getattr(self, 'csv_pop_file')
            outputs['pop_bad_subjects_file'] = getattr(self, 'pop_bad_subjects_file')

        return outputs
