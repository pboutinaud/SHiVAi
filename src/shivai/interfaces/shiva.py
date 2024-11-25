"""Interfaces for SHIVA project deep learning segmentation and prediction tools."""
import os
import glob

from shivai.utils.misc import md5

from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec,
                                    CommandLineInputSpec,
                                    CommandLine, BaseInterface,
                                    isdefined)

from shivai.interfaces.singularity import (SingularityCommandLine,
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

from nipype.interfaces.dcm2nii import (Dcm2niix,
                                       Dcm2niixInputSpec,
                                       Dcm2niixOutputSpec)


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

    use_cpu = traits.Int(0,
                         usedefault=True,
                         argstr='--use_cpu %d',
                         desc='Set to a positive integer to ignore GPUs and use CPUs instead. Limit the CPU usage by the given number')


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
    _cmd = 'shiva_predict'  # shivai.scripts.predict:main

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.basename(str(self.inputs.out_filename)))
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
        outputs["segmentation"] = os.path.abspath(os.path.basename(str(self.inputs.out_filename)))
        return outputs


class Predict_Multi_InputSpec(BaseInterfaceInputSpec):
    """ Input parameter for the Predict_Multi interface """
    primary_image_file = traits.Dict(key_trait=traits.String,
                                     value_trait=traits.File,
                                     argstr='--subjects %s --img1_files %s',
                                     desc=('Dict containing {sub_id: file_path} for all subjects, for the '
                                           'main aquisition image file.'),
                                     mandatory=True)

    second_image_file = traits.Dict(key_trait=traits.String,
                                    value_trait=traits.File,
                                    argstr='--img2_files %s',
                                    desc=('Dict containing {sub_id: file_path} for all subjects, for the '
                                          'secondary aquisition image file in case of multi-modal prediction.'),
                                    mandatory=False)

    brainmask_files = traits.Dict(key_trait=traits.String,
                                  value_trait=traits.File,
                                  argstr='--mask_files %s',
                                  desc=('Dict containing {sub_id: file_path} for all subjects, for the '
                                        'brain mask used to filter-out out-of-brain segmentations.'),
                                  mandatory=False)

    model_dir = traits.Directory(exists=True,
                                 desc='Folder containing the AI models',
                                 argstr='--model_dir %s',
                                 mandatory=True)

    descriptor = traits.File(exists=True,
                             desc='File information about models and models location within model_dir',
                             argstr='--descriptor %s',
                             mandatory=True)

    # acq_types = traits.List(traits.String,
    #                         argstr='--acq_types %s',
    #                         desc=('List if the type of aquisition (lower case) for primary_image_file'
    #                               'and second_image_file, in order.'))

    batch_size = traits.Int(20,
                            desc='Number of images to load at the same time in memmory with nib.load',
                            argstr='--batch_size %d',
                            usedefault=True)

    input_size = traits.Tuple(traits.Int, traits.Int, traits.Int,  # default=(160, 214, 176)
                              argstr='--input_size %dx%dx%d',
                              desc='Expected image size input for the models')

    foutname = traits.Str('{sub}_segmentation.nii.gz',
                          desc='Output name to be formatted with the subject name "sub"',
                          argstr='--foutname %s',
                          usedefault=True)

    use_cpu = traits.Int(0,
                         usedefault=True,
                         argstr='--use_cpu',
                         desc='Set to a positive integer to ignore GPUs and use CPUs instead. Limit the CPU usage by the given number')


class Predict_Multi_SingularityInputSpec(SingularityInputSpec, Predict_Multi_InputSpec):
    """PredictVRS input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    out_dir = traits.Directory(exists=False,
                               desc='Folder where the results will be saved',  # Only for Singularity
                               argstr='--out_dir %s',
                               mandatory=True)


class Predict_Multi_OutputSpec(TraitedSpec):
    """ Output of the Predict_Multi interface """
    segmentations = traits.Dict(key_trait=traits.String,
                                value_trait=traits.File,
                                desc='The segmentation images')


class Predict_Multi(CommandLine):
    input_spec = Predict_Multi_InputSpec
    output_spec = Predict_Multi_OutputSpec
    _cmd = 'shiva_predict_multi'

    def _format_arg(self, name, spec, value):
        if spec.is_trait_type(traits.Dict):
            argstr = spec.argstr
            sub_list = list(self.inputs.primary_image_file.keys())
            file_list = [value[sub] for sub in sub_list]  # Making sure all file lists have the same order
            if argstr.count('%s') == 2:
                return spec.argstr % (' '.join(sub_list), ' '.join(file_list))
            else:
                return spec.argstr % (' '.join(file_list))
        return super(Predict_Multi, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        sub_list = list(self.inputs.primary_image_file.keys())
        outnames = [self.inputs.foutname.format(sub=sub) for sub in sub_list]
        outputs['segmentations'] = {sub: os.path.abspath(file) for sub, file in zip(sub_list, outnames)}

        return outputs


class Predict_Multi_Singularity(SingularityCommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net with tensorflow-gpu in a container (apptainer/singularity).
    """
    input_spec = Predict_Multi_SingularityInputSpec
    output_spec = Predict_Multi_OutputSpec
    _cmd = 'shiva_predict_multi'

    def _format_arg(self, name, spec, value):
        if spec.is_trait_type(traits.Dict):
            argstr = spec.argstr
            sub_list = list(self.inputs.primary_image_file.keys())
            file_list = [value[sub] for sub in sub_list]  # Making sure all file lists have the same order
            if argstr.count('%s') == 2:
                return spec.argstr % (' '.join(sub_list), ' '.join(file_list))
            else:
                return spec.argstr % (' '.join(file_list))
        return super(Predict_Multi_Singularity, self)._singularity_format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        sub_list = list(self.inputs.primary_image_file.keys())
        outnames = [self.inputs.foutname.format(sub=sub) for sub in sub_list]
        outputs['segmentations'] = {sub: os.path.abspath(file) for sub, file in zip(sub_list, outnames)}
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
        outputs["segmentation"] = os.path.abspath(os.path.basename(str(self.inputs.out_filename)))
        outputs["qc"] = os.path.abspath(os.path.basename(str(self.inputs.qc)))
        outputs["volumes"] = os.path.abspath(os.path.basename(str(self.inputs.vol)))
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
        outputs["segmentation"] = os.path.abspath(os.path.basename(str(self.inputs.out_filename)))
        outputs["qc"] = os.path.abspath(os.path.basename(str(self.inputs.qc)))
        outputs["volumes"] = os.path.abspath(os.path.basename(str(self.inputs.vol)))
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


class Dcm2niix_Singularity_InputSpec(SingularityInputSpec, Dcm2niixInputSpec):
    """Quickshear input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """
    pass


class Dcm2niix_Singularity(Dcm2niix, SingularityCommandLine):
    def __init__(self):
        """Call parent constructor."""
        super(Dcm2niix_Singularity, self).__init__()

    input_spec = Dcm2niix_Singularity_InputSpec
    output_spec = Dcm2niixOutputSpec
    _cmd = Dcm2niix._cmd


class Shivai_InputSpec(CommandLineInputSpec):
    """Input arguments structure for the shiva command"""

    in_dir = traits.Directory(argstr='--in %s',
                              desc='Directory containing the input data for all participants. Important for the Singularity bindings',
                              exists=True,
                              mandatory=True)

    out_dir = traits.Directory(argstr='--out %s',
                               desc='Directory where the results will be saved (in and "results" sub-directory).',
                               exists=False,
                               mandatory=True)

    input_type = traits.Enum('swomed', 'standard', 'BIDS',
                             argstr='--input_type %s',
                             desc='Input data structure. Chose "swomed" when directly inputing the paths to the acquisitions',
                             usedefault=True,
                             mandatory=False)

    file_type = traits.Enum('nifti', 'dicom',
                            argstr='--file_type %s',
                            desc='Type of input file (nifti or dicom) for the input images',
                            usedefault=True,
                            mandatory=False)

    t1_image_nii = traits.File(argstr='--swomed_t1 %s',
                               desc='Path to the T1 (or T1-like) image (in nifti format). Required for PVS, WMH, and Lacunas',
                               exists=True,
                               mandatory=False,
                               xor=['t1_image_dcm'])

    t1_image_dcm = traits.Directory(argstr='--swomed_t1 %s',
                                    desc='Path to the T1 (or T1-like) dicom folder. Required for PVS, WMH, and Lacunas',
                                    exists=True,
                                    mandatory=False,
                                    xor=['t1_image_nii'])

    flair_image_nii = traits.File(argstr='--swomed_flair %s',
                                  desc='Path to the FLAIR (or FLAIR-like) image. Required for PVS2, WMH, and Lacunas',
                                  exists=True,
                                  mandatory=False,
                                  xor=['flair_image_dcm'])

    flair_image_dcm = traits.Directory(argstr='--swomed_flair %s',
                                       desc='Path to the FLAIR (or FLAIR-like) image. Required for PVS2, WMH, and Lacunas',
                                       exists=True,
                                       mandatory=False,
                                       xor=['flair_image_nii'])

    swi_image_nii = traits.File(argstr='--swomed_swi %s',
                                desc='Path to the SWI (or SWI-like) image. Required for CMB',
                                exists=True,
                                mandatory=False,
                                xor=['swi_image_dcm'])

    swi_image_dcm = traits.Directory(argstr='--swomed_swi %s',
                                     desc='Path to the SWI (or SWI-like) image. Required for CMB',
                                     exists=True,
                                     mandatory=False,
                                     xor=['swi_image_nii'])

    replace_t1 = traits.Str(argstr='--replace_t1 %s',
                            desc='Data type that will replace the "t1" image.',
                            mandatory=False)

    replace_flair = traits.Str(argstr='--replace_flair %s',
                               desc='Data type that will replace the "flair" image (e.g. "t2s").',
                               mandatory=False)

    replace_swi = traits.Str(argstr='--replace_swi %s',
                             desc='Data type that will replace the "swi" image (e.g. "t2gre").',
                             mandatory=False)

    swi_file_num = traits.Int(argstr='--swi_file_num %s',
                              desc='Data type that will replace the "swi" image (e.g. "t2gre").',
                              mandatory=False)

    db_name = traits.Str(argstr='--db_name %s',
                         desc='Name of the dataset for the report (e.g. "UKBB").',
                         mandatory=False)

    sub_name = traits.Str(argstr='--sub_names %s',
                          desc='Participant ID to be processed from the input directory.',
                          mandatory=True)

    prediction = traits.Enum("PVS", "PVS2", "WMH", "CMB", "LAC", "all",
                             "all PVS",
                             "PVS WMH", "PVS CMB", "PVS LAC",
                             "PVS WMH CMB", "PVS WMH LAC",
                             "WMH CMB", "WMH LAC",
                             "CMB LAC",
                             "PVS2 WMH", "PVS2 CMB", "PVS2 LAC",
                             "PVS2 WMH CMB", "PVS2 WMH LAC",
                             argstr="--prediction %s",
                             desc='Prediction to run ("PVS", "PVS2", "WMH", "CMB", "LAC", "all")',
                             usedefault=True,
                             mandatory=False)

    use_t1 = traits.Bool(argstr='--use_t1',
                         desc='Used to perform segmentation on T1 image when running CMB prediction alone (otherwise will use SWI for the seg).',
                         mandatory=False)

    brain_seg = traits.Enum("shiva", "shiva_gpu", "synthseg", "premasked", "custom",
                            argstr='--brain_seg %s',
                            desc='Type of segmentation to run. Chose among: "shiva", "shiva_gpu", "synthseg" (requires precomputed parcellation plugged in synthseg_parc), "premasked", "custom"',
                            usedefault=True,
                            mandatory=False)

    synthseg_parc = traits.File(argstr='--swomed_parc %s',
                                desc='Path to the synthseg parcellation of the current subject. Requires brain_seg = "synthseg"',
                                exists=True,
                                mandatory=False)
    synthseg_vol = traits.File(argstr='--swomed_ssvol %s',
                               desc='Path to the synthseg volume.csv file.',
                               exists=True,
                               mandatory=False)
    synthseg_qc = traits.File(argstr='--swomed_ssqc %s',
                              desc='Path to the synthseg qc.csv file.',
                              exists=True,
                              mandatory=False)

    # ai_threads = traits.Int(argstr='--ai_threads %d',
    #                               desc='Number of thread to run with "synthseg_cpu".',
    #                               mandatory=False)

    custom_LUT = traits.File(argstr='--custom_LUT %s',
                             desc='Look-up table file to pair with the custom segmentation when used ("custom" brain_seg)',
                             exists=True,
                             mandatory=False)

    anonymize = traits.Bool(argstr='--anonymize',
                            desc='Anonymize the report',
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

    brainmask_descriptor = traits.File(argstr='--brainmask_descriptor %s',
                                       desc='Descriptor json file (with md5) for brain mask models to check if the loaded models are correct (and keep a tract in swomed)',
                                       mandatory=False)

    pvs_descriptor = traits.File(argstr='--pvs_descriptor %s',
                                 desc='Descriptor json file (with md5) for monomodal PVS models to check if the loaded models are correct (and keep a tract in swomed)',
                                 mandatory=False)

    pvs2_descriptor = traits.File(argstr='--pvs2_descriptor %s',
                                  desc='Descriptor json file (with md5) for bimodal PVS models to check if the loaded models are correct (and keep a tract in swomed)',
                                  mandatory=False)

    wmh_descriptor = traits.File(argstr='--wmh_descriptor %s',
                                 desc='Descriptor json file (with md5) for WMH models to check if the loaded models are correct (and keep a tract in swomed)',
                                 mandatory=False)

    cmb_descriptor = traits.File(argstr='--cmb_descriptor %s',
                                 desc='Descriptor json file (with md5) for CMB models to check if the loaded models are correct (and keep a tract in swomed)',
                                 mandatory=False)

    lac_descriptor = traits.File(argstr='--lac_descriptor %s',
                                 desc='Descriptor json file (with md5) for lacuna models to check if the loaded models are correct (and keep a tract in swomed)',
                                 mandatory=False)


class Shivai_OutputSpec(TraitedSpec):
    """Shivai ports."""
    # result_dir = traits.File(desc='Folder where the results are stored',
    #                          exists=True)
    pvs_census = traits.File(desc='PVS census csv file (size and location of each PVS)', exists=False)
    wmh_census = traits.File(desc='WMH census csv file (size and location of each WMH)', exists=False)
    cmb_census = traits.File(desc='CMB census csv file (size and location of each CMB)', exists=False)
    lac_census = traits.File(desc='Lacuna census csv file (size and location of each lacuna)', exists=False)
    pvs_stats = traits.File(desc='PVS stats csv file (size metrics and counts for PVS in each region)', exists=False)
    wmh_stats = traits.File(desc='WMH stats csv file (size metrics and counts for WMH in each region)', exists=False)
    cmb_stats = traits.File(desc='CMB stats csv file (size metrics and counts for CMB in each region)', exists=False)
    lac_stats = traits.File(desc='Lacuna stats csv file (size metrics and counts for lacunas in each region)', exists=False)
    pvs_labelled_map = traits.File(desc='PVS map file with labelled clusters', exists=False)
    wmh_labelled_map = traits.File(desc='WMH map file with labelled clusters', exists=False)
    cmb_labelled_map = traits.File(desc='CMB map file with labelled clusters', exists=False)
    lac_labelled_map = traits.File(desc='Lacuna map file with labelled clusters', exists=False)
    pvs_raw_map = traits.File(desc='PVS map (raw)', exists=False)
    wmh_raw_map = traits.File(desc='WMH map (raw)', exists=False)
    cmb_raw_map = traits.File(desc='CMB map (raw)', exists=False)
    lac_raw_map = traits.File(desc='Lacuna map (raw)', exists=False)
    summary_report = traits.File(desc='PDF report file for the analysis', exists=True)
    converted_t1 = traits.File(desc='Nifti T1 file converted from DICOM', exists=False)
    converted_flair = traits.File(desc='Nifti FLAIR file converted from DICOM', exists=False)
    converted_swi = traits.File(desc='Nifti SWI file converted from DICOM', exists=False)
    sidecar_t1 = traits.File(desc='JSON file sidecar from T1 DICOM conversion', exists=False)
    sidecar_flair = traits.File(desc='JSON file sidecar from FLAIR DICOM conversion', exists=False)
    sidecar_swi = traits.File(desc='JSON file sidecar from SWI DICOM conversion', exists=False)
    t1_preproc = traits.File(desc='Preprocessed T1 file (conformed, cropped, intensity normalized, defaced)', exists=False)
    flair_preproc = traits.File(desc='Preprocessed FLAIR file (conformed, cropped, intensity normalized, defaced)', exists=False)
    swi_preproc = traits.File(desc='Preprocessed SWI file (conformed, cropped, intensity normalized, defaced)', exists=False)
    brain_mask = traits.File(desc='Brain mask (conformed, cropped)', exists=False)
    # brain_parc = traits.File(desc='Brain parcellation (conformed, cropped)', exists=False)
    swi2t1_transform = traits.File(desc='ANTs affine matrix for SWI to T1 registration', exists=False)  # for CMB when whith_t1 and brain parc
    brain_mask_swi = traits.File(desc='Brain mask in SWI space (conformed, cropped)', exists=False)  # for CMB when whith_t1 without brain parc
    qc_metrics = traits.File(desc='Metrics used for group-level QC in a CSV file', exists=False)
    ss_cleaned = traits.File(desc='Synthseg parcellation after cleaning mislabeled "islands"', exists=False)
    ss_volumes = traits.File(desc='CSV file with the volumes of each region from the Synthseg parcellation', exists=False)
    ss_qc = traits.File(desc='CSV file with the qc from the Synthseg parcellation', exists=False)
    ss_derived_parc = traits.File(desc='Brain parcellation derived from Synthseg to be used for custom biomarker parcelllation', exists=False)
    parc4pvs = traits.File(desc='Parcellation for PVS counting (derived from Synthseg)', exists=False)
    parc4wmh = traits.File(desc='Parcellation for WMH counting (derived from Synthseg)', exists=False)
    parc4cmb = traits.File(desc='Parcellation for CMB counting (derived from Synthseg)', exists=False)
    parc4lac = traits.File(desc='Parcellation for lacuna counting (derived from Synthseg)', exists=False)
    # parc4pvs_dict = traits.File(desc='Dictionnary (in JSON file) with label/region name in parc4pvs', exists=False)
    # parc4wmh_dict = traits.File(desc='Dictionnary (in JSON file) with label/region name in parc4wmh', exists=False)
    # parc4cmb_dict = traits.File(desc='Dictionnary (in JSON file) with label/region name in parc4cmb', exists=False)
    # parc4lac_dict = traits.File(desc='Dictionnary (in JSON file) with label/region name in parc4lac', exists=False)


class Shivai(CommandLine):
    """Runs the shiva workflow (and thus must be called through the Shivai_Singularity inteface, not this one)."""

    input_spec = Shivai_InputSpec
    output_spec = Shivai_OutputSpec
    _cmd = 'shiva'

    def _list_outputs(self):
        subject_id = self.inputs.sub_name
        res_dir = os.path.join(os.path.abspath(str(self.inputs.out_dir)), 'results')
        output_dict = {
            'pvs_census': f'segmentations/pvs_segmentation/{subject_id}/pvs_census.csv',
            'wmh_census': f'segmentations/wmh_segmentation/{subject_id}/wmh_census.csv',
            'cmb_census': f'segmentations/cmb_segmentation*/{subject_id}/cmb*_census.csv',
            'lac_census': f'segmentations/lac_segmentation/{subject_id}/lac_census.csv',
            'pvs_stats': f'segmentations/pvs_segmentation/{subject_id}/pvs_stats_wide.csv',
            'wmh_stats': f'segmentations/wmh_segmentation/{subject_id}/wmh_stats_wide.csv',
            'cmb_stats': f'segmentations/cmb_segmentation*/{subject_id}/cmb*_stats_wide.csv',
            'lac_stats': f'segmentations/lac_segmentation/{subject_id}/lac_stats_wide.csv',
            'pvs_labelled_map': f'segmentations/pvs_segmentation/{subject_id}/labelled_pvs.nii.gz',
            'wmh_labelled_map': f'segmentations/wmh_segmentation/{subject_id}/labelled_wmh.nii.gz',
            'cmb_labelled_map': f'segmentations/cmb_segmentation*/{subject_id}/labelled_cmb.nii.gz',
            'lac_labelled_map': f'segmentations/lac_segmentation/{subject_id}/labelled_lac.nii.gz',
            'pvs_raw_map': f'segmentations/pvs_segmentation/{subject_id}/pvs_map.nii.gz',
            'wmh_raw_map': f'segmentations/wmh_segmentation/{subject_id}/wmh_map.nii.gz',
            'cmb_raw_map': f'segmentations/cmb_segmentation*/{subject_id}/cmb_map.nii.gz',
            'lac_raw_map': f'segmentations/lac_segmentation/{subject_id}/lac_map.nii.gz',
            'summary_report': f'report/{subject_id}/Shiva_report.pdf',
            'converted_t1': f'shiva_preproc/t1_preproc/{subject_id}/converted_*.nii.gz',
            'converted_flair': f'shiva_preproc/flair_preproc/{subject_id}/converted_*.nii.gz',
            'converted_swi': f'shiva_preproc/swi_preproc/{subject_id}/converted_*.nii.gz',
            'sidecar_t1': f'shiva_preproc/t1_preproc/{subject_id}/converted_*.json',
            'sidecar_flair': f'shiva_preproc/flair_preproc/{subject_id}/converted_*.json',
            'sidecar_swi': f'shiva_preproc/swi_preproc/{subject_id}/converted_*.json',
            't1_preproc': f'shiva_preproc/t1_preproc/{subject_id}/*_defaced_cropped_intensity_normed.nii.gz',
            'flair_preproc': f'shiva_preproc/flair_preproc/{subject_id}/*_defaced_cropped_intensity_normed.nii.gz',
            'swi_preproc': f'shiva_preproc/swi_preproc/{subject_id}/*_defaced_cropped_intensity_normed.nii.gz',
            'brain_mask': f'shiva_preproc/*_preproc/{subject_id}/brainmask_cropped.nii.gz',
            'swi2t1_transform': f'shiva_preproc/swi_preproc/{subject_id}/swi_to_t1_0GenericAffine.mat',
            'brain_mask_swi': f'shiva_preproc/swi_preproc/{subject_id}/brainmask_cropped*.nii.gz',
            'qc_metrics': f'shiva_preproc/qc_metrics/{subject_id}/qc_metrics.csv',
            'ss_cleaned': f'shiva_preproc/synthseg/{subject_id}/cleaned_synthseg_parc.nii.gz',
            'ss_derived_parc': f'shiva_preproc/synthseg/{subject_id}/derived_parc.nii.gz',
            'ss_volumes': f'shiva_preproc/synthseg/{subject_id}/volumes.csv',
            'ss_qc': f'shiva_preproc/synthseg/{subject_id}/qc.csv',
            'parc4pvs': f'segmentations/pvs_segmentation/{subject_id}/Brain_Seg_for_PVS.nii.gz',
            'parc4wmh': f'segmentations/wmh_segmentation/{subject_id}/Brain_Seg_for_WMH.nii.gz',
            'parc4cmb': f'segmentations/cmb_segmentation*/{subject_id}/Brain_Seg_for_CMB*.nii.gz',
            'parc4lac': f'segmentations/lac_segmentation/{subject_id}/Brain_Seg_for_LAC.nii.gz',
        }
        outputs = self.output_spec().get()
        for output_name, ouput_path in output_dict.items():
            out_path = os.path.join(res_dir, ouput_path)
            if '*' in out_path:
                foundpath = glob.glob(out_path)  # There should only be one file
                if foundpath:
                    out_path = foundpath[0]
                else:
                    continue
            if not os.path.exists(out_path):
                continue
            outputs[output_name] = out_path
        return outputs


class Shivai_Singularity_InputSpec(SingularityInputSpec, Shivai_InputSpec):
    """Shivai input specification (singularity mixin).

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


class Direct_File_Provider_InputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Str(mandatory=True, desc='Dummy argument')
    img1 = traits.Any(exists=True, mandatory=True)  # Using any as can be File (for nifti) or Directory (for dcm)
    img2 = traits.Any(exists=True, mandatory=False)
    img3 = traits.Any(exists=True, mandatory=False)
    seg = traits.File(exists=True, mandatory=False)
    synthseg_vol = traits.File(exists=True, mandatory=False)
    synthseg_qc = traits.File(exists=True, mandatory=False)


class Direct_File_Provider_OutputSpec(TraitedSpec):
    img1 = traits.Any(exists=True, mandatory=True)
    img2 = traits.Any(exists=True, mandatory=False)
    img3 = traits.Any(exists=True, mandatory=False)
    seg = traits.File(exists=True, mandatory=False)
    synthseg_vol = traits.File(exists=True, mandatory=False)
    synthseg_qc = traits.File(exists=True, mandatory=False)


class Direct_File_Provider(BaseInterface):
    """Pass the input path to the ouput in order to replace a datagrabber when
    you want to directly pass the full path to the images (typically with SWOMed)"""

    input_spec = Direct_File_Provider_InputSpec
    output_spec = Direct_File_Provider_OutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for interface_trait in outputs:
            if isdefined(getattr(self.inputs, interface_trait)):
                outputs[interface_trait] = getattr(self.inputs, interface_trait)
        return outputs
