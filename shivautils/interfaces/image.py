"""Custom nipype interfaces for image resampling/cropping and
other preliminary tasks"""
import os
import nibabel.processing as nip
import nibabel as nb
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, TraitedSpec
from nipype.utils.filemanip import split_filename
from shivautils.image import normalization, crop, threshold


class ConformInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply conform function on the
    image """
    img = traits.File(exists=True, desc='NIfTI formated input image to conform to agiven shape',
                      mandatory=True)

    dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                              default=(256, 256, 256),
                              usedefault=True,
                              desc='The minimal array dimensions for the'
                              'intermediate conformed image.')

    voxel_size = traits.Tuple(float, float, float,
                              desc='<please fill me>')

    orientation = traits.Enum('RAS', 'LAS', 'RPS', 'LPS', 'RAI', 'LPI', 'RPI',
                              'LAP', 'RAP',
                              desc="orientation of image volume brain",
                              use_default=True)
    order = traits.Int(3,
                       desc="orientation of image volume brain",
                       usedefault=True,
                       )


class ConformOutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nb.Nifti1Image): transformed image
    """
    resampled = traits.File(exists=True,
                            desc='Image conformed to the required voxel size and shape.')


class Conform(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            NIfTI image file to process
            dimensions (int, int, int): minimal dimension
            voxel_size (float, float, float): Voxel size of final image
            orientation (string): orientation of the volume brain
        output_spec (nb.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            conform image to desired voxel sizes and dimensions
    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
                               function
        Return: runtime
        """
        fname = self.inputs.img
        img = nb.load(fname)
        resampled = nip.conform(img, out_shape=self.inputs.dimensions,
                                voxel_size=self.inputs.voxel_size, 
                                order=self.inputs.order,
                                cval=0.0,
                                orientation=self.inputs.orientation,
                                out_class=None)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(resampled, base + 'resampled.nii')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["resampled"] = os.path.abspath(base +
                                               'resampled.nii')
        return outputs


class IntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply normalization to the nifti
    image"""
    input_image = traits.File(exists=True, desc='NIfTI image input.',
                              mandatory=True)

    percentile = traits.Float(exists=True, desc='value to threshold above this'
                              'percentile',
                              mandatory=True)
                              
    brain_mask = traits.File(desc='brain_mask to adapt normalization to'
    			             'the greatest number', mandatory=False)


class IntensityNormalizationOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): file img IRM nifti transformed
    """
    intensity_normalized = traits.File(exists=True,
                                       desc='Intensity normalized image')

    report = traits.File(exists=True, 
                         desc='html file with histogram voxel value,'
                         'mode and percentile value')

    mode = traits.Float(exists=True,
                        desc='Most frequent value of intensities voxel'
                        'histogram in an interval given')


class Normalization(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image): NIfTI image input
        output_spec (nb.Nifti1Image): Intensity-normalized image

    Methods:
        _run_interface(runtime):
            transformed an image into another with specified arguments
    """
    input_spec = IntensityNormalizationInputSpec
    output_spec = IntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # conform image to desired voxel sizes and dimensions
        fname = self.inputs.input_image
        img = nb.load(fname)
        if self.inputs.brain_mask:
            brain_mask = nb.load(self.inputs.brain_mask)
        else:
            brain_mask = self.inputs.brain_mask
        img_normalized, report, mode = normalization(img,
                                                     self.inputs.percentile,
                                                     brain_mask)
        # Save it for later use in _list_outputs
        setattr(self, 'mode_attr', mode)
        with open('report.html', 'w', encoding='utf-8') as fid:
            fid.write(report)
        _, base, _ = split_filename(fname)
        nb.save(img_normalized, base + '_img_normalized.nii')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        outputs['report'] = os.path.abspath('report.html')
        fname = self.inputs.input_image
        _, base, _ = split_filename(fname)
        outputs['mode'] = getattr(self, 'mode_attr')
        outputs["intensity_normalized"] = os.path.abspath(base +
                                                          '_img_normalized.nii'
                                                          )
        return outputs


class ThresholdInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    img = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    threshold = traits.Float(0.5, exists=True, type=float, mandatory=True,
                             desc='Value of the treshold to apply to the image'
                            )
    
    sign = traits.Enum('+', '-',
                       usedefault=True,
                       desc='Whether to keep data above threshold or below threshold.')

    binarize = traits.Float(False,exists=True, type=bool,
                            desc='Binarized intensities voxel of brain_mask')


class ThresholdOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): file img IRM nifti transformed
    """
    thresholded = traits.File(exists=True,
                              desc='Thresholded image')


class Threshold(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            file img Nifti
            Threshold for the brain mask
        output_spec (nb.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            transformed an image into brain mask
    """
    input_spec = ThresholdInputSpec
    output_spec = ThresholdOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        fname = self.inputs.img
        img = nb.load(fname)
        thresholded = threshold(img,
                                self.inputs.threshold,
                                sign=self.inputs.sign,
                                binarize=self.inputs.binarize)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(thresholded, base + '_thresholded.nii.gz')

        return runtime

    def _list_outputs(self):
        """
        Just gets the absolute path to the scheme file name
        """
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["thresholded"] = os.path.abspath(base +
                                                 '_thresholded.nii.gz')
        return outputs


class CropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply cropping to the
    nifti image"""
    roi_mask = traits.File(exists=True,
                           desc='Mask for computation of center of gravity and'
                           'cropping coordinates', mandatory=True)

    apply_to = traits.File(exists=True, 
                           desc='Image to crop', mandatory=True)

    final_dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                    default=(160, 214, 176),
                                    usedefault=True,
                                    desc='Final image array size in i, j, k.')
                                   
    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
    			           desc='center of gravity of nifti image cropped with first' 
    			           'voxel intensities normalization', mandatory=False)


class CropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): Cropped image
    """
    cropped = traits.File(exists=True,
                          desc='nb.Nifti1Image: preprocessed image')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc='center of gravity brain_mask')

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')


class Crop(BaseInterface):
    """Transform an image to desired dimensions."""
    input_spec = CropInputSpec
    output_spec = CropOutputSpec

    def _run_interface(self, runtime):
        """Run crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        mask = nb.load(self.inputs.roi_mask)
        target = nb.load(self.inputs.apply_to)

        # process
        cropped, cdg_ijk, bbox1, bbox2 = crop(
            mask,
            target,
            self.inputs.final_dimensions,
            self.inputs.cdg_ijk
            )
        cdg_ijk = tuple([cdg_ijk[0], cdg_ijk[1], cdg_ijk[2]])
        bbox1 = tuple([bbox1[0], bbox1[1], bbox1[2]])
        bbox2 = tuple([bbox2[0], bbox2[1], bbox2[2]])

        # Save it for later use in _list_outputs
        setattr(self, 'cdg_ijk_attr', cdg_ijk)
        setattr(self, 'bbox1_attr', bbox1)
        setattr(self, 'bbox2_attr', bbox2)
        _, base, _ = split_filename(self.inputs.apply_to)
        nb.save(cropped, base + '_cropped.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        outputs['cdg_ijk'] = getattr(self, 'cdg_ijk_attr')
        outputs['bbox1'] = getattr(self, 'bbox1_attr')
        outputs['bbox2'] = getattr(self, 'bbox2_attr')
        _, base, _ = split_filename(fname)
        outputs["cropped"] = os.path.abspath(base + '_cropped.nii.gz')

        return outputs
        
        
class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    apply_to = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    model = traits.Any(exists=True,
                    	desc='tensor flow model of pretrained brain_mask')


class ApplyMaskOutputSpec(TraitedSpec):
    """Output class

    Args:
        brain_mask (nb.Nifti1Image): brain mask of input NIfTI image
    """
    brain_mask = traits.File(exists=True,
                             desc='Segmented brain mask of NIfTI image ')


class ApplyMask(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            file img Nifti
            tensorflow model of pretrained brain mask
        output_spec (nb.Nifti1Image): brain mask for nifti image given

    Methods:

        _run_interface(runtime):
            apply a model of brain mask on a nifti image
    """
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        fname = self.inputs.apply_to
        apply_to = nb.load(fname)
        model = self.inputs.model
        # brain_mask = apply_mask(apply_to, model)
        brain_mask = None

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(brain_mask, base + '_brain_mask.nii.gz')

        return runtime


    def _list_outputs(self):
        """Get the absolute path to the output brain mask file."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        _, base, _ = split_filename(fname)
        outputs["brain_mask"] = os.path.abspath(base + '_brain_mask.nii.gz')

        return outputs        
