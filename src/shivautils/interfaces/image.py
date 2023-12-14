"""Custom nipype interfaces for image resampling/cropping and
other preliminary tasks"""
import os.path as op
from shivautils.postprocessing.pvs import quantify_clusters
from shivautils.postprocessing.basalganglia import create_basalganglia_slice_mask
from shivautils.postprocessing.wmh import metrics_clusters_latventricles
from shivautils.utils.stats import prediction_metrics, get_mask_regions
from shivautils.utils.preprocessing import normalization, crop, threshold, reverse_crop, make_offset, apply_mask
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, isdefined

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, TraitedSpec
import os
import warnings
import nibabel.processing as nip
import nibabel as nib
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
# from bokeh.io import export_png

import sys
sys.path.append('/mnt/devt')


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

    order = traits.Int(3, desc="Order of spline interpolation", usedefault=True)

    voxel_size = traits.Tuple(float, float, float,
                              desc='resampled voxel size',
                              mandatory=False)

    orientation = traits.Enum('RAS', 'LAS',
                              'RPS', 'LPS',
                              'RAI', 'LPI',
                              'RPI', 'LAP',
                              'RAP',
                              desc="orientation of image volume brain",
                              usedefault=True)

    border_too_close = traits.Int(0.15,
                                  usedefault=True,
                                  mandatory=False,
                                  desc=(
                                      'Ratio of the image that defines if the image origine '
                                      'is too close to the border of the image (and therefore '
                                      'likely erroneous). If it is, set the origin to the '
                                      'image center of mass.\n'
                                      'To disable this feature, set to 0'
                                  ))


class ConformOutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nib.Nifti1Image): transformed image
    """
    resampled = traits.File(exists=True,
                            desc='Image conformed to the required voxel size and shape.')


class Conform(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image):
            NIfTI image file to process
            dimensions (int, int, int): minimal dimension
            order (int): Order of spline interpolation
            voxel_size (float, float, float): Voxel size of final image
            orientation (string): orientation of the volume brain
        output_spec (nib.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            conform image to desired voxel sizes and dimensions
    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        """Run main programm
        Return: runtime
        """
        fname = self.inputs.img
        img = nib.funcs.squeeze_image(nib.load(fname))

        # Center the origin (to the image's center of mass) if it's too close to the image border
        close_to_border = self.inputs.border_too_close
        if close_to_border:  # only does it if close_to_border != 0
            rot, trans = nib.affines.to_matvec(img.affine)
            origin_ijk = np.linalg.inv(rot).dot(-trans)
            position_ratio = origin_ijk/img.shape
            too_close = False
            for dim_ratio in position_ratio:
                if (dim_ratio < close_to_border) or (dim_ratio > 1-close_to_border):
                    too_close = True
                    break
            if too_close:
                warn_msg = (
                    f"BAD ORIGIN: in {fname}\n"
                    "The image origin (coordinates 0x0x0 in image space) is too close to a border (so likely corrupted*).\n"
                    "To avoid problems during registration, a new affine was createdusing the center of mass as origin and "
                    "ignoring any rotation specified by the affine (but keeping voxel dim and left/right orientation).\n"
                    "This will misalign the masks (brain masks and cSVD biomarkers) compared to the raw images but will not "
                    "be a problem if you use the intensity normalized images from the img_preproc folder of the results."
                )
                warnings.warn(warn_msg)
                vol = img.get_fdata()
                cdg_ijk = np.round(ndimage.center_of_mass(vol))
                # As the affine may be corrupted, we discard it and create a simplified version (without rotations)
                simplified_rot = np.eye(3) * img.header['pixdim'][1:4]  # Keeping the voxel dimensions
                simplified_rot[0] *= img.header['pixdim'][0]  # Keeping the L/R orientation
                trans_centered = -simplified_rot.dot(cdg_ijk)
                simplified_affine_centered = nib.affines.from_matvec(simplified_rot, trans_centered)
                img = nib.Nifti1Image(vol, simplified_affine_centered)

        if not (isdefined(self.inputs.voxel_size)):
            # resample so as to keep FOV
            voxel_size = np.divide(np.multiply(img.header['dim'][1:4], img.header['pixdim'][1:4]).astype(np.double),
                                   self.inputs.dimensions)
        else:
            voxel_size = self.inputs.voxel_size
        resampled = nip.conform(img,
                                out_shape=self.inputs.dimensions,
                                voxel_size=voxel_size,
                                order=self.inputs.order,
                                cval=0.0,
                                orientation=self.inputs.orientation,
                                out_class=None)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nib.save(resampled, base + 'resampled.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["resampled"] = os.path.abspath(base +
                                               'resampled.nii.gz')
        return outputs


class Resample_from_to_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply Resample_from_to function on the
    image """
    moving_image = traits.File(exists=True,
                               desc='NIfTI file to resample',
                               mandatory=True)

    fixed_image = traits.File(exists=True,
                              desc='NIfTI file to use as reference for the resampling',
                              mandatory=True)

    spline_order = traits.Int(3,
                              desc="Order of spline interpolation",
                              usedefault=True)


class Resample_from_to_OutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nib.Nifti1Image): transformed image
    """
    resampled_image = traits.File(exists=True,
                                  desc='Nifti file of the image after resampling')


class Resample_from_to(BaseInterface):
    """Apply the nibabel function resample_from_to: put a Nifti image in the
    space (dimensions and resolution) of another Nifti image, while keeping
    the correct orientation and spatial position 

    Attributes:
        input_spec:
            moving_image: NIfTI file to resample
            fixed_image: NIfTI file used as template for the resampling
            splin_order: order used for the splin interpolation (see scipy.ndimage.affine_transform)
        output_spec: 
            resampled_image: Nifti file resampled

    Methods:
        _run_interface(runtime):
            Resample an image based on the dimensions and resolution of another
    """
    input_spec = Resample_from_to_InputSpec
    output_spec = Resample_from_to_OutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Return: runtime
        """
        in_img = nib.funcs.squeeze_image(nib.load(self.inputs.moving_image))
        ref_img = nib.funcs.squeeze_image(nib.load(self.inputs.fixed_image))
        resampled = nip.resample_from_to(in_img,
                                         ref_img,
                                         self.inputs.spline_order)

        nib.save(resampled, 'resampled.nii.gz')
        # Save it for later use in _list_outputs
        setattr(self, 'resampled_image', os.path.abspath('resampled.nii.gz'))
        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().trait_get()
        outputs['resampled_image'] = getattr(self, 'resampled_image')
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
        img_crop (nib.Nifti1Image): file img IRM nifti transformed
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
        input_spec (nib.Nifti1Image): NIfTI image input
        output_spec (nib.Nifti1Image): Intensity-normalized image

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
        img = nib.load(fname)
        if self.inputs.brain_mask:
            brain_mask = nib.load(self.inputs.brain_mask)
        else:
            brain_mask = self.inputs.brain_mask
        img_normalized, report, mode = normalization(img,
                                                     self.inputs.percentile,
                                                     brain_mask)

        # Save it for later use in _list_outputs
        setattr(self, 'mode', mode)
        with open('report.html', 'w', encoding='utf-8') as fid:
            fid.write(report)

        _, base, _ = split_filename(fname)
        setattr(self, 'report', os.path.abspath('report.html'))
        nib.save(img_normalized, base + '_img_normalized.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        outputs['report'] = getattr(self, 'report')
        fname = self.inputs.input_image
        _, base, _ = split_filename(fname)
        outputs['mode'] = getattr(self, 'mode')
        outputs["intensity_normalized"] = os.path.abspath(base +
                                                          '_img_normalized.nii.gz'
                                                          )
        return outputs


class ThresholdInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    img = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    threshold = traits.Float(0.5, exists=True, mandatory=True,
                             desc='Value of the treshold to apply to the image'
                             )

    sign = traits.Enum('+', '-',
                       usedefault=True,
                       desc='Whether to keep data above threshold or below threshold.')

    binarize = traits.Bool(False, exists=True,
                           desc='Binarize image')

    open = traits.Int(0, usedefault=True,
                      desc=('For binary opening of the clusters, radius of the ball used '
                            'as footprint (skip opening if <= 0'))

    clusterCheck = traits.Str('size', usedefault=True,
                              desc=("Can be 'size', 'top' or 'all'. Select one cluster "
                                    "(if not 'all') in the mask, biggest or highest on z-axis"))

    minVol = traits.Int(0, usedefault=True,
                        desc='Minimum size of the clusters to keep (if > 0, else keep all)')

    outname = traits.Str(mandatory=False,
                         desc='name of the output file. If not specified, will be the input witn "_thresholded" appended.')


class ThresholdOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): file img IRM nifti transformed
    """
    thresholded = traits.File(exists=True,
                              desc='Thresholded image')


class Threshold(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image):
            file img Nifti
            Threshold for the brain mask
        output_spec (nib.Nifti1Image): file img brain mask IRM nifti

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
        img = nib.funcs.squeeze_image(nib.load(fname))
        thresholded = threshold(img,
                                self.inputs.threshold,
                                sign=self.inputs.sign,
                                binarize=self.inputs.binarize,
                                open=self.inputs.open,
                                clusterCheck=self.inputs.clusterCheck,
                                minVol=self.inputs.minVol)

        # Save it for later use in _list_outputs
        if not isdefined(self.inputs.outname):
            _, base, _ = split_filename(fname)
            outname = base + '_thresholded.nii.gz'
        else:
            outname = self.inputs.outname
        nib.save(thresholded, outname)
        setattr(self, 'outname', os.path.abspath(outname))

        return runtime

    def _list_outputs(self):
        """
        Just gets the absolute path to the scheme file name
        """
        outputs = self.output_spec().get()
        outputs["thresholded"] = getattr(self, 'outname')
        return outputs


class CropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply cropping to the
    nifti image"""
    roi_mask = traits.File(exists=True,
                           desc='Mask for computation of center of gravity and'
                           'cropping coordinates', mandatory=False)

    apply_to = traits.File(exists=True,
                           desc='Image to crop', mandatory=True)

    final_dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                    default=(160, 214, 176),
                                    usedefault=True,
                                    desc='Final image array size in i, j, k.')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc='center of gravity of nifti image cropped with first'
                           'voxel intensities normalization', mandatory=False)

    default = traits.Enum("ijk", "xyz", usedefault=True, desc="Default crop center strategy (voxels or world).")


class CropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): Cropped image
    """
    cropped = traits.File(exists=True,
                          desc='nib.Nifti1Image: preprocessed image')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc="brain_mask's center of gravity")

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')

    cdg_ijk_file = traits.File(desc='Saved center of gravity of the brain_mask')

    bbox1_file = traits.File(desc='Saved bounding box first point')

    bbox2_file = traits.File(desc='Saved bounding box second point')


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
        if isdefined(self.inputs.roi_mask):
            mask = nib.load(self.inputs.roi_mask)
        else:
            # get from ijk
            mask = None
        if not isdefined(self.inputs.cdg_ijk):
            cdg_ijk = None
        else:
            cdg_ijk = self.inputs.cdg_ijk
        target = nib.load(self.inputs.apply_to)

        # process
        cropped, cdg_ijk, bbox1, bbox2 = crop(
            mask,
            target,
            self.inputs.final_dimensions,
            cdg_ijk,
            self.inputs.default,
            safety_marger=5)

        # TODO: rewrite this using np.savetxt and change reader where needed
        cdg_ijk = cdg_ijk[0], cdg_ijk[1], cdg_ijk[2]
        bbox1 = bbox1[0], bbox1[1], bbox1[2]
        bbox2 = bbox2[0], bbox2[1], bbox2[2]

        with open('cdg_ijk.txt', 'w') as fid:
            fid.write(str(cdg_ijk))
        cdg_ijk_file = os.path.abspath('cdg_ijk.txt')
        with open('bbox1.txt', 'w') as fid:
            fid.write(str(bbox1))
        bbox1_file = os.path.abspath('bbox1.txt')
        with open('bbox2.txt', 'w') as fid:
            fid.write(str(bbox2))
        bbox2_file = os.path.abspath('bbox2.txt')

        # Save it for later use in _list_outputs
        setattr(self, 'cdg_ijk', cdg_ijk)
        setattr(self, 'bbox1', bbox1)
        setattr(self, 'bbox2', bbox2)
        setattr(self, 'cdg_ijk_file', cdg_ijk_file)
        setattr(self, 'bbox1_file', bbox1_file)
        setattr(self, 'bbox2_file', bbox2_file)

        _, base, _ = split_filename(self.inputs.apply_to)
        nib.save(cropped, base + '_cropped.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.apply_to
        outputs['cdg_ijk'] = getattr(self, 'cdg_ijk')
        outputs['bbox1'] = getattr(self, 'bbox1')
        outputs['bbox2'] = getattr(self, 'bbox2')
        outputs['cdg_ijk_file'] = getattr(self, 'cdg_ijk_file')
        outputs['bbox1_file'] = getattr(self, 'bbox1_file')
        outputs['bbox2_file'] = getattr(self, 'bbox2_file')
        _, base, _ = split_filename(fname)
        outputs["cropped"] = os.path.abspath(base + '_cropped.nii.gz')

        return outputs


class Apply_mask_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    apply_to = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    model = traits.Any(exists=True,
                       desc='tensor flow model of pretrained brain_mask')


class Apply_mask_OutputSpec(TraitedSpec):
    """Output class

    Args:
        brain_mask (nib.Nifti1Image): brain mask of input NIfTI image
    """
    brain_mask = traits.File(exists=True,
                             desc='Segmented brain mask of NIfTI image ')


class Apply_mask(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image):
            file img Nifti
            tensorflow model of pretrained brain mask
        output_spec (nib.Nifti1Image): brain mask for nifti image given

    Methods:

        _run_interface(runtime):
            apply a model of brain mask on a nifti image
    """
    input_spec = Apply_mask_InputSpec
    output_spec = Apply_mask_OutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # TODO: reactivate or remove if superseded buy other procedure
        fname = self.inputs.apply_to
        apply_to = nib.load(fname)
        model = self.inputs.model
        # brain_mask = apply_mask(apply_to, model)
        brain_mask = None

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nib.save(brain_mask, base + '_brain_mask.nii.gz')

        return runtime

    def _list_outputs(self):
        """Get the absolute path to the output brain mask file."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        _, base, _ = split_filename(fname)
        outputs["brain_mask"] = os.path.abspath(base + '_brain_mask.nii.gz')

        return outputs


class ReverseCropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply reverse cropping to the
    nifti cropped image"""
    original_img = traits.File(exists=True,
                               desc='reference image in the original space to modify the cropped image',
                               mandatory=False)

    apply_to = traits.File(exists=True,
                           desc='Image to move in the original shape ', mandatory=True)

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')


class ReverseCropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): Cropped image
    """
    reverse_crop = traits.File(exists=True,
                               desc='nib.Nifti1Image: image cropped in the original space')


class ReverseCrop(BaseInterface):
    """Re-transform an image into the originel dimensions."""
    input_spec = CropInputSpec
    output_spec = CropOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        original_img = nib.load(self.inputs.original_img)
        apply_to = nib.load(self.inputs.apply_to)

        # process
        reverse_img = reverse_crop(
            original_img,
            apply_to,
            self.inputs.bbox1,
            self.inputs.bbox2)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(self.inputs.apply_to)
        nib.save(reverse_img, base + '_reverse_img.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        _, base, _ = split_filename(fname)
        outputs["reverse_img"] = os.path.abspath(base + '_reverse_img.nii.gz')

        return outputs


class DataGrabberSlicerInputSpec(BaseInterfaceInputSpec):
    """Input parameter to datagrabber for 3D slicer workflow"""
    args = traits.Any(desc='dictionary with arguments for workflow',
                      mandatory=False
                      )

    subject = traits.String(desc='name of subject')


class DataGrabberSlicerOutputSpec(TraitedSpec):
    """Output class

    Args:
        raw (str): path of raw image
        seg (str): path of segmentation brain mask image
    """
    raw = traits.File(exists=True,
                      desc='path for raw image')
    seg = traits.File(exists=True,
                      desc='path for segmentation brain mask image')


class DataGrabberSlicer(BaseInterface):
    """DataGrabber to handle corresponding image for 3D slicer workflow"""
    input_spec = DataGrabberSlicerInputSpec
    output_spec = DataGrabberSlicerOutputSpec

    def _run_interface(self, runtime):
        """Run DataGrabber 3D Slicer

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        args = self.inputs.args
        subject = self.inputs.subject
        raw = os.path.join(args['files_dir'], args['all_files'][subject]['raw'])
        seg = os.path.join(args['files_dir'], args['all_files'][subject]['seg'])
        setattr(self, 'raw', raw)
        setattr(self, 'seg', seg)

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().get()
        outputs['raw'] = getattr(self, 'raw')
        outputs['seg'] = getattr(self, 'seg')

        return outputs


class MakeOffsetInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply shifting image with
    a random offset"""
    img = traits.File(exists=True,
                      desc='image use to shifted with a random offset',
                      mandatory=False)

    offset = traits.Any(False,
                        desc="offset apply to images if specified")


class MakeOffsetOutputSpec(TraitedSpec):
    """Output class

    Args:
        shifted_img (nib.Nifti1Image): nifti image shifted with a random offset
    """
    shifted_img = traits.File(exists=True,
                              desc='nib.Nifti1Image: image shifted with a random offset')

    offset_number = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                 desc="voxel shift in image applied")


class MakeOffset(BaseInterface):
    """Re-transform an image into the originel dimensions."""
    input_spec = MakeOffsetInputSpec
    output_spec = MakeOffsetOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        img = nib.load(self.inputs.img)

        # process
        shifted_img, offset_number = make_offset(img, offset=self.inputs.offset)

        # Save it for later use in _list_outputs
        setattr(self, 'offset_number', offset_number)
        _, base, _ = split_filename(self.inputs.img)
        nib.save(shifted_img, base + '_shifted_img.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        outputs['offset_number'] = getattr(self, 'offset_number')
        _, base, _ = split_filename(fname)
        outputs["shifted_img"] = os.path.abspath(base + '_shifted_img.nii.gz')

        return outputs


class Regionwise_Prediction_metrics_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    img = traits.File(exists=True,
                      desc='Nifti file of the biomarker segmentation',
                      mandatory=True)

    thr_cluster_val = traits.Float(exists=True,
                                   desc='Value to threshold segmentation image',
                                   mandatory=True)

    thr_cluster_size = traits.Int(exists=True,
                                  desc='Value to threshold segmentation image',
                                  )

    biomarker_type = traits.String('biomarker',
                                   usedefault=True,
                                   mandatory=False,
                                   desc='Type of studied biomarker')

    brain_seg = traits.File(exists=True,
                            desc=('Brain mask or brain segmentation delimiting the parts '
                                  'of the biomarker segmentation ("img" argument) to explore'),
                            mandatory=True)

    region_list = traits.List(traits.Str,
                              value=['Whole_brain'],
                              desc='List of regions to use from the brain segmentation for the metrics',
                              mandatory=True)

    brain_seg_type = traits.Str('brain_mask',
                                desc=(
                                    'Type of brain segmentation provided. Can be "brain_mask" or "synthseg".'
                                    'To add more options, do it in the present interface'))


class Regionwise_Prediction_metrics_OutputSpec(TraitedSpec):
    """Output class

    Args:
        prediction_metrics_csv (csv): csv file with metrics about the cluster prediction
        file
    """
    biomarker_census_csv = traits.File(exists=True,
                                       desc='csv file listing all segmented biomarkers with their size')
    biomarker_stats_csv = traits.File(exists=True,
                                      desc='csv file with statistics on the segmented biomarkers')
    labelled_biomarkers = traits.File(exists=True,
                                      desc='Nifti file with labelled segmented biomarkers, keeping only those inside of the brain')


class Regionwise_Prediction_metrics(BaseInterface):
    """Get cluster metrics about one prediction file"""
    input_spec = Regionwise_Prediction_metrics_InputSpec
    output_spec = Regionwise_Prediction_metrics_OutputSpec

    def _run_interface(self, runtime):
        path_images = self.inputs.img
        thr_cluster_val = self.inputs.thr_cluster_val
        thr_cluster_size = self.inputs.thr_cluster_size
        brain_seg = self.inputs.brain_seg
        region_list = self.inputs.region_list
        brain_seg_type = self.inputs.brain_seg_type

        img = nib.load(path_images)
        segmentation_vol = img.get_fdata()
        brain_seg_vol = nib.load(brain_seg).get_fdata()

        if brain_seg_type == "brain_mask":
            region_dict = {'Region_names': region_list,
                           'Region_labels': [-1]}
            if len(region_list) > 1:
                raise ValueError('The list of regions given when using only the brain mask can only be 1 '
                                 '(taking the whole mask)')
        elif brain_seg_type == 'synthseg':  # Preparing brain segmentation from SynthSeg:
            # TODO: Implement and create a function that prepares the raw segmentation for periventricular regions, etc.
            raise NotImplementedError('Sorry, automatic brain segmentation is not implemented yet')
        else:
            raise ValueError(f'Unrecognised segmentation type: {brain_seg_type}. Should be "brain_mask" or "synthseg"')

        cluster_measures, cluster_stats, clusters_vol = prediction_metrics(
            segmentation_vol, thr_cluster_val, thr_cluster_size, brain_seg_vol, region_dict)

        biomarker = self.inputs.biomarker_type
        cluster_measures.to_csv(f'{biomarker}_census.csv')
        cluster_stats.to_csv(f'{biomarker}_stats.csv')
        clusters_im = nib.Nifti1Image(clusters_vol, img.affine, img.header)
        nib.save(clusters_im, f'labeled_{biomarker}s.nii.gz')

        # Set the attribute to pass as output
        setattr(self, 'biomarker_census_csv', os.path.abspath(f"{biomarker}_census.csv"))
        setattr(self, 'biomarker_stats_csv', os.path.abspath(f"{biomarker}_stats.csv"))
        setattr(self, 'labelled_biomarkers', os.path.abspath(f"labeled_{biomarker}s.nii.gz"))

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['biomarker_census_csv'] = getattr(self, 'biomarker_census_csv')
        outputs['biomarker_stats_csv'] = getattr(self, 'biomarker_stats_csv')
        outputs['labelled_biomarkers'] = getattr(self, 'labelled_biomarkers')
        return outputs


class Apply_mask_InputSpec(BaseInterfaceInputSpec):
    """Delete prediction outside space of brainmask"""
    segmentation = traits.File(exists=True,
                               desc='prediction file to change',
                               mandatory=False)

    brainmask = traits.Any(False,
                           desc="brainmask reference to delete prediction voxels")


class Apply_mask_OutputSpec(TraitedSpec):
    """Output class

    Args:
        shifted_img (nib.Nifti1Image): prediction file filtered with brainmask file
    """
    segmentation_filtered = traits.File(exists=True,
                                        desc='prediction file filtered with brainmask file')


class Apply_mask(BaseInterface):  # TODO: Unused, to remove
    """Re-transform an image into the originel dimensions."""
    input_spec = Apply_mask_InputSpec
    output_spec = Apply_mask_OutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        segmentation = nib.load(self.inputs.segmentation)
        brainmask = nib.load(self.inputs.brainmask)

        # process
        segmentation_filtered = apply_mask(segmentation, brainmask)

        # Save it for later use in _list_outputs
        setattr(self, 'segmentation_filtered', segmentation_filtered)
        _, base, _ = split_filename(self.inputs.segmentation)
        nib.save(segmentation_filtered, base + 'segmentation_filtered.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.segmentation
        _, base, _ = split_filename(fname)
        outputs["segmentation_filtered"] = os.path.abspath(base + 'segmentation_filtered.nii.gz')

        return outputs


class MaskRegionsInputSpec(BaseInterfaceInputSpec):
    """Filter voxels according to a specific regions"""
    img = traits.File(exists=True,
                      desc='Nifti file with segmented regions',
                      mandatory=False)

    list_labels_regions = traits.Any(False,
                                     desc="list of indices or labels regions")


class MaskRegionsOutputSpec(TraitedSpec):
    """Output class

    Args:
        mask_regions (nib.Nifti1Image): prediction file filtered with brainmask file
    """
    mask_regions = traits.File(exists=True,
                               desc='Nifti mask file filtered with specific regions')


class MaskRegions(BaseInterface):
    """Filter voxels according to a specific regions of brain"""
    input_spec = MaskRegionsInputSpec
    output_spec = MaskRegionsOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        img = nib.load(self.inputs.img)
        list_labels_regions = self.inputs.list_labels_regions

        # process
        mask_regions = get_mask_regions(img, list_labels_regions)

        # Save it for later use in _list_outputs
        setattr(self, 'mask_regions', mask_regions)
        _, base, _ = split_filename(self.inputs.img)
        nib.loadsave.save(mask_regions, base + 'mask_regions.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["mask_regions"] = os.path.abspath(base + 'mask_regions.nii.gz')

        return outputs


class QuantificationWMHLatVentriclesInputSpec(BaseInterfaceInputSpec):
    """Compute metrics about prediction around lateral ventricles"""
    wmh = traits.File(exists=True,
                      desc='Nifti file prediction',
                      mandatory=False)

    threshold_clusters = traits.Float(exists=True,
                                      desc='Threshold to compute clusters metrics')

    subject_id = traits.Any(desc="id for each subject")

    latventricles_distance_maps = traits.Any(False,
                                             desc="distance maps in millimeter to lateral ventricles")


class QuantificationWMHLatVentriclesOutputSpec(TraitedSpec):
    """Output class

    Args:
        csv_clusters_localization (csv): metrics of clusters around lateral ventricles
    """
    csv_clusters_localization = traits.File(exists=True,
                                            desc='metrics of clusters in lateral ventricles for each clusters')

    predictions_latventricles_DWMH = traits.File(exists=True,
                                                 desc='metrics of DMWH and lateral ventricles clusters')


class QuantificationWMHLatVentricles(BaseInterface):
    """Compute metrics about prediction in lateral ventricles"""
    input_spec = QuantificationWMHLatVentriclesInputSpec
    output_spec = QuantificationWMHLatVentriclesOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        wmh = nib.load(self.inputs.wmh)
        latventricles_distance_maps = nib.load(self.inputs.latventricles_distance_maps)
        subject_id = self.inputs.subject_id
        threshold_clusters = self.inputs.threshold_clusters

        # process
        (dataframe_clusters_localization,
         nib_latventricles_clusters,
         nib_clusters_DWMH,
         nib_voxels_latventricles,
         nib_voxels_DWMH,
         threshold) = metrics_clusters_latventricles(latventricles_distance_maps,
                                                     wmh,
                                                     subject_id,
                                                     threshold=threshold_clusters)

        out_csv = 'WMH_clusters_localization.csv'
        with open(out_csv, 'w') as f:
            dataframe_clusters_localization.to_csv(f, index=False)

        nib_total_clusters = nib_latventricles_clusters + nib_clusters_DWMH
        nib_total_voxels = nib_voxels_latventricles + nib_voxels_DWMH

        # Création du fichier CSV dans le Sink
        with open("predictions_latventricles_DWMH.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Écriture de l'en-tête du fichier CSV
            writer.writerow(["Cluster Threshold", "DWMH clusters number", "DWMH voxels number",
                            "Lateral Ventricles clusters number", "Lateral ventricles voxels number",
                             "Total clusters number", "Total voxels number"])

            # Ajout des lignes de données
            writer.writerow([threshold, nib_clusters_DWMH, nib_voxels_DWMH,
                             nib_latventricles_clusters, nib_voxels_latventricles,
                             nib_total_clusters, nib_total_voxels])

        # Save it for later use in _list_outputs
        setattr(self, 'csv_clusters_localization', os.path.abspath("WMH_clusters_localization.csv"))
        setattr(self, 'predictions_latventricles_DWMH', os.path.abspath("predictions_latventricles_DWMH.csv"))

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.wmh
        _, base, _ = split_filename(fname)
        outputs["csv_clusters_localization"] = getattr(self, 'csv_clusters_localization')
        outputs["predictions_latventricles_DWMH"] = getattr(self, 'predictions_latventricles_DWMH')

        return outputs


class BGMaskInputSpec(BaseInterfaceInputSpec):
    """Filter voxels according to a basal ganglia and insula regions"""
    segmented_regions = traits.File(exists=True,
                                    desc='Nifti file with segmented freesurfer synthseg regions',
                                    mandatory=False)


class BGMaskOutputSpec(TraitedSpec):
    """Output class

    Args:
        mask_regions (nib.Nifti1Image): prediction file filtered with basal ganglia regions
    """
    bg_mask = traits.File(exists=True,
                          desc='Nifti mask file filtered with specific regions')


class BGMask(BaseInterface):
    """Filter voxels according to a specific regions of brain"""
    input_spec = BGMaskInputSpec
    output_spec = BGMaskOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        regions_segmented_img = nib.load(self.inputs.segmented_regions)

        # process
        bg_mask = create_basalganglia_slice_mask(regions_segmented_img)

        # Save it for later use in _list_outputs
        setattr(self, 'bg_mask', bg_mask)
        _, base, _ = split_filename(self.inputs.segmented_regions)
        nib.loadsave.save(bg_mask, base + 'bg_mask.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.segmented_regions
        _, base, _ = split_filename(fname)
        outputs["bg_mask"] = os.path.abspath(base + 'bg_mask.nii.gz')

        return outputs


class PVSQuantificationibGInputSpec(BaseInterfaceInputSpec):
    """Compute metrics clusters and voxels about regions Basal Ganglia and deep white matter"""
    img = traits.File(exists=True,
                      desc='Nifti prediction pvs file')

    threshold_clusters = traits.Float(exists=True,
                                      desc='Threshold to compute clusters metrics')

    bg_mask = traits.File(exists=True,
                          desc='Basal Ganglions mask Nifti file',
                          mandatory=False)


class PVSQuantificationibGOutputSpec(TraitedSpec):
    """Output class

    Args:
        prediction_metrics_pvs (nib.Nifti1Image): CSV file with metrics about clusters and voxels Basal Ganglia and Deep White Matters regions
    """
    metrics_bg_pvs = traits.File(exists=True,
                                 desc='CSV file with metrics about clusters and voxels Basal Ganglia and Deep White Matters regions')


class PVSQuantificationibG(BaseInterface):
    """Filter voxels according to a specific regions of brain"""
    input_spec = PVSQuantificationibGInputSpec
    output_spec = PVSQuantificationibGOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        img = nib.load(self.inputs.img)
        bg_mask = nib.load(self.inputs.bg_mask)
        thr = self.inputs.threshold_clusters
        cl_filter = 6
        # process
        metrics_bg_pvs = quantify_clusters(img, bg_mask, thr, cl_filter)

        out_csv = 'metrics_bg_pvs.csv'
        with open(out_csv, 'w') as f:
            metrics_bg_pvs.to_csv(f, index=False)

        # Save it for later use in _list_outputs
        setattr(self, 'metrics_bg_pvs', os.path.abspath("metrics_bg_pvs.csv"))

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["metrics_bg_pvs"] = getattr(self, "metrics_bg_pvs")

        return outputs


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
    out_file = traits.File(exists=True)


class MakeDistanceMap(CommandLine):
    """Create distance maps using ventricles binarized maps (niimaths)."""

    _cmd = 'niimath'

    input_spec = MakeDistanceMapInputSpec
    output_spec = MakeDistanceMapOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
