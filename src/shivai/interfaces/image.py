"""Custom nipype interfaces for image resampling/cropping and
other preliminary tasks"""
import os.path as op
from shivai.postprocessing.lobarseg import lobar_and_wm_segmentation
from shivai.postprocessing.custom_parc import seg_for_pvs, seg_for_wmh, seg_from_mars
from shivai.postprocessing.pvs import quantify_clusters
from shivai.postprocessing.basalganglia import create_basalganglia_slice_mask
from shivai.postprocessing.wmh import metrics_clusters_latventricles
from shivai.utils.stats import prediction_metrics, get_mask_regions
from shivai.utils.preprocessing import normalization, crop, threshold, reverse_crop, make_offset, apply_mask, seg_cleaner
from shivai.utils.quality_control import create_edges, save_histogram, bounding_crop, overlay_brainmask
from shivai.utils.misc import label_clusters, cluster_registration
from shivai.interfaces.singularity import SingularityCommandLine, SingularityInputSpec
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, isdefined
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, TraitedSpec, TraitError)
import os
import warnings
import nibabel.processing as nip
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, ornt2axcodes
import numpy as np
import csv
import json
from scipy import ndimage
from scipy.io import loadmat

from itertools import permutations, product
# from bokeh.io import export_png

import sys
# sys.path.append('/mnt/devt')


def generate_orientation_codes():
    """Generate all valid 3-letter orientation codes from L/R, A/P, I/S.

    Returns a list containing every permutation of the three axis pairs with
    their possible directions, e.g. RAS, LPI, PIR, etc. RAS is placed first to
    keep the previous default unchanged.
    """
    axis_pairs = [('L', 'R'), ('A', 'P'), ('I', 'S')]
    orientations = [''.join(choice)
                    for axes in permutations(axis_pairs)
                    for choice in product(*axes)]
    orientations.sort(key=lambda code: code != 'RAS')  # Keep RAS as default
    return orientations


ORIENTATION_CODES = generate_orientation_codes()

# %% Preprocessing and general image manipulation


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

    adaptive_dim = traits.Bool(False,
                               desc='If True, adapt the dimensions to keep the FOV',
                               usedefault=True,
                               mandatory=False)

    order = traits.Int(3, desc="Order of spline interpolation", usedefault=True)

    voxel_size = traits.Tuple(float, float, float,
                              desc='Resampled voxel size',
                              mandatory=False)

    voxels_tolerance = traits.Tuple(float, float, float,
                                    default=(0, 0, 0),
                                    usedefault=True,
                                    desc='How close to the "voxel_size" can a dimension be to not be resampled',
                                    mandatory=False)

    orientation = traits.Enum(*ORIENTATION_CODES,
                              desc="orientation of image volume brain",
                              usedefault=True)

    ignore_bad_affine = traits.Bool(False,
                                    mandatory=False,
                                    usedefault=True,
                                    desc='If True, does not check if the affine is correct')


class ConformOutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nib.Nifti1Image): transformed image
    """
    resampled = traits.File(exists=True,
                            desc='Image conformed to the required voxel size and shape.')

    corrected_affine = traits.Any(desc=('If the conformed image had a bad affine matrix that needed to be '
                                        'corrected before the conformation, this output contains the corrected affine '
                                        'as a 2D Numpy array. Otherwise it stays undefined'))


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
        img: nib.Nifti1Image = nib.funcs.squeeze_image(nib.load(fname))

        simplified_affine_centered = None

        outdim = self.inputs.dimensions
        ori_dim = img.header['dim'][1:4]
        ori_vox_size = img.header["pixdim"][1:4]
        order = self.inputs.order

        if not (isdefined(self.inputs.voxel_size)):
            # resample so as to keep FOV
            voxel_size = np.divide(np.multiply(ori_dim, ori_vox_size).astype(np.double),
                                   outdim)
        else:
            voxel_size_param = np.array(self.inputs.voxel_size)
            voxel_size = voxel_size_param.copy()
            diff_size = np.abs(voxel_size-ori_vox_size)
            kept_vox_size = diff_size <= self.inputs.voxels_tolerance
            if all(kept_vox_size):
                order = 0  # No resampling needed
            # We keep the original voxel size if it's in the tolerance margin
            voxel_size[kept_vox_size] = ori_vox_size[kept_vox_size]
            voxel_size = tuple(voxel_size)
            if self.inputs.adaptive_dim:
                _outdim = []
                # adapt dimensions to keep expected FOV on dimensions where voxel sizes are kept
                for odim, kept, pvox, ivox in zip(outdim, kept_vox_size, voxel_size_param, ori_vox_size):
                    fov = odim * pvox
                    new_odim = np.ceil(fov / ivox).astype(int)
                    if kept:
                        _outdim.append(new_odim)
                    else:
                        _outdim.append(odim)
                outdim = tuple(_outdim)

        if voxel_size == tuple(ori_vox_size):
            order = 0  # No resampling needed

        if not self.inputs.ignore_bad_affine:
            # Create new affine (no rotation, centered on center of mass) if the affine is corrupted
            rot, trans = nib.affines.to_matvec(img.affine)
            rot_norm = rot.dot(np.diag(1/ori_vox_size))  # putting the rotation in isotropic space
            test1 = np.isclose(rot_norm.dot(rot_norm.T), np.eye(3), atol=0.0001).all()  # rot x rot.T must give an indentity matrix
            test2 = np.isclose(np.abs(np.linalg.det(rot_norm)), 1, atol=0.0001)  # Determinant for the rotation must be 1
            if not all([test1, test2]):
                warn_msg = (
                    f"BAD AFFINE: in {fname}\n"
                    "The image's affine is corrupted (not encoding a proper rotation).\n"
                    "To avoid problems during registration, a new affine was createdusing the center of mass as origin and "
                    "ignoring any rotation specified by the affine (but keeping voxel dim and left/right orientation).\n"
                    "This will misalign the masks (brain masks and cSVD biomarkers) compared to the raw images but will not "
                    "be a problem if you use the intensity normalized images from the img_preproc folder of the results."
                )
                warnings.warn(warn_msg)
                vol = img.get_fdata()
                cdg_ijk = np.round(ndimage.center_of_mass(vol))
                # As the affine may be corrupted, we discard it and create a simplified version (without rotations)
                simplified_rot = np.eye(3) * ori_vox_size  # Keeping the voxel dimensions
                simplified_rot[0] *= img.header['pixdim'][0]  # Keeping the L/R orientation
                trans_centered = -simplified_rot.dot(cdg_ijk)
                simplified_affine_centered = nib.affines.from_matvec(simplified_rot, trans_centered)
                img = nib.Nifti1Image(vol.astype('f'), simplified_affine_centered)
        setattr(self, 'corrected_affine', simplified_affine_centered)

        resampled = nip.conform(img,
                                out_shape=outdim,
                                voxel_size=voxel_size,
                                order=order,
                                cval=0.0,
                                orientation=self.inputs.orientation,
                                out_class=None)

        # Make sure the resampled data is still in the correct range (cubic spline can mess it up)
        # And save it as float32 to ensure there is no problem.
        vol = img.get_fdata(dtype='f')
        resampled_vol = resampled.get_fdata(dtype='f')
        resampled_vol[resampled_vol < vol.min()] = vol.min()
        resampled_vol[resampled_vol > vol.max()] = vol.max()
        resampled_correct = nib.Nifti1Image(resampled_vol, resampled.affine)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nib.save(resampled_correct, base + '_resampled.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["resampled"] = os.path.abspath(base + '_resampled.nii.gz')
        if self.corrected_affine is not None:
            outputs['corrected_affine'] = self.corrected_affine
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

    out_name = traits.Str('resampled.nii.gz',
                          usedefault=True,
                          desc='Output filename')

    out_suffix = traits.Str(mandatory=False,
                            desc=('If set, uses the moving image name and add the given suffix to create the '
                                  'output filename. This will override the "out_name" input of the node.'))

    corrected_affine = traits.Any(desc=('Affine matrix to use instead of the input affine, if defined, '
                                        'as it means that the original space had a bad affine (e.g. img1 '
                                        'needed a correction before its conformation)'))


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
        in_img = nib.load(self.inputs.moving_image)
        if isdefined(self.inputs.corrected_affine):
            # Apply the affine correction to the image before resampling
            # (assuming that in_img is in the same space as the input to the conform node)
            trans_centered = self.inputs.corrected_affine[:3, 3]
            simpl_rot_sign = np.sign(np.diag(self.inputs.corrected_affine[:3, :3]))
            simplified_rot = np.eye(3) * in_img.header['pixdim'][1:4] * simpl_rot_sign  # Keeping the voxel dimensions and signs
            simplified_affine_centered = nib.affines.from_matvec(simplified_rot, trans_centered)
            in_img.set_sform(affine=simplified_affine_centered)
            in_img.set_qform(affine=simplified_affine_centered)
        in_img = nib.funcs.squeeze_image(in_img)
        ref_img = nib.funcs.squeeze_image(nib.load(self.inputs.fixed_image))
        if isdefined(self.inputs.out_suffix):
            basename = os.path.basename(self.inputs.moving_image).split('.nii')[0]
            self.outname = basename + self.inputs.out_suffix + '.nii.gz'
        else:
            self.outname = self.inputs.out_name
        resampled = nip.resample_from_to(in_img,
                                         ref_img,
                                         self.inputs.spline_order)

        nib.save(resampled, self.outname)
        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().trait_get()
        outputs['resampled_image'] = os.path.abspath(self.outname)
        return outputs


class IntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply normalization to the nifti
    image"""
    input_image = traits.File(exists=True, desc='NIfTI image input.',
                              mandatory=True)

    percentile = traits.Float(exists=True, desc='value to threshold above this '
                              'percentile',
                              mandatory=True)

    brain_mask = traits.File(desc='brain_mask to adapt normalization to '
                             'the greatest number', mandatory=False)

    inverse = traits.Bool(False,
                          desc='If set to True, the normalized value of the voxels in '
                               'the brain will be "inversed" (1-val)',
                          mandatory=False,
                          usedefault=True
                          )


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
                                                     brain_mask,
                                                     self.inputs.inverse)

        # Save it for later use in _list_outputs
        setattr(self, 'mode', mode)
        with open('report.html', 'w', encoding='utf-8') as fid:
            fid.write(report)

        _, base, _ = split_filename(fname)
        setattr(self, 'report', os.path.abspath('report.html'))
        if self.inputs.inverse:
            self.outname = base + '_img_normalized_inv.nii.gz'
        else:
            self.outname = base + '_img_normalized.nii.gz'
        nib.save(img_normalized, self.outname)

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        outputs['report'] = getattr(self, 'report')
        fname = self.inputs.input_image
        _, base, _ = split_filename(fname)
        outputs['mode'] = getattr(self, 'mode')
        outputs["intensity_normalized"] = os.path.abspath(self.outname)
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
                                rad=self.inputs.open,
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
    """Re-transform an image into the original dimensions."""
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


class Segmentation_Cleaner_InputSpec(BaseInterfaceInputSpec):
    input_seg = traits.File(exists=True,
                            mandatory=True,
                            desc='Raw SynthSeg ouput')

    max_island_size = traits.Int(100,
                                 mandatory=False,
                                 usedefault=True,
                                 desc=('Maximum size (in voxel) under which a detached part of a brain '
                                       'region is considered as an "island" to be removed.'))

    seg_type = traits.Str('synthseg',
                          mandatory=False,
                          usedefault=True,
                          desc=('Type of segmentation input in "input_seg". Is used to define '
                                'which label values should be ignored.'))


class Segmentation_Cleaner_OutputSpec(TraitedSpec):
    ouput_seg = traits.File(exists=True,
                            desc='Cleaned SynthSeg segmentation')

    sunk_islands = traits.File(exists=True,
                               desc=('"Islands" that have been removed in "ouput_seg". If no island was '
                                     'detected, does not return any file.'))

    # kept_islands = traits.File(exists=True,
    #                            desc=('"Islands" that have been kept in "ouput_seg" as they have mutliple neighbors, '
    #                                  'but are suspecious. If no island was detected, does not return any file.'))


class Segmentation_Cleaner(BaseInterface):
    """Remove small 'islands' parcels detached from the main region in a brain parcellisation"""
    input_spec = Segmentation_Cleaner_InputSpec
    output_spec = Segmentation_Cleaner_OutputSpec

    def _run_interface(self, runtime):
        if self.inputs.seg_type == 'synthseg':
            ignore_list = [24]  # CSF
        seg_im = nib.load(self.inputs.input_seg)
        seg_vol = seg_im.get_fdata().astype('int16')
        cleaned_vol, sunk_islands_vol = seg_cleaner(seg_vol,
                                                    self.inputs.max_island_size,
                                                    ignore_list)

        cleaned_im = nib.Nifti1Image(cleaned_vol, affine=seg_im.affine)
        outname = 'cleaned_' + os.path.basename(self.inputs.input_seg)
        nib.save(cleaned_im, outname)
        setattr(self, 'outfile', os.path.abspath(outname))

        if np.any(sunk_islands_vol):  # islands_vol not filled with 0 only
            sunk_islands_im = nib.Nifti1Image(sunk_islands_vol, affine=seg_im.affine)
            sunk_island_name = 'removed_clusters_' + os.path.basename(self.inputs.input_seg)
            nib.save(sunk_islands_im, sunk_island_name)
            setattr(self, 'sunk_islands', os.path.abspath(sunk_island_name))

        # if np.any(kept_islands_vol):  # islands_vol not filled with 0 only
        #     kept_islands_im = nib.Nifti1Image(kept_islands_vol, affine=seg_im.affine)
        #     kept_island_name = 'kept_clusters_' + os.path.basename(self.inputs.input_seg)
        #     nib.save(kept_islands_im, kept_island_name)
        #     setattr(self, 'kept_islands', os.path.abspath(kept_island_name))

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().trait_get()
        outputs['ouput_seg'] = getattr(self, 'outfile')
        if hasattr(self, 'sunk_islands'):
            outputs['sunk_islands'] = getattr(self, 'sunk_islands')
        if hasattr(self, 'kept_islands'):
            outputs['kept_islands'] = getattr(self, 'kept_islands')
        return outputs


# %% Quality control
class Isocontour_InputSpec(BaseInterfaceInputSpec):
    path_image = traits.File(exists=True,
                             mandatory=True,
                             desc='Registered image from which the contours will be extracted'
                             )
    path_ref_image = traits.File(exists=True,
                                 mandatory=True,
                                 desc='Reference image on which the contours will be overlaid'
                                 )
    path_brainmask = traits.File(exists=True,
                                 mandatory=True,
                                 desc='Brain mask to delimit where to look for the brain contours'
                                 )
    nb_of_slices = traits.Int(12,
                              usedefault=True,
                              mandatory=False,
                              desc='Number of brain slices to display')
    slice_orient = traits.Enum('axial', 'sagittal', 'coronal',
                               usedefault=True,
                               mandatory=False,
                               desc='Orientation of the displayed slices (default is "axial")'
                               )


class Isocontour_OutputSpec(TraitedSpec):

    qc_coreg = traits.File(exists=True,
                           desc='PNG file with the displayed slices for QC')


class Isocontour(BaseInterface):
    """Generates an image showing the brain contours from a registered image overlayed a the reference image"""
    input_spec = Isocontour_InputSpec
    output_spec = Isocontour_OutputSpec

    def _run_interface(self, runtime):
        path_image = self.inputs.path_image
        path_ref_image = self.inputs.path_ref_image
        path_brainmask = self.inputs.path_brainmask
        nb_of_slices = self.inputs.nb_of_slices
        slice_orient = self.inputs.slice_orient

        qc_coreg = create_edges(path_image, path_ref_image, path_brainmask, nb_of_slices, slice_orient)
        setattr(self, 'qc_coreg', qc_coreg)  # qc_coreg is already an absolute path

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['qc_coreg'] = getattr(self, 'qc_coreg')
        return outputs


class Save_Histogram_InputSpec(BaseInterfaceInputSpec):
    img_normalized = traits.File(exists=True,
                                 mandatory=True,
                                 desc='Intensity-normalized image')

    bins = traits.Int(64,
                      usedefault=True,
                      mandatory=False,
                      desc='Number of brain bins to display in the histogram')


class Save_Histogram_OutputSpec(TraitedSpec):

    histo = traits.File(exists=True,
                        desc='PNG file with the displayed histogram')

    peak = traits.Float(desc='Most frequent value in the histogram (away from min and max values)')


class Save_Histogram(BaseInterface):
    """Generates the histogram of a normalized image and get the its peak value (away from min and max values)"""
    input_spec = Save_Histogram_InputSpec
    output_spec = Save_Histogram_OutputSpec

    def _run_interface(self, runtime):
        img_normalized = self.inputs.img_normalized
        bins = self.inputs.bins

        histo, peak = save_histogram(img_normalized, bins)
        setattr(self, 'histo', histo)  # histo is already an absolute path
        setattr(self, 'peak', peak)

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['histo'] = getattr(self, 'histo')
        outputs['peak'] = getattr(self, 'peak')
        return outputs


class Mask_and_Crop_QC_InputSpec(BaseInterfaceInputSpec):
    brain_img = traits.File(exists=True,
                            mandatory=True,
                            desc='Reference nifti images to overlay with brainmask and crop box')
    brainmask = traits.File(exists=True,
                            mandatory=True,
                            desc='Brain mask file')
    bbox1 = traits.Tuple(mandatory=True,
                         desc='First coordinates for the crop-box')
    bbox2 = traits.Tuple(mandatory=True,
                         desc='Second coordinates for the crop-box')
    slice_coord = traits.Tuple(mandatory=True,
                               desc='Coordinates to be used to place the slices shown')


class Mask_and_Crop_QC_OutputSpec(TraitedSpec):

    crop_brain_img = traits.File(exists=True,
                                 desc='PNG file with the brain, mask, and crop-box')


class Mask_and_Crop_QC(BaseInterface):
    """Generates an image showing the brain mask and the crop-box overlayed on the original brain"""
    input_spec = Mask_and_Crop_QC_InputSpec
    output_spec = Mask_and_Crop_QC_OutputSpec

    def _run_interface(self, runtime):
        brain_img = self.inputs.brain_img
        brainmask = self.inputs.brainmask
        bbox1 = self.inputs.bbox1
        bbox2 = self.inputs.bbox2
        slice_coord = self.inputs.slice_coord

        crop_brain_img = bounding_crop(brain_img, brainmask, bbox1, bbox2, slice_coord)
        setattr(self, 'crop_brain_img', crop_brain_img)  # crop_brain_img is already an absolute path

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['crop_brain_img'] = getattr(self, 'crop_brain_img')
        return outputs


class Brainmask_Overlay_InputSpec(BaseInterfaceInputSpec):
    img_ref = traits.File(exists=True,
                          mandatory=True,
                          desc='Reference nifti images to overlay with the brainmask')
    brainmask = traits.File(exists=True,
                            mandatory=True,
                            desc='Brain mask file')
    outname = traits.Str(usedefault=True,
                         mandatory=True,
                         desc='Name of the output image')
    cols_nb = traits.Int(6,
                         mandatory=False,
                         usedefault=True,
                         desc='Number of columns in the ouput images')
    orient = traits.Str('XYZ',
                        mandatory=False,
                        usedefault=True,
                        desc=('Orientations to slice through. Can be "X", "Y", or "Z". '
                              'Each instance of an orientation letter will add a row '
                              'with slices in this orientation in the images (e.g. "ZZZ" '
                              'will ouput an image with 3 rows showing slices in the Z dim)'))
    alpha = traits.Float(0.5,
                         usedefault=True,
                         mandatory=False,
                         desc='Alpha transparency for the overlaid mask. Default is 0.5.')
    fov_mask = traits.File(mandatory=False,
                           desc=('Nifti file with binary image that will be used to define '
                                 'the field of view of the slices by both masking the images '
                                 'and cropping to adjust to the mask.'))


class Brainmask_Overlay_OutputSpec(TraitedSpec):

    overlayed_brainmask = traits.File(exists=True,
                                      desc='PNG file with the brainmask overlayed on the brain')


class Brainmask_Overlay(BaseInterface):
    """Generates an image showing the brain mask and the crop-box overlayed on the original brain"""
    input_spec = Brainmask_Overlay_InputSpec
    output_spec = Brainmask_Overlay_OutputSpec

    def _run_interface(self, runtime):
        img_ref = self.inputs.img_ref
        brainmask = self.inputs.brainmask
        outname = self.inputs.outname
        cols_nb = self.inputs.cols_nb
        slice_orients = self.inputs.orient
        alpha = self.inputs.alpha

        ref_im = nib.load(img_ref)
        ref_vol = ref_im.get_fdata().squeeze()
        ref_orient = ornt2axcodes(io_orientation(ref_im.affine))
        brainmask_im = nib.load(brainmask)
        mask_orient = ornt2axcodes(io_orientation(brainmask_im.affine))
        brainmask_vol = brainmask_im.get_fdata().squeeze().astype(bool)
        if not ref_orient == mask_orient:
            raise ValueError('The brainmask/segmentation and the reference image do not share the same orientation. '
                             'This should not be possible...')

        if isdefined(self.inputs.fov_mask):
            fov_mask_vol = nib.load(self.inputs.fov_mask).get_fdata().squeeze().astype(bool)
            fov_box = np.ones(fov_mask_vol.shape)
            fov_min_max = [(ind.min(), ind.max()) for ind in np.where(fov_mask_vol)]
            # There must be a better way...
            fov_box_shape = tuple(ind_max - ind_min for ind_min, ind_max in fov_min_max)
            fov_box[:fov_min_max[0][0], :, :] = 0
            fov_box[fov_min_max[0][1]:, :, :] = 0
            fov_box[:, :fov_min_max[1][0], :] = 0
            fov_box[:, fov_min_max[1][1]:, :] = 0
            fov_box[:, :, :fov_min_max[2][0]] = 0
            fov_box[:, :, fov_min_max[2][1]:] = 0
            fov_box = fov_box.astype(bool)
            brainmask_vol_masked = brainmask_vol * fov_mask_vol  # masking
            brainmask_vol = brainmask_vol_masked[fov_box].reshape(fov_box_shape)  # new fov
            ref_vol_masked = ref_vol * fov_mask_vol  # masking
            ref_vol = ref_vol_masked[fov_box].reshape(fov_box_shape)  # new fov

        overlayed_brainmask = overlay_brainmask(ref_vol, brainmask_vol, outname, cols_nb, slice_orients, mask_orient, alpha)

        setattr(self, 'overlayed_brainmask', overlayed_brainmask)  # overlayed_brainmask is already an absolute path

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['overlayed_brainmask'] = getattr(self, 'overlayed_brainmask')
        return outputs


# %% Predictions and postprocessing
class Label_clusters_InputSpec(BaseInterfaceInputSpec):
    biomarker_raw = traits.File(exists=True,
                                desc='Nifti file of the biomarker segmentation directly from the AI model',
                                mandatory=True)

    thr_cluster_val = traits.Float(exists=True,
                                   desc='Value to threshold segmentation image',
                                   mandatory=True)

    thr_cluster_size = traits.Int(exists=True,
                                  desc='Value to threshold segmentation image',
                                  )

    brain_seg = traits.File(exists=True,
                            desc=('Brain mask or brain segmentation delimiting the parts '
                                  'of the biomarker segmentation ("img" argument) to explore'),
                            mandatory=True)

    out_name = traits.Str('labelled_clusters.nii.gz',
                          mandatory=False,
                          desc='Output name of the file containing the labelled biomarkers')


class Label_clusters_OutputSpec(TraitedSpec):

    labelled_biomarkers = traits.File(exists=True,
                                      desc='Nifti file with labelled segmented biomarkers, keeping only those inside of the brain')


class Label_clusters(BaseInterface):
    """Generates an image showing the brain mask and the crop-box overlayed on the original brain"""
    input_spec = Label_clusters_InputSpec
    output_spec = Label_clusters_OutputSpec

    def _run_interface(self, runtime):
        biomarker_raw = self.inputs.biomarker_raw
        thr_cluster_val = self.inputs.thr_cluster_val
        thr_cluster_size = self.inputs.thr_cluster_size
        brain_seg = self.inputs.brain_seg
        out_name = self.inputs.out_name

        biomarker_im = nib.load(biomarker_raw)
        biomarker_vol = biomarker_im.get_fdata()
        brain_seg_vol = nib.load(brain_seg).get_fdata()

        labelled_clusters = label_clusters(biomarker_vol, brain_seg_vol, thr_cluster_val, thr_cluster_size)
        labelled_clusters_im = nib.Nifti1Image(labelled_clusters.astype('int16'), affine=biomarker_im.affine)
        nib.save(labelled_clusters_im, out_name)
        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['labelled_biomarkers'] = op.abspath(self.inputs.out_name)
        return outputs


class Regionwise_Prediction_metrics_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    labelled_clusters = traits.File(exists=True,
                                    desc='Biomarker clusters labelled with unique integers',
                                    mandatory=True)

    biomarker_type = traits.String('biomarker',
                                   usedefault=True,
                                   mandatory=False,
                                   desc='Type of studied biomarker')

    brain_seg = traits.File(exists=True,
                            desc=('Brain mask or brain segmentation delimiting the parts '
                                  'of the biomarker segmentation ("img" argument) to explore'),
                            mandatory=True)

    region_dict = traits.Dict(key_trait=traits.Str,
                              value_trait=traits.Int,
                              value={'Whole brain': -1},
                              desc=('Dictionnary with keys = brain region names, '
                                    'and values = brain region labels (i.e. the corresponding value in brain_seg)\n'
                                    'To be used in conjonction with brain_seg_type = "custom"'),
                              mandatory=False,
                              usedefault=True)

    region_list = traits.List(traits.Str,
                              value=['Whole brain'],
                              desc='List of regions to use from the Synthseg segmentation (specifically) for the metrics',
                              mandatory=False)

    brain_seg_type = traits.Str('brain_mask',
                                desc='Type of brain segmentation provided. Can be "brain_mask", "synthseg", or "custom".')

    # prio_labels = traits.List(traits.Str,
    #                           mandatory=False,
    #                           usedefault=True,
    #                           desc=('For the given labels (must have corresponding keys in region_dict), the clusters '
    #                                 'will be assigned to these labels if they just touch them, instead of using the '
    #                                 'winner-takes-all approach. If there is a competition between priority label, will '
    #                                 'use w-t-a approach among them.'))


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
    biomarker_stats_wide_csv = traits.File(exists=True,
                                           desc='csv file with statistics on the segmented biomarkers with "wide')


class Regionwise_Prediction_metrics(BaseInterface):
    """Get cluster metrics about one prediction file"""
    input_spec = Regionwise_Prediction_metrics_InputSpec
    output_spec = Regionwise_Prediction_metrics_OutputSpec

    def _run_interface(self, runtime):
        labelled_clusters = self.inputs.labelled_clusters
        brain_seg = self.inputs.brain_seg
        region_list = self.inputs.region_list
        brain_seg_type = self.inputs.brain_seg_type
        biomarker = self.inputs.biomarker_type
        # prio_labels = self.inputs.prio_labels

        clusters_im = nib.load(labelled_clusters)
        clusters_vol = clusters_im.get_fdata().astype(int)
        brain_seg_vol = nib.load(brain_seg).get_fdata()

        if brain_seg_type == "brain_mask":
            region_dict = {'Whole brain': -1}
            if len(region_list) > 1:
                raise ValueError('The list of regions given when using only the brain mask can only be 1 '
                                 '(taking the whole mask)')
        elif brain_seg_type in ['synthseg', 'freesurfer']:
            if isdefined(self.inputs.region_dict):
                region_dict = self.inputs.region_dict
            else:  # Requieres "region_list" given as input, works with Synthseg (and Freesurfer) labels
                fs_labels = {'Whole brain': -1,  # To complete
                             'Left cerebral WM': 2,
                             'Left cerebral ctx': 3,
                             'Left cerebellum WM': 7,
                             'Left cerebellum ctx': 8,
                             'Left thalamus': 10,
                             'Left caudate': 11,
                             'Left putamen': 12,
                             'Left pallidum': 13,
                             'Brain stem': 16,
                             'Left hippocampus': 17,
                             'Left amygdala': 18,
                             'Left accumbens area': 26,
                             'Left ventral DC': 28,
                             'Right cerebral WM': 41,
                             'Right cerebral ctx': 42,
                             'Right cerebellum WM': 46,
                             'Right cerebellum ctx': 47,
                             'Right thalamus': 49,
                             'Right caudate': 50,
                             'Right putamen': 51,
                             'Right pallidum': 52,
                             'Right hippocampus': 53,
                             'Right amygdala': 54,
                             'Right accumbens area': 58,
                             'Right ventral DC': 60,
                             }
                brain_seg_vol[brain_seg_vol == 24] = 0  # label 24 = CSF
                brain_seg_vol[(brain_seg_vol >= 1000) | (brain_seg_vol <= 1035)] = 8  # parc labels for left ctx
                brain_seg_vol[(brain_seg_vol >= 2000) | (brain_seg_vol <= 2035)] = 42  # parc labels for right ctx
                region_dict = {region: fs_labels[region] for region in region_list if region in fs_labels.keys()}
        elif brain_seg_type == 'custom':
            region_dict = self.inputs.region_dict
            if 'Whole brain' not in region_dict:
                region_dict = dict({'Whole brain': -1}, **region_dict)  # To ensure "Whole brain" is the first key
        else:
            raise TraitError(f'Unrecognised segmentation type: {brain_seg_type}. Should be "brain_mask", "synthseg" or "custom"')

        if biomarker == 'wmh' and brain_seg_type == 'synthseg':
            prio_labels = ['Left PV WM', 'Right PV WM']
        else:
            prio_labels = []

        cluster_measures, cluster_stats = prediction_metrics(
            clusters_vol, brain_seg_vol, region_dict, prio_labels)

        cluster_measures.to_csv(f'{biomarker}_census.csv', index=False)
        cluster_stats.to_csv(f'{biomarker}_stats.csv', index=False)

        # Set cluster_stats in wide format
        metrics_col = cluster_stats.columns.to_list()
        metrics_col.remove('Region')
        cluster_stats['Index'] = 'Values'  # Dummy index for pivot_table
        cluster_stats_wide = cluster_stats.pivot_table(index='Index',
                                                       columns=['Region'],
                                                       values=metrics_col,
                                                       dropna=False)
        cluster_stats_wide.columns = [f'{s2} - {s1}' for (s1, s2) in cluster_stats_wide.columns.tolist()]
        cluster_stats_wide.reset_index(inplace=True)
        cluster_stats_wide.set_index('Index', inplace=True)
        cluster_stats_wide = cluster_stats_wide.reindex(sorted(cluster_stats_wide.columns), axis=1)
        cluster_stats_wide.to_csv(f'{biomarker}_stats_wide.csv', float_format='%.2f', index=False)

        # Set the attribute to pass as output
        setattr(self, 'biomarker_census_csv', os.path.abspath(f"{biomarker}_census.csv"))
        setattr(self, 'biomarker_stats_csv', os.path.abspath(f"{biomarker}_stats.csv"))
        setattr(self, 'biomarker_stats_wide_csv', os.path.abspath(f"{biomarker}_stats_wide.csv"))

        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['biomarker_census_csv'] = getattr(self, 'biomarker_census_csv')
        outputs['biomarker_stats_csv'] = getattr(self, 'biomarker_stats_csv')
        outputs['biomarker_stats_wide_csv'] = getattr(self, 'biomarker_stats_wide_csv')
        return outputs


class Labelled_Clusters_Registration_InputSpec(BaseInterfaceInputSpec):
    input_image = traits.File(exists=True,
                              desc='Biomarker clusters labelled with unique integers',
                              mandatory=True)
    reference_image = traits.File(exists=True,
                                  desc='Image defining the arriving space after the registration',
                                  mandatory=True)
    transform_affine = traits.File(exists=True,
                                   desc='Affine of the transformation from ANTs',
                                   mandatory=True)
    out_name = traits.Str('registered_clusters.nii.gz',
                          usedefault=True,
                          mandatory=False,
                          desc='Ouput file name')


class Labelled_Clusters_Registration_OutputSpec(TraitedSpec):
    output_image = traits.File(exists=True,
                               desc='Biomarker labelled clusters registered in new space')


class Labelled_Clusters_Registration(BaseInterface):
    """Register clusers in a new space, ensuring all clusters are kept by moving them through voxel coordinates"""
    input_spec = Labelled_Clusters_Registration_InputSpec
    output_spec = Labelled_Clusters_Registration_OutputSpec

    def _run_interface(self, runtime):
        input_im = nib.load(self.inputs.input_image)
        ref_im = nib.load(self.inputs.reference_image)
        mat = loadmat(self.inputs.transform_affine)
        key_name = [k for k in mat if 'AffineTransform_' in k][0]  # AffineTransform_*_3_3
        transform_affine_raw = mat[key_name]
        transform_affine = np.eye(4)
        transform_affine[:3, :3] = transform_affine_raw[:9].reshape((3, 3))
        transform_affine[:3, 3] = transform_affine_raw[9:12].squeeze()
        # TODO: Make cluster_registration work
        clusters_reg_im = cluster_registration(input_im, ref_im, transform_affine)
        nib.save(clusters_reg_im, self.inputs.out_name)
        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['output_image'] = os.path.abspath(self.inputs.out_name)
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


class MakeDistanceMap_Singularity_InputSpec(SingularityInputSpec, MakeDistanceMapInputSpec):
    """MakeDistanceMap input specification (singularity mixin).

    Inherits from Singularity command line fields.
    """


class MakeDistanceMap_Singularity(SingularityCommandLine):
    """Create distance maps using ventricles binarized maps (niimaths)."""

    _cmd = 'niimath'

    input_spec = MakeDistanceMapInputSpec
    output_spec = MakeDistanceMapOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class Parc_from_Synthseg_InputSpec(BaseInterfaceInputSpec):
    brain_seg = traits.File(exists=True,
                            mandatory=True,
                            desc='Synthseg (or comparible FreeSurfer) brain segmentation.')


class Parc_from_Synthseg_OutputSpec(TraitedSpec):
    brain_parc = traits.File(exists=True,
                             desc='Brain parcellation with lobar gm and wm, juxtacortical/deep/perivascular wm, and more')


class Parc_from_Synthseg(BaseInterface):
    '''
    Transform a Synthseg segmentation into a wm & gm parcellation that can be used in our cSVD biomarkers metrics
    '''
    input_spec = Parc_from_Synthseg_InputSpec
    output_spec = Parc_from_Synthseg_OutputSpec

    def _run_interface(self, runtime):
        seg_im = nib.load(self.inputs.brain_seg)
        seg_vol = seg_im.get_fdata().astype(int)
        custom_parc = lobar_and_wm_segmentation(seg_vol)
        custom_parc_im = nib.Nifti1Image(custom_parc, seg_im.affine)
        nib.save(custom_parc_im, 'derived_parc.nii.gz')
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['brain_parc'] = op.abspath('derived_parc.nii.gz')
        return outputs


class Brain_Seg_for_biomarker_InputSpec(BaseInterfaceInputSpec):

    brain_seg = traits.File(exists=True,
                            mandatory=True,
                            desc='"derived_parc" brain segmentation from "Parc_from_Synthseg" node.')

    custom_parc = traits.Str('mars',
                             usedefault=True,
                             desc='Type of custom parcellisation scheme to use. Can be "pvs", "wmh, or "mars"')

    out_file = traits.Str('Brain_Seg_for_biomarker.nii.gz',
                          usedefault=True,
                          desc='Filename of the ouput segmentation')


class Brain_Seg_for_biomarker_OutputSpec(TraitedSpec):
    brain_seg = traits.File(exists=True,
                            desc='Brain segmentation, derived from synthseg segmentation, and used for the biomarker metrics')
    region_dict = traits.Dict(key_trait=traits.Str,
                              value_trait=traits.Int,
                              desc=('Dictionnary with keys = brain region names, '
                                    'and values = brain region labels (i.e. the corresponding value in brain_seg)'))
    region_dict_json = traits.File(exists=True,
                                   desc='json file where region_dict will be saved for future reference')


class Brain_Seg_for_biomarker(BaseInterface):
    """
    Transform our parcellation (derived from Synthseg) to one that is customized for the studied biomarker metrics.
    'mars' refers to the Microbleed Anatomical Rating Scale (MARS)
    """
    input_spec = Brain_Seg_for_biomarker_InputSpec
    output_spec = Brain_Seg_for_biomarker_OutputSpec

    def _run_interface(self, runtime):
        seg_im = nib.load(self.inputs.brain_seg)
        seg_vol = seg_im.get_fdata().astype(int)

        seg_scheme = self.inputs.custom_parc
        if seg_scheme == 'pvs':
            custom_seg_vol, seg_dict = seg_for_pvs(seg_vol)
        elif seg_scheme == 'wmh':
            custom_seg_vol, seg_dict = seg_for_wmh(seg_vol)
        elif seg_scheme == 'mars':
            custom_seg_vol, seg_dict = seg_from_mars(seg_vol)
        else:
            raise ValueError(f'Unrecognised segmentation scheme: Expected "pvs", "wmh", or "mars" but bot "{seg_scheme}"')

        region_dict = {'Whole brain': -1, **seg_dict}
        custom_seg_im = nib.Nifti1Image(custom_seg_vol, affine=seg_im.affine)
        nib.save(custom_seg_im, self.inputs.out_file)

        setattr(self, 'region_dict', region_dict)
        json_name = f'{seg_scheme}_region_dict.json'
        setattr(self, 'json_name', json_name)
        with open(json_name, 'w') as jsonfile:
            json.dump(region_dict, jsonfile, indent=4)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['brain_seg'] = op.abspath(self.inputs.out_file)
        outputs['region_dict'] = getattr(self, 'region_dict')
        outputs['region_dict_json'] = op.abspath(getattr(self, 'json_name'))

        return outputs


# %% Old interfaces

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
        return runtime

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["metrics_bg_pvs"] = getattr(self, "metrics_bg_pvs")

        return outputs
