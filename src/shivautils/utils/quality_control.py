
import os.path as op
import warnings
import math
import nibabel as nib
import numpy as np
import skimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import (
    gaussian_filter1d,
    distance_transform_edt
)
from shivautils.utils.stats import get_mode


def overlay_brainmask(img_ref, brainmask):
    """Overlay brainmask on t1 images for 40 slices

    Args:
        img_ref (nib.Nifti1Image): t1 nifti images overlay with brainmask
        brainmask (nib.Nifti1Image): brainmask file to overlay with t1 nifti file

    Returns:
        png: png file with 40 slices of overlay brainmask on t1 nifti file
    """

    ref_im = nib.load(img_ref)
    ref_vol = ref_im.get_fdata()
    brainmask_im = nib.load(brainmask)
    brainmask_vol = brainmask_im.get_fdata()

    nb_of_slices = 6  # define the number of columns in the image

    # creating cmap to overlay reference image and given image
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    # Getting slices in each dim
    crop_ratio = 0.1  # 10% of the slice
    border_crop_X = math.ceil(crop_ratio * ref_vol.shape[0])
    croped_shape_X = (1-2*crop_ratio)*ref_vol.shape[0]  # can be non-integer at this step, not important
    slices_ind_X = np.arange(border_crop_X, ref_vol.shape[0]-border_crop_X, croped_shape_X/nb_of_slices).astype(int)
    slices_X = [(ref_vol[ind, :, :], brainmask_vol[ind, :, :], ind) for ind in slices_ind_X]

    border_crop_Y = math.ceil(crop_ratio * ref_vol.shape[1])
    croped_shape_Y = (1-2*crop_ratio)*ref_vol.shape[1]
    slices_ind_Y = np.arange(border_crop_Y, ref_vol.shape[1]-border_crop_Y, croped_shape_Y/nb_of_slices).astype(int)
    slices_Y = [(ref_vol[:, ind, :], brainmask_vol[:, ind, :], ind) for ind in slices_ind_Y]

    border_crop_Z = math.ceil(crop_ratio * ref_vol.shape[2])
    croped_shape_Z = (1-2*crop_ratio)*ref_vol.shape[2]
    slices_ind_Z = np.arange(border_crop_Z, ref_vol.shape[2]-border_crop_Z, croped_shape_Z/nb_of_slices).astype(int)
    slices_Z = [(ref_vol[:, :, ind], brainmask_vol[:, :, ind], ind) for ind in slices_ind_Z]

    # Affichage dans les trois axes
    fig, ax = plt.subplots(3, nb_of_slices, figsize=(nb_of_slices, 3), dpi=300)
    fig.patch.set_facecolor('k')
    alpha = 0.5

    for col in range(nb_of_slices):
        # X
        ax[0, col].imshow(slices_X[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[0, col].imshow(slices_X[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_X = ax[0, col].set_xlabel(f'k = {slices_X[col][2]}')
        label_X.set_fontsize(5)  # Définir la taille de la police du label
        label_X.set_color('white')
        ax[0, col].get_xaxis().set_ticks([])
        ax[0, col].get_yaxis().set_ticks([])
        # Y
        ax[1, col].imshow(slices_Y[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[1, col].imshow(slices_Y[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_Y = ax[1, col].set_xlabel(f'k = {slices_Y[col][2]}')
        label_Y.set_fontsize(5)  # Définir la taille de la police du label
        label_Y.set_color('white')
        ax[1, col].get_xaxis().set_ticks([])
        ax[1, col].get_yaxis().set_ticks([])
        # Z
        ax[2, col].imshow(slices_Z[col][0].T,
                          origin='lower',
                          cmap='gray')
        ax[2, col].imshow(slices_Z[col][1].T,
                          origin='lower',
                          alpha=alpha,
                          cmap=my_cmap)
        label_Z = ax[2, col].set_xlabel(f'k = {slices_Z[col][2]}')
        label_Z.set_fontsize(5)  # Définir la taille de la police du label
        label_Z.set_color('white')
        ax[2, col].get_xaxis().set_ticks([])
        ax[2, col].get_yaxis().set_ticks([])

    fig.tight_layout()

    plt.savefig('qc_overlay_brainmask_T1.png')
    overlayed_brainmask = op.abspath('qc_overlay_brainmask_T1.png')
    return overlayed_brainmask


def bounding_crop(brain_img: str,
                  brainmask: str,
                  bbox1: tuple,
                  bbox2: tuple,
                  slice_coord: tuple):
    """Overlay of brainmask over nifti images

    Args:
        brain_img (path): Reference nifti images to overlay with brainmask and crop box
        brainmask (path): nifti brainmask file
        bbox1 (tuple): first coordoninates point of cropping box
        bbox2 (tuple): second coordonaites point of cropping box
        slice_coord (tuple): Coordinate of the 3 slices to show (for x,y,z respectively)

    Returns:
        crop_brain_img: svg file of the crop-box and brainmask overlayed on the brain
    """

    ref_vol = nib.load(brain_img).get_fdata()
    brainmask = nib.load(brainmask).get_fdata().astype(bool)
    original_dims = ref_vol.shape
    # Coordonnées de la boîte de recadrage
    x_start, y_start, z_start = bbox1
    x_end, y_end, z_end = bbox2

    # Création d'une matrice tridimensionnelle pour les contours de la boîte de recadrage
    contour = np.zeros(original_dims, dtype=np.uint8)
    contour[x_start:x_end, y_start:y_end, z_start:z_end] = original_dims[0] - 1  # Why?!

    contour[slice_coord] = 10

    # Affichage dans les trois axes
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # fig.patch.set_facecolor('k')
    alpha = 0.7

    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    # Axe x
    ax[0].imshow(np.rot90(contour[slice_coord[0], :, :]), alpha=0.8)
    ax[0].imshow(np.rot90(ref_vol[slice_coord[0], :, :]), cmap='gray', alpha=0.85)
    ax[0].imshow(np.rot90(brainmask[slice_coord[0], :, :]), cmap=my_cmap, alpha=alpha)
    ax[0].axis('off')
    ax[0].set_title('Sagittal Axis')

    # Axe y
    ax[1].imshow(np.rot90(contour[:, slice_coord[1], :]), alpha=0.8)
    ax[1].imshow(np.rot90(ref_vol[:, slice_coord[1], :]), cmap='gray', alpha=0.85)
    ax[1].imshow(np.rot90(brainmask[:, slice_coord[1], :]), cmap=my_cmap, alpha=alpha)
    ax[1].axis('off')
    ax[1].set_title('Coronal Axis')

    # Axe z
    ax[2].imshow(np.rot90(contour[:, :, slice_coord[2]]), alpha=0.8)
    ax[2].imshow(np.rot90(ref_vol[:, :, slice_coord[2]]), cmap='gray', alpha=0.85)
    ax[2].imshow(np.rot90(brainmask[:, :, slice_coord[2]]), cmap=my_cmap, alpha=alpha)
    ax[2].axis('off')
    ax[2].set_title('Axial Axis')

    fig.tight_layout()
    plt.savefig('bounding_crop.svg', bbox_inches='tight', pad_inches=0.1)
    crop_brain_img = op.abspath('bounding_crop.svg')
    return crop_brain_img


def save_histogram(img_normalized, bins=64):
    """Save histogram of intensity normalization voxels on the disk

    Args:
        img_normalized (nib.Nifti1Image): intensity normalize images to obtain histogram
        bins (int, optional): Number of element on histogram file. Defaults to 64.

    Returns:
        path: file path of histogram intensity voxels
    """

    if not isinstance(img_normalized, nib.Nifti1Image):
        if op.isfile(img_normalized):
            img_normalized = nib.load(img_normalized)
        else:
            raise ValueError('The input should have been a Nifti1Image or a nifti file, but was neither.')
    array = img_normalized.get_fdata()
    x = array.flatten()
    _, ax = plt.subplots()
    (hist_val, hist_edges, _) = plt.hist(x, bins=bins)
    mode = get_mode(hist_val, hist_edges, bins)  # Peak intensity away from min and max intensities
    ax.set_yscale('log')
    ax.set_title("Histogram of intensities voxel values")
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("Number of Voxels")

    histogram = 'hist.svg'
    plt.savefig('hist.svg')
    return op.abspath(histogram), mode


def create_edges(path_image, path_ref_image, path_brainmask, nb_of_slices=12, slice_orient='axial'):
    """Create image edges based on: https://github.com/neurolabusc/PyDog/blob/main/dog.py to check the coregistration.
    Copyright (c) 2021, Chris Rorden
    All rights reserved.

    INPUT: You must provide the create_edges function with:

    - path_image: Path to the image that was coregistered (the image must be cropped and the intensity normalized)
    - path_ref_img: Path to the reference image on which the images have been coregistered (the reference image must be cropped and its intensity normalized)
    - path_brainmask: Cropped brain mask (in the same space as reference image)
    - nb_of_slices: number of slices shown (default 12) for PNG output
    - slice_orient: orientation of the slices (default is 'axial'). Can be 'axial', 'sagittal', or 'coronal'

    OUTPUT: the create_edges function will produce one PNG per subject with the reference image in the background 
    and the edges of the given image on top.

    EXAMPLE: If you want to check the FLAIR coregistration on T1w, your image will be cropped and the intensity normalized FLAIR image, 
    reference image - cropped and intensity normalized T1w image
    Brain mask in T1w space and cropped.

    @autor: iastafeva
    @date: 24-04-2023
    """

    # Creating edges using: https://github.com/neurolabusc/PyDog/blob/main/dog.py

    def dehaze(img, brainmask, level, verbose=0):
        """use Otsu to threshold https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
            n.b. threshold used to mask image: dark values are zeroed, but result is NOT binary
        Parameters
        ----------
        img : Niimg-like object
            Image(s) to run DoG on (see :ref:`extracting_data`
            for a detailed description of the valid input types).
        level : int
            value 1..5 with larger values preserving more bright voxels
            dark_classes/total_classes
                1: 3/4
                2: 2/3
                3: 1/2
                4: 1/3
                5: 1/4
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """
        fdata = img.get_fdata()
        brainmask = brainmask.get_fdata()
        fdata = fdata*brainmask
        level = max(1, min(5, level))
        n_classes = abs(3 - level) + 2
        dark_classes = 4 - level
        dark_classes = max(1, min(3, dark_classes))
        thresholds = skimage.filters.threshold_multiotsu(fdata, n_classes)
        thresh = thresholds[dark_classes - 1]

        fdata[fdata < thresh] = 0
        return fdata

    def _fast_smooth_array(arr):
        """Simple smoothing which is less computationally expensive than
        applying a Gaussian filter.
        Only the first three dimensions of the array will be smoothed. The
        filter uses [0.2, 1, 0.2] weights in each direction and use a
        normalisation to preserve the local average value.
        Parameters
        ----------
        arr : :class:`numpy.ndarray`
            4D array, with image number as last dimension. 3D arrays are
            also accepted.
        Returns
        -------
        :class:`numpy.ndarray`
            Smoothed array.
        Notes
        -----
        Rather than calling this function directly, users are encouraged
        to call the high-level function :func:`smooth_img` with
        `fwhm='fast'`.
        """
        neighbor_weight = 0.2
        # 6 neighbors in 3D if not on an edge
        nb_neighbors = 6
        # This scale ensures that a uniform array stays uniform
        # except on the array edges
        scale = 1 + nb_neighbors * neighbor_weight

        # Need to copy because the smoothing is done in multiple statements
        # and there does not seem to be an easy way to do it in place
        smoothed_arr = arr.copy()
        weighted_arr = neighbor_weight * arr

        smoothed_arr[:-1] += weighted_arr[1:]
        smoothed_arr[1:] += weighted_arr[:-1]
        smoothed_arr[:, :-1] += weighted_arr[:, 1:]
        smoothed_arr[:, 1:] += weighted_arr[:, :-1]
        smoothed_arr[:, :, :-1] += weighted_arr[:, :, 1:]
        smoothed_arr[:, :, 1:] += weighted_arr[:, :, :-1]
        smoothed_arr /= scale

        return smoothed_arr

    def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
        """Smooth images by applying a Gaussian filter.
        Apply a Gaussian filter along the three first dimensions of `arr`.
        Parameters
        ----------
        arr : :class:`numpy.ndarray`
            4D array, with image number as last dimension. 3D arrays are also
            accepted.
        affine : :class:`numpy.ndarray`
            (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
            are also accepted (only these coefficients are used).
            If `fwhm='fast'`, the affine is not used and can be None.
        fwhm : scalar, :class:`numpy.ndarray`/:obj:`tuple`/:obj:`list`, 'fast' or None, optional
            Smoothing strength, as a full-width at half maximum, in millimeters.
            If a nonzero scalar is given, width is identical in all 3 directions.
            A :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` must have 3 elements,
            giving the FWHM along each axis.
            If any of the elements is zero or None, smoothing is not performed
            along that axis.
            If  `fwhm='fast'`, a fast smoothing will be performed with a filter
            [0.2, 1, 0.2] in each direction and a normalisation
            to preserve the local average value.
            If fwhm is None, no filtering is performed (useful when just removal
            of non-finite values is needed).
        ensure_finite : :obj:`bool`, optional
            If True, replace every non-finite values (like NaNs) by zero before
            filtering. Default=True.
        copy : :obj:`bool`, optional
            If True, input array is not modified. True by default: the filtering
            is not performed in-place. Default=True.
        Returns
        -------
        :class:`numpy.ndarray`
            Filtered `arr`.
        Notes
        -----
        This function is most efficient with arr in C order.
        """
        # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
        # See issue #1537
        if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
            warnings.warn("The parameter 'fwhm' for smoothing is specified "
                          "as {0}. Setting it to None "
                          "(no smoothing will be performed)"
                          .format(fwhm))
            fwhm = None
        if arr.dtype.kind == 'i':
            if arr.dtype == np.int64:
                arr = arr.astype(np.float64)
            else:
                arr = arr.astype(np.float32)  # We don't need crazy precision.
        if copy:
            arr = arr.copy()
        if ensure_finite:
            # SPM tends to put NaNs in the data outside the brain
            arr[np.logical_not(np.isfinite(arr))] = 0
        if isinstance(fwhm, str) and (fwhm == 'fast'):
            arr = _fast_smooth_array(arr)
        elif fwhm is not None:
            fwhm = np.asarray([fwhm]).ravel()
            fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])
            affine = affine[:3, :3]  # Keep only the scale part.

            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            # fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
            # n.b. FSL specifies blur in sigma, SPM in FWHM
            # FWHM = sigma*sqrt(8*ln(2)) = sigma*2.3548.
            # convert fwhm to sd in voxels see https://github.com/0todd0000/spm1d
            fwhmvox = fwhm / vox_size
            sd = fwhmvox / math.sqrt(8 * math.log(2))
            for n, s in enumerate(sd):
                if s > 0.0:
                    gaussian_filter1d(arr, s, output=arr, axis=n)
        return arr

    def binary_zero_crossing(fdata):
        """binarize (negative voxels are zero)
        Parameters
        ----------
        fdata : numpy.memmap from Niimg-like object
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """
        edge = np.where(fdata > 0.0, 1, 0)
        edge = distance_transform_edt(edge)
        edge[edge > 1] = 0
        edge[edge > 0] = 1
        edge = edge.astype('uint8')
        return edge

    def difference_of_gaussian(fdata, affine, fwhmNarrow, verbose=0):
        """Apply Difference of Gaussian (DoG) filter.
        https://en.wikipedia.org/wiki/Difference_of_Gaussians
        https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm
        D. Marr and E. C. Hildreth. Theory of edge detection. Proceedings of the Royal Society, London B, 207:187-217, 1980
        Parameters
        ----------
        fdata : numpy.memmap from Niimg-like object
        affine : :class:`numpy.ndarray`
            (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
            are also accepted (only these coefficients are used).
        fwhmNarrow : int
            Narrow kernel width, in millimeters. Is an arbitrary ratio of wide to narrow kernel.
                human cortex about 2.5mm thick
                Large values yield smoother results
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """

        # Hardcode 1.6 as ratio of wide versus narrow FWHM
        # Marr and Hildreth (1980) suggest narrow to wide ratio of 1.6
        # Wilson and Giese (1977) suggest narrow to wide ratio of 1.5
        fwhmWide = fwhmNarrow * 1.6
        # optimization: we will use the narrow Gaussian as the input to the wide filter
        fwhmWide = math.sqrt((fwhmWide*fwhmWide) - (fwhmNarrow*fwhmNarrow))

        imgNarrow = _smooth_array(fdata, affine, fwhmNarrow)
        imgWide = _smooth_array(imgNarrow, affine, fwhmWide)
        img = imgNarrow - imgWide
        img = binary_zero_crossing(img)
        return img

    def dog_img(img, brainmask, fwhm, verbose=0):
        """Find edges of a NIfTI image using the Difference of Gaussian (DoG).
        Parameters
        ----------
        img : Niimg-like object
            Image(s) to run DoG on (see :ref:`extracting_data`
            for a detailed description of the valid input types).
        fwhm : int
            Edge detection strength, as a full-width at half maximum, in millimeters.
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """

        dog_fdata = dehaze(img, brainmask, 3, verbose)

        dog = difference_of_gaussian(dog_fdata, img.affine, fwhm, verbose)

        out_img = nib.Nifti1Image(dog, img.affine, img.header)
        # update header
        out_img.header.set_data_dtype(np.uint8)
        out_img.header['intent_code'] = 0
        out_img.header['scl_slope'] = 1.0
        out_img.header['scl_inter'] = 0.0
        out_img.header['cal_max'] = 0.0
        out_img.header['cal_min'] = 0.0

        return out_img

    # load data
    dataNii = nib.load(path_ref_image)
    ref_vol = dataNii.get_fdata(dtype=np.float32)
    contour_img = nib.load(path_image)
    brainmask = nib.load(path_brainmask)

    # get edges in a NIfTI image
    dog_imported_img = dog_img(contour_img, brainmask, fwhm=3, verbose=1)
    contour_vol = dog_imported_img.get_fdata(dtype=np.float32)

    # creating cmap to overlay reference image and given image
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    slice_orient_dict = {'sagittal': 0,
                         'coronal': 1,
                         'axial': 2}
    if slice_orient not in slice_orient_dict.keys():
        raise ValueError(f'"slice_orient" must be one of the following: {list(slice_orient_dict.keys())}, but "{slice_orient}" was given.')
    slice_orient_dim = slice_orient_dict[slice_orient]
    slice_shape = contour_vol.shape[slice_orient_dim]

    # Define the amount of the image to ignore near the border when slicing
    crop_ratio = 0.1
    border_crop = math.ceil(crop_ratio * slice_shape)  # 10% of the slice
    croped_shape = (1-2*crop_ratio)*slice_shape  # can be non-integer at this step, not important
    slices_ind = np.arange(border_crop, slice_shape-border_crop, croped_shape/nb_of_slices).astype(int)

    # Quick and dirty (and readable) slicing depending on the sliced dimension
    # "slices" is a list of tuple, each tuple containing: ("ref slice", "contour slice", "corresponding index")
    if slice_orient_dim == 0:
        slices = [(ref_vol[ind, :, :], contour_vol[ind, :, :], ind) for ind in slices_ind]
    if slice_orient_dim == 1:
        slices = [(ref_vol[:, ind, :], contour_vol[:, ind, :], ind) for ind in slices_ind]
    if slice_orient_dim == 2:
        slices = [(ref_vol[:, :, ind], contour_vol[:, :, ind], ind) for ind in slices_ind]

    # Prep the figure
    max_col_nb = 6  # Max number of slices per row
    row_nb = math.ceil(nb_of_slices/max_col_nb)
    if nb_of_slices < max_col_nb:
        col_nb = nb_of_slices
    else:
        col_nb = max_col_nb
    figure, axes = plt.subplots(row_nb, col_nb, figsize=(max_col_nb, row_nb), dpi=300)
    figure.patch.set_facecolor('k')

    # Iterate on both figure axes and slices
    for (slice_ref_vol, slice_contour_vol, ind), ax in zip(slices, axes.reshape(-1)[:len(slices)]):
        ax.imshow(
            slice_ref_vol.T,  # swap X and Y in the figure with .T to better display
            origin='lower',
            cmap='gray')
        ax.imshow(
            slice_contour_vol.T,
            origin='lower',
            cmap=my_cmap)

        label = ax.set_xlabel(f'k = {ind}')
        label.set_fontsize(5)
        label.set_color('white')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    figure.tight_layout()

    QC_coreg = 'QC_coreg_edges.png'
    plt.savefig('QC_coreg_edges.png')
    return (op.abspath(QC_coreg))
