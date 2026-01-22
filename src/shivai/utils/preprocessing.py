"""All module needed to implementaton of preprocessing image"""
from typing import Tuple
import numpy as np
import random
# import typing
from copy import deepcopy
from skimage.measure import label
from skimage.morphology import opening, binary_erosion, binary_dilation, ball, cube
from itertools import permutations, product

import nibabel.processing as nip
import nibabel as nib
from scipy import ndimage
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from shivai.utils.misc import histogram, fisin


# def roll_binary_dilation(vol: np.ndarray[typing.Any, bool], kernel: np.ndarray[typing.Any, bool] = None, outtype=None):
#     """Binary dilation using the np.roll function to go super fast.
#     However, there is no checks for if the dilation goes over the edge (it will loop around the axis).

#     Args:
#         vol (np.ndarray[typing.Any, bool]): Input array to be dilated
#         kernel (np.ndarray[typing.Any, bool]): Pseudo-morphology kernel. Must have the same number of dim than vol.
#             By default, will use a fully connected kernel (square for 2D, cube for 3D) of width 3.
#         outtype : casting type for the output array. By default, will be the same as the input vol.

#     Returns:

#     """
#     outtype = vol.dtype if outtype is None else outtype
#     kernel = np.ones(tuple(3 for i in range(len(vol.shape)))) if kernel is None else kernel
#     if any([dim % 2 == 0 for dim in kernel.shape]):
#         raise ValueError('Kernel must have an odd number of voxels in each dim (to have a center voxel)')
#     if len(vol.shape) != len(kernel.shape):
#         raise ValueError('input volume and kernel must have the same number of dimensions')
#     if not vol.dtype == bool:
#         vol = vol.astype(bool)
#     res = vol.copy()
#     center_vox = tuple(i//2 for i in kernel.shape)
#     shifts = [tuple(s) for s in np.argwhere(kernel) - center_vox]
#     for shift in shifts:
#         res += np.roll(vol, shift, tuple(range(len(vol.shape))))
#     return res.astype(outtype)


# def roll_binary_erosion(vol: np.ndarray[typing.Any, bool], kernel: np.ndarray[typing.Any, bool] = None, outtype=None):
#     """See roll_binary_dilation
#     """
#     outtype = vol.dtype if outtype is None else outtype
#     vol = vol.astype(bool)
#     inv_res = roll_binary_dilation(~vol, kernel)
#     res = ~inv_res
#     return res.astype(outtype)


# def roll_binary_opening(vol: np.ndarray[typing.Any, bool], kernel: np.ndarray[typing.Any, bool] = None, outtype=None, iterations=1, rep=1):
#     """
#     iter : number of time running the operation
#     rep : number of erosions to perform before dilations
#     """
#     res = vol.copy()
#     for i in range(iterations):
#         for r in range(rep):
#             res = roll_binary_erosion(res, kernel)
#         for r in range(rep):
#             res = roll_binary_dilation(res, kernel, outtype)
#     return res


# def roll_binary_closing(vol: np.ndarray[typing.Any, bool], kernel: np.ndarray[typing.Any, bool] = None, outtype=None, iterations=1, rep=1):
#     """
#     iter : number of time running the operation
#     rep : number of dilations to perform before erosions
#     """
#     res = vol.copy()
#     for i in range(iterations):
#         for r in range(rep):
#             res = roll_binary_erosion(res, kernel)
#         for r in range(rep):
#             res = roll_binary_dilation(res, kernel, outtype)
#     return res


def create_anisotropic_ellipsoid(radius_voxels, max_radius=None):
    """Create an ellipsoid footprint with different radii per dimension.

    Args:
        radius_voxels (tuple): Radius in voxels for each dimension (x, y, z)
        max_radius (int): Optional maximum radius to cap any dimension

    Returns:
        np.ndarray: Binary 3D ellipsoid footprint
    """
    radii = np.array(radius_voxels, dtype=float)
    if max_radius is not None:
        radii = np.minimum(radii, max_radius)

    # Ensure at least radius of 1 in each dimension
    radii = np.maximum(radii, 1)
    radii = radii.astype(int)

    # Create meshgrid
    ranges = [np.arange(-r, r + 1) for r in radii]
    grids = np.meshgrid(*ranges, indexing='ij')

    # Create ellipsoid: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 <= 1
    ellipsoid = sum((g.astype(float) / r)**2 for g, r in zip(grids, radii)) <= 1

    return ellipsoid.astype(np.uint8)


def normalization(img: nib.Nifti1Image,
                  percentile: int,
                  brain_mask: nib.Nifti1Image = None,
                  inverse: bool = False) -> nib.Nifti1Image:
    """We remove values above the 99th percentile to avoid hot spots,
       set values below 0 to 0, set values above 1.3 to 1.3 and normalize
       the data between 0 and 1.

       Args:
           img (nib.Nifti1Image): image to process
           percentile (int): value to threshold above this percentile
           brain_mask (nib.Nifti1Image): Brain mask
           inverse (bool): Wether to "inverse" the resulting image (1-val for all voxels in the brain). Requires a brain mask.

       Returns:
        nib.Nifti1Image: normalized image
    """
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")

    if inverse and not brain_mask:
        raise ValueError('No brain mask was provided while the "inverse" option was selected.')

    # We suppress values above the 99th percentile to avoid hot spots
    array = np.nan_to_num(img.get_fdata())
    print(np.max(array))
    array[array < 0] = 0
    # calculate percentile
    if not brain_mask:
        value_percentile = np.percentile(array, percentile)
    else:
        brain_mask_array = np.squeeze(brain_mask.get_fdata())
        value_percentile = np.percentile(array[np.squeeze(brain_mask_array) != 0], percentile)

    # scaling the array with the percentile value
    array /= value_percentile

    # anything values less than 0 are set to 0 and we set to 1.3 the values greater than 1.3
    array[array > 1.3] = 1.3

    # We normalize the data between 0 and 1
    array_normalized = array / 1.3

    # Inversion (usually for T2w) if needed
    if inverse:
        array_normalized[brain_mask_array.astype(bool)] = 1 - array_normalized[brain_mask_array.astype(bool)]

    # Normalization information
    report, mode = histogram(array_normalized, percentile, bins=64)

    img_nifti_normalized = nip.Nifti1Image(array_normalized.astype('f'), img.affine)

    return img_nifti_normalized, report, mode


def threshold(img: nib.Nifti1Image,
              thr: float = 0.4,
              sign: str = '+',
              binarize: bool = False,
              open_iter: int = 0,
              clusterCheck: str = 'size',
              minVol: int = 0) -> nib.Nifti1Image:
    """Create a brain_mask by putting all the values below the threshold.
       to 0. Offer filtering options if multiple clusters are expected.

       Args:
            img (nib.Nifti1Image): image to process
            thr (float): appropriate value to select mostly brain tissue
                (white matter) and remove background
            sign (str): '+' zero anything below, '-' zero anythin above threshold
            binarize (bool): make a binary mask
            open_iter (int): do a morphological opening using the given int for the radius
                of the ball used as footprint. If 0 is given, skip this step.
            clusterCheck (str): Can be 'top', 'size', or 'all'. Labels the clusters in the mask,
                then keep the one highest in the brain if 'top' was selected, or keep
                the biggest cluster if 'size' was selected (but will raise an error if
                it's not the one at the top). 'all' doesn't do any check.
            minVol (int): Removes clusters with volume under the specified value. Should
                be used if clusterCheck = 'top'

       Returns:
           nib.Nifti1Image: preprocessed image
    """
    import numpy as np
    import nibabel as nib

    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")
    if not isinstance(thr, float):
        raise TypeError("'thr' must be a float")
    if not clusterCheck in ('top', 'size', 'all'):
        raise ValueError(
            f"Input for clusterCheck should be 'top', 'size' or 'all' but {clusterCheck} was given.")

    array = img.get_fdata().squeeze()
    if sign == '+' and not binarize:
        array[array < thr] = 0
    elif sign == '+' and binarize:
        array = array > thr
        array = array.astype(np.uint8)
    elif sign == '-' and not binarize:
        array[array > thr] = 0
    elif sign == '-' and binarize:
        array = array < thr
        array = array.astype(np.uint8)
    else:
        raise ValueError(f'Unsupported sign argument value {sign} (+ or -)...')

    voxel_size = img.header['pixdim'][1:4]
    min_voxel_size = voxel_size.min()
    if open_iter:
        # Calculate adaptive radius based on voxel spacing
        # Scale the radius by voxel dimensions to maintain similar physical size
        # Normalize by the smallest voxel dimension
        radius_voxels = tuple((open_iter * min_voxel_size / voxel_size).astype(int))

        # For very anisotropic images, cap the maximum radius to avoid issues
        max_radius = max(array.shape) // 4  # Don't let exceed 1/4 of any dimension
        radius_voxels = tuple(np.minimum(radius_voxels, max_radius))
        # Ensure at least 1 iteration per dimension if open_iter > 0
        radius_voxels = tuple(np.maximum(radius_voxels, 1))

        ori_array = array.copy()

        if binarize:
            # Apply dimension-adaptive erosion/dilation using 1D footprints
            # This is faster than using a 3D footprint and preserves the original speed benefit
            for dim in range(3):
                # Create 1D footprint along current dimension
                footprint_1d = np.zeros((3, 3, 3), dtype=bool)
                footprint_1d[1, 1, 1] = True
                # Add the line along the dimension axis
                footprint_1d[tuple([1 if i != dim else slice(0, 3) for i in range(3)])] = True

                # Apply erosion for this dimension
                for _ in range(radius_voxels[dim]):
                    array = binary_erosion(array, footprint=footprint_1d)

            # Apply dilation (reverse order, same dimensions)
            for dim in range(3):
                footprint_1d = np.zeros((3, 3, 3), dtype=bool)
                footprint_1d[1, 1, 1] = True
                footprint_1d[tuple([1 if i != dim else slice(0, 3) for i in range(3)])] = True

                for _ in range(radius_voxels[dim]):
                    array = binary_dilation(array, footprint=footprint_1d)

            array = array.astype(np.uint8)
        else:
            # Use anisotropic ellipsoid footprint for non-binary case
            footprint = create_anisotropic_ellipsoid(radius_voxels)
            array = opening(array, footprint=footprint)
    if clusterCheck in ('top', 'size') or minVol:
        labeled_clusters = label(array)
        clst,  clst_cnt = np.unique(
            labeled_clusters[labeled_clusters > 0],
            return_counts=True)
        # Sorting the clusters by size
        sort_ind = np.argsort(clst_cnt)[::-1]
        clst,  clst_cnt = clst[sort_ind],  clst_cnt[sort_ind]
        if clst.size > 1:
            if minVol:
                clst = clst[clst_cnt > minVol]
            if clusterCheck in ('top', 'size'):
                maxInd = []
                for c in clst:
                    zmax = np.where(labeled_clusters == c)[2].max()
                    maxInd.append(zmax)
                topClst = clst[np.argmax(maxInd)]  # Highest (z-axis) cluster
                if clusterCheck == 'top':
                    cluster_mask = (labeled_clusters == topClst)
                else:
                    if not topClst == clst[0]:
                        raise ValueError(
                            'The biggest cluster in the mask is not the one at '
                            'the top of the brain. Check the data for that participant.')
                    cluster_mask = (labeled_clusters == clst[0])
            else:  # only minVol filtering
                cluster_mask = fisin(labeled_clusters, clst)
            # Dilating the main cluster mask to make sure to keep the full brain
            dil_size = 5  # in mm
            radius_voxels = (dil_size / voxel_size).astype(int)
            radius_voxels = tuple(np.maximum(radius_voxels, 1))
            if binarize:
                for dim in range(3):
                    # Create 1D footprint along current dimension
                    footprint_1d = np.zeros((3, 3, 3), dtype=bool)
                    footprint_1d[1, 1, 1] = True
                    # Add the line along the dimension axis
                    footprint_1d[tuple([1 if i != dim else slice(0, 3) for i in range(3)])] = True

                    # Apply dilation for this dimension
                    for _ in range(radius_voxels[dim]):
                        cluster_mask = binary_dilation(cluster_mask, footprint=footprint_1d)
                cluster_mask = cluster_mask.astype(np.uint8)
            else:
                footprint = create_anisotropic_ellipsoid(radius_voxels)
                cluster_mask = binary_dilation(cluster_mask, footprint=footprint)
            array *= cluster_mask

    thresholded = nip.Nifti1Image(array.astype('f'), img.affine)

    return thresholded


def crop(roi_mask: nib.Nifti1Image,
         apply_to: nib.Nifti1Image,
         dimensions: Tuple[int, int, int],
         cdg_ijk: np.ndarray = None,
         default: str = 'ijk',
         safety_marger: int = 5
         ) -> Tuple[nib.Nifti1Image,
                    Tuple[int, int, int],
                    Tuple[int, int, int],
                    Tuple[int, int, int]]:
    """Adjust the real-world referential and crop image.

    If a mask is supplied, the procedure uses the center of mass of the mask as a crop center.

    If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine
    referential coordiantes origin. If set to 'ijk', the middle of the image is used.

    Args:
        roi_mask (nib.Nifti1Image): mask used to define the center
                                   of the bounding box (center of gravity of mask)
        apply_to (nib.Nifti1Image): image to crop
        dimensions (Tuple[int, int, int], optional): volume dimensions.
                                                     Defaults to (256 , 256 , 256).
        cdg_ijk: arbitrary crop center ijk coordinates
        safety_marger (int): added deviation from the top of the image if the brain mask is offset

    Returns:
        nib.Nifti1Image: preprocessed image
        crop center ijk coordiantes
        bouding box top left ijk coordiantes
        bounding box bottom right coordinates
    """
    start_ornt = io_orientation(apply_to.affine)
    end_ornt = axcodes2ornt("RAS")
    transform = ornt_transform(start_ornt, end_ornt)

    # Reorient first to ensure shape matches expectations
    apply_to = apply_to.as_reoriented(transform)
    if not isinstance(apply_to, nib.nifti1.Nifti1Image):
        raise TypeError("apply_to: only Nifti images are supported")

    if roi_mask and not isinstance(roi_mask, nib.nifti1.Nifti1Image):
        raise TypeError("roi_mask: only Nifti images are supported")
    elif not roi_mask and not cdg_ijk:
        if default == 'xyz':
            # get cropping center from xyz origin
            cdg_ijk = np.linalg.inv(apply_to.affine) @ np.array([0.0, 0.0, 0.0, 1.0])
            cdg_ijk = np.ceil(cdg_ijk).astype(int)[:3]
        elif default == "ijk":
            cdg_ijk = np.ceil(np.array(apply_to.shape) / 2).astype(int)
        else:
            raise ValueError(f"argument 'default' value {default} not valid")
    elif roi_mask and not cdg_ijk:
        # get CoG from mask as center
        start_ornt = io_orientation(roi_mask.affine)
        end_ornt = axcodes2ornt("RAS")
        transform = ornt_transform(start_ornt, end_ornt)

        # Reorient first to ensure shape matches expectations
        roi_mask = roi_mask.as_reoriented(transform)
        required_ndim = 3
        if roi_mask.ndim != required_ndim:
            raise ValueError("Only 3D images are supported.")
        if len(dimensions) != required_ndim:
            raise ValueError(f"`dimensions` must have {required_ndim} values")
        cdg_ijk = np.ceil(np.array(
            ndimage.center_of_mass(
                roi_mask.get_fdata().astype(bool)))).astype(int)

    # Calculation of the center of gravity of the mask, we round and convert
    # to integers

    # We will center the block on the center of gravity, so we cut the size in
    # 2
    halfs = np.array(dimensions)/2
    # we need integers because it is used for indices of the "array of voxels"
    halfs = halfs.astype(int)

    # the ijk of the lowest voxel in the box
    bbox1 = cdg_ijk - halfs
    # the highest ijk voxel of the bounding box
    bbox2 = halfs + cdg_ijk

    array_out = np.zeros(dimensions, dtype=apply_to.header.get_data_dtype())
    print(f"bbox1: {bbox1}")
    print(f"bbox2: {bbox2}")
    print(f"cdg_ijk: {cdg_ijk}")
    offset_ijk = abs(bbox1) * abs(np.uint8(bbox1 < 0))
    bbox1[bbox1 < 0] = 0
    for i in range(3):
        if bbox2[i] > apply_to.shape[i]:
            bbox2[i] = apply_to.shape[i]
    span = bbox2 - bbox1
    print(f"span: {span}")
    print(f"offset: {offset_ijk}")

    if roi_mask:
        vec = np.sum(roi_mask.get_fdata().astype(bool), axis=(0, 1))
        top_mask_slice_index = np.where(np.squeeze(vec != 0))[0].tolist()[-1]

        if bbox2[2] <= top_mask_slice_index:

            # we are too low, we nned to move the crop box up
            # (because brain mask is wrong and includes stuff in the neck and shoulders)

            delta = top_mask_slice_index - bbox2[2] + safety_marger
            bbox1[2] = bbox1[2] + delta
            bbox2[2] = bbox2[2] + delta
            cdg_ijk[2] = cdg_ijk[2] + delta
            print(f"reworked bbox1: {bbox1}")
            print(f"reworked bbox2: {bbox2}")

    array_out[offset_ijk[0]:offset_ijk[0] + span[0],
              offset_ijk[1]:offset_ijk[1] + span[1],
              offset_ijk[2]:offset_ijk[2] + span[2]] = apply_to.get_fdata()[
        bbox1[0]:bbox2[0],
        bbox1[1]:bbox2[1],
        bbox1[2]:bbox2[2]]

    # We correct the coordinates, so first we have to convert ijk to xyz for
    # half block size and centroid
    cdg_xyz = apply_to.affine @ np.append(cdg_ijk, 1)
    halfs_xyz = apply_to.affine @ np.append(cdg_ijk - bbox1, 1)
    padding_xyz = apply_to.affine @ np.append(tuple(offset_ijk), 1)
    offset_padding = apply_to.affine[:, 3] - padding_xyz
    print(f"padding: {padding_xyz}")
    print(f"padding offset: {offset_padding}")
    print(f"halfs xyz: {halfs_xyz}")

    # on recopie la matrice affine de l'image, on va la modifier
    affine_out = deepcopy(apply_to.affine)

    # We shift the center of the image reference frame because we start from
    # less far (image is smaller)
    # And the center is no longer quite the same

    affine_out[:, 3] = affine_out[:, 3] + (cdg_xyz - halfs_xyz) + offset_padding

    # We write the result image
    cropped = nip.Nifti1Image(array_out.astype('f'), affine_out)

    return cropped, cdg_ijk, bbox1, bbox2


def reverse_crop(original_img: nib.Nifti1Image,
                 apply_to: nib.Nifti1Image,
                 bbox1: Tuple[int, int, int],
                 bbox2: Tuple[int, int, int]):
    """
    Re-modifies the dimensions of the cropped 
    image in the original space
    """
    conform_array = original_img.get_fdata()
    array_apply_to = apply_to.get_fdata()
    reverse_crop_array = np.zeros_like(conform_array)
    reverse_crop_array[bbox1[0]:bbox2[0], bbox1[1]:bbox2[1], bbox1[2]:bbox2[2]] = array_apply_to
    reverse_img = nip.Nifti1Image(reverse_crop_array.astype('f'), original_img.affine)
    return reverse_img


def make_offset(img: nib.Nifti1Image, offset: tuple = False):
    """Make a random offset of array volume in image space

    Args:
        img (nib.Nifti1Image): image to apply offset
        offset (tuple): axis offset apply to nifti image

    """
    if offset == False:
        offset_number = []
        for _ in range(3):
            number = random.randint(-3, 3)
            offset_number.append(number)
    else:
        offset_number = offset

    offset_xyz = img.affine @ np.append(offset_number, 1)
    offset_number = tuple(offset_number)

    array_img = img.get_fdata()
    array_img = np.squeeze(array_img)
    padded_array = np.pad(array_img, pad_width=3, mode='constant')
    shifted_array_img = np.roll(padded_array, offset_number, axis=(0, 1, 2))

    coord_center_img = [int(array_img.shape[0]/2), int(array_img.shape[1]/2), int(array_img.shape[2]/2)]
    new_coord_center_img = [coord_center_img[0] + offset_number[0], coord_center_img[1] + offset_number[1], coord_center_img[2] + offset_number[2]]

    halfs = np.array(array_img.shape)/2
    # we need integers because it is used for indices of the "array of voxels"
    halfs = (int(halfs[0]), int(halfs[1]), int(halfs[2]))
    # the ijk of the lowest voxel in the box
    bbox1 = (new_coord_center_img[0] - halfs[0] + 3, new_coord_center_img[1] - halfs[1] + 3, new_coord_center_img[2] - halfs[2] + 3)
    # the highest ijk voxel of the bounding box
    bbox2 = (new_coord_center_img[0] + halfs[0] + 3, new_coord_center_img[1] + halfs[1] + 3, new_coord_center_img[2] + halfs[2] + 3)

    # cropping image
    new_array = shifted_array_img[bbox1[0]:bbox2[0], bbox1[1]:bbox2[1], bbox1[2]:bbox2[2]]

    affine_out = img.affine
    affine_out[:, 3] = affine_out[:, 3] - (offset_xyz - affine_out[:, 3])

    shifted_img = nib.Nifti1Image(new_array.astype('f'), affine_out, img.header)

    return shifted_img, offset_number


def apply_mask(file_prediction: nib.Nifti1Image,
               brainmask: nib.Nifti1Image):
    """Apply brainmask on prediction file to avoid prediction out of brain image

    Args:
        file_prediction (nib.Nifti1Image): nifti prediction file
        brainmask (nib.Nifti1Image): nifti brainmask file

    Returns:
        masked_prediction_Nifti: file with all prediction out of brain deleted
    """

    array_prediction = file_prediction.get_fdata()
    array_brainmask = brainmask.get_fdata()

    if len(array_prediction.shape) == 4:
        array_prediction = array_prediction.squeeze()
    if len(array_brainmask.shape) == 4:
        array_brainmask = array_brainmask.squeeze()

    masked_prediction = array_prediction * array_brainmask

    masked_prediction_Nifti = nib.Nifti1Image(masked_prediction.astype('f'), file_prediction.affine, file_prediction.header)

    return masked_prediction_Nifti


def seg_cleaner(raw_seg: np.ndarray, max_size: int = 300, ignore_labels: list = []) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Removes isolated islands from segmented labels, replacing the value of the island
    by the value of the neighbor with the most neighboring voxels.

    max_size is the maximum size (in voxel number) under which an "island" is considered as such.
    If it's too big, it propably isn't a faulty segmentation.

    ignore_labels is a list of regions (given by their label number) that will be ignored here
    (typically for CSF)

    Outputs the cleaned segmentation and all the deleted islands (with their original labels)
    '''
    labels = np.unique(raw_seg).tolist()
    try:
        labels.remove(0)
    except ValueError:
        pass

    cleaned_seg = raw_seg.copy()
    removed_islands = np.zeros(raw_seg.shape, dtype='int16')
    # kept_islands = np.zeros(raw_seg.shape, dtype='int16')

    relabeled_seg = label(raw_seg)
    for lab in labels:
        if lab in ignore_labels:
            continue
        relabeled_lab = relabeled_seg * (raw_seg == lab)
        clust_vals, clust_cnt = np.unique(relabeled_lab[relabeled_lab != 0], return_counts=True)
        if len(clust_cnt) > 1:
            main_clust_val = clust_vals[clust_cnt.argmax()]
            isls_clust_vals = [(val, n) for val, n in zip(clust_vals, clust_cnt) if n <= max_size and val != main_clust_val]
            for isl_val, n in isls_clust_vals:
                clust_isle = (relabeled_lab == isl_val)
                if n == 1:  # Manually getting the neighbors is faster here
                    vox_coord_dif = np.concatenate([np.eye(3), -np.eye(3)], axis=0).astype(int)
                    vox_coord = (np.argwhere(clust_isle) + vox_coord_dif).T
                    # removing coords outside of the image
                    for i in range(3):
                        vox_coord = vox_coord[:, (vox_coord[i] >= 0) & (vox_coord[i] < raw_seg.shape[i])]
                    neighbors = tuple(vox_coord)
                else:
                    neighbors = ndimage.binary_dilation(clust_isle) & ~clust_isle
                neighbors_vals, neighbors_cnt = np.unique(raw_seg[neighbors], return_counts=True)
                # if len(neighbors_vals) == 1:  # Island is enclosed in one region
                # cleaned_seg[clust_isle] = neighbors_vals[0]
                # removed_islands[clust_isle] = lab
                # else:
                #     kept_islands[clust_isle] = lab
                cleaned_seg[clust_isle] = neighbors_vals[np.argmax(neighbors_cnt)]
                removed_islands[clust_isle] = lab
    return cleaned_seg, removed_islands  # , kept_islands
