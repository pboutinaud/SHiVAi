"""All module needed to implementaton of preprocessing image"""
from typing import Tuple
import numpy as np
import random
from copy import deepcopy
from skimage.measure import label
from skimage.morphology import opening, binary_opening, ball

import nibabel.processing as nip
import nibabel as nb
from scipy import ndimage
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from shivautils.stats import histogram


def normalization(img: nb.Nifti1Image,
                  percentile: int,
                  brain_mask: nb.Nifti1Image = None) -> nb.Nifti1Image:
    """We remove values above the 99th percentile to avoid hot spots,
       set values below 0 to 0, set values above 1.3 to 1.3 and normalize
       the data between 0 and 1.

       Args:
           img (nb.Nifti1Image): image to process
           percentile (int): value to threshold above this percentile

       Returns:
        nb.Nifti1Image: normalized image
    """
    if not isinstance(img, nb.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")

    # We suppress values above the 99th percentile to avoid hot spots
    array = img.get_fdata()
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

    # Normalization information
    report, mode = histogram(array_normalized, percentile, bins=64)

    img_nifti_normalized = nip.Nifti1Image(array_normalized, img.affine)

    return img_nifti_normalized, report, mode


def threshold(img: nb.Nifti1Image,
              thr: float = 0.4,
              sign: str = '+',
              binarize: bool = False,
              open: int = 0,
              clusterCheck: str = 'size',
              minVol: int = 0) -> nb.Nifti1Image:
    """Create a brain_mask by putting all the values below the threshold.
       to 0. Offer filtering options if multiple clusters are expected.

       Args:
            img (nb.Nifti1Image): image to process
            thr (float): appropriate value to select mostly brain tissue
                (white matter) and remove background
            sign (str): '+' zero anything below, '-' zero anythin above threshold
            binarize (bool): make a binary mask
            open (int): do a morphological opening using the given int for the radius
                of the ball used as footprint. If 0 is given, skip this step.
            clusterCheck (str): Can be 'top', 'size', or 'all'. Labels the clusters in the mask,
                then keep the one highest in the brain if 'top' was selected, or keep
                the biggest cluster if 'size' was selected (but will raise an error if
                it's not the one at the top). 'all' doesn't do any check.
            minVol (int): Removes clusters with volume under the specified value. Should
                be used if clusterCheck = 'top'

       Returns:
           nb.Nifti1Image: preprocessed image
    """
    import numpy as np
    import nibabel as nb

    if not isinstance(img, nb.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")
    if not isinstance(thr, float):
        raise TypeError("'thr' must be a float")
    if not clusterCheck in ('top', 'size', 'all'):
        raise ValueError(
            f"Input for clusterCheck should be 'top', 'size' or 'all' but {clusterCheck} was given.")

    array = img.get_fdata()
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
    
    if open:
        if binarize:
            array = binary_opening(array, footprint=ball(open)).astype(np.uint8)
        else:
            array = opening(array, footprint=ball(open))
    if clusterCheck in ('top', 'size') or minVol:
        labeled_clusters = label(array)
        clst,  clst_cnt = np.unique(  # already sorted by size
            labeled_clusters[labeled_clusters>0],
            return_counts=True)
        if minVol:
            clst = clst[clst_cnt > minVol]
        if clusterCheck in ('top', 'size'):
            maxInd = []
            for c in clst:
                zmax = np.where(labeled_clusters==c)[2].max()
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
        array *= cluster_mask
        
    thresholded = nip.Nifti1Image(array, img.affine)

    return thresholded


def crop(roi_mask: nb.Nifti1Image,
         apply_to: nb.Nifti1Image,
         dimensions: Tuple[int, int, int],
         cdg_ijk: np.ndarray = None,
         default: str = 'ijk',
         safety_marger: int = 5
         ) -> Tuple[nb.Nifti1Image,
                    Tuple[int, int, int],
                    Tuple[int, int, int],
                    Tuple[int, int, int]]:
    """Adjust the real-world referential and crop image.

    If a mask is supplied, the procedure uses the center of mass of the mask as a crop center.

    If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine
    referential coordiantes origin. If set to 'ijk', the middle of the image is used.

    Args:
        roi_mask (nb.Nifti1Image): mask used to define the center
                                   of the bounding box (center of gravity of mask)
        apply_to (nb.Nifti1Image): image to crop
        dimensions (Tuple[int, int, int], optional): volume dimensions.
                                                     Defaults to (256 , 256 , 256).
        cdg_ijk: arbitrary crop center ijk coordinates
        safety_marger (int): added deviation from the top of the image if the brain mask is offset

    Returns:
        nb.Nifti1Image: preprocessed image
        crop center ijk coordiantes
        bouding box top left ijk coordiantes
        bounding box bottom right coordinates
    """
    start_ornt = io_orientation(apply_to.affine)
    end_ornt = axcodes2ornt("RAS")
    transform = ornt_transform(start_ornt, end_ornt)

    # Reorient first to ensure shape matches expectations
    apply_to = apply_to.as_reoriented(transform)
    if not isinstance(apply_to, nb.nifti1.Nifti1Image):
        raise TypeError("apply_to: only Nifti images are supported")

    if roi_mask and not isinstance(roi_mask, nb.nifti1.Nifti1Image):
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
                roi_mask.get_fdata()))).astype(int)

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

    array_out = np.empty(dimensions, dtype=apply_to.header.get_data_dtype())
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

    vec = np.sum(roi_mask.get_fdata(), axis=(0, 1))
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
    cropped = nip.Nifti1Image(array_out, affine_out)

    return cropped, cdg_ijk, bbox1, bbox2


def reverse_crop(original_img: nb.Nifti1Image,
                 apply_to: nb.Nifti1Image,
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
    reverse_img = nip.Nifti1Image(reverse_crop_array, original_img.affine)
    return reverse_img


def make_offset(img: nb.Nifti1Image, offset: tuple = False):
    """Make a random offset of array volume in image space

    Args:
        img (nb.Nifti1Image): image to apply offset
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

    shifted_img = nb.Nifti1Image(new_array, affine_out, img.header)

    return shifted_img, offset_number


def apply_mask(file_prediction: nb.Nifti1Image,
               brainmask: nb.Nifti1Image):
    """Apply brainmask on prediction file to avoid prediction out of brain image

    Args:
        file_prediction (nb.Nifti1Image): nifti prediction file
        brainmask (nb.Nifti1Image): nifti brainmask file

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

    masked_prediction_Nifti = nb.Nifti1Image(masked_prediction, file_prediction.affine, file_prediction.header)

    return masked_prediction_Nifti
