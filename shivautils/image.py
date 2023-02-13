"""All module needed to implementaton of preprocessing image"""
from typing import Tuple
from copy import deepcopy
import nibabel.processing as nip
import nibabel as nb
import numpy as np
from scipy import ndimage

from shivautils.stats import histogram


def normalization(img: nb.Nifti1Image,
                  percentile: int,
                  brain_mask: nb.Nifti1Image=None) -> nb.Nifti1Image:
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
    array_b = array
    array_b[array_b < 0] = 0
    # calculate percentile 
    if not brain_mask:
        value_percentile = np.percentile(array_b, percentile)
    else: 
        brain_mask_array = brain_mask.get_fdata()
        value_percentile = np.percentile(array_b[np.squeeze(brain_mask_array) != 0], percentile)

    array += array.min()
    # scaling the array with the percentile value
    array /= value_percentile
    # anything values less than 0 are set to 0 and we set to 1.3 the values greater than 1.3
    array = np.clip(array, 0, 1.3)

    # We normalize the data between 0 and 1
    array_normalized = (array - array.min()) / (array.max() - array.min())

    # Normalization information
    report, mode = histogram(array_normalized, percentile, bins=100)

    img_nifti_normalized = nip.Nifti1Image(array_normalized, img.affine)

    return img_nifti_normalized, report, mode


def threshold(img: nb.Nifti1Image,
              thr: float = 0.4,
              sign: str = '+',
              binarize: bool = False) -> nb.Nifti1Image:
    """Create a brain_mask by putting all the values below the threshold.
       to 0

       Args:
          img (nb.Nifti1Image): image to process
          thr (float): appropriate value to select mostly brain tissue
             (white matter) and remove background
          sign (str): '+' zero anything below, '-' zero anythin above threshold
          binarize (bool): make a binary mask

       Returns:
           nb.Nifti1Image: preprocessed image
    """
    import numpy as np
    import nibabel as nb

    if not isinstance(img, nb.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")
    if not isinstance(thr, float):
        raise TypeError("'thr' must be a float")

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
    thresholded = nip.Nifti1Image(array, img.affine)
    
    return thresholded


def crop(roi_mask: nb.Nifti1Image,
	     apply_to: nb.Nifti1Image,
         dimensions: Tuple[int, int, int],
         cdg_ijk: np.ndarray[int] = None
         ) -> nb.Nifti1Image:
    """Adjust the real-world referential and crop image.

    Args:
        roi_mask (nb.Nifti1Image): mask used to define the center
                                   of the bounding box (center of gravity of mask)
        apply_to (nb.Nifti1Image): image to crop
        dimensions (Tuple[int, int, int], optional): volume dimensions.
                                                     Defaults to (256 , 256 , 256).

    Returns:
        nb.Nifti1Image: preprocessed image
    """
    if not isinstance(roi_mask, nb.nifti1.Nifti1Image):
        raise TypeError("roi_mask: only Nifti images are supported")
    if not isinstance(apply_to, nb.nifti1.Nifti1Image):
        raise TypeError("apply_to: only Nifti images are supported")

    # Calculation of the center of gravity of the mask, we round and convert
    # to integers
    if not cdg_ijk:
        cdg_ijk = np.ceil(np.array(ndimage.center_of_mass(
		roi_mask.get_fdata()))).astype(int)
 
    # We will center the block on the center of gravity, so we cut the size in
    # 2
    halfs = np.array(dimensions)/2
    # we need integers because it is used for indices of the "array of voxels"
    halfs = halfs.astype(int)

    # the ijk of the lowest voxel in the box
    bbox1 = (cdg_ijk[0] - halfs[0], cdg_ijk[1] - halfs[1],
             cdg_ijk[2] - halfs[2])
    # the highest ijk voxel of the bounding box
    bbox2 = halfs + cdg_ijk

    # Check if value are negative shift the bounding box
    negative_shift = False
    for i in bbox1:
        padding_ijk = [0, 3, 0]
        if i < 0:
            negative_shift = True
            padding = abs(i)
            padding_ijk[bbox1.index(i)] = padding
            padding_data = np.zeros((apply_to.shape[0],
                                     apply_to.shape[1] + 2 * padding,
                                     apply_to.shape[2]))
            padding_data[:, padding:-padding, :] = apply_to.get_fdata()

            bbox1 = (bbox1[0], bbox1[1] + padding, bbox1[2])
            bbox2 = (bbox2[0], bbox2[1] + padding, bbox2[2])
    # We extract the box i1 -> i2, j1 -> j2, k1 -> k2 (we "slice")
            array_out = padding_data[
                bbox1[0]:bbox2[0],
                bbox1[1]:bbox2[1],
                bbox1[2]:bbox2[2]]

        else:
            array_out = apply_to.get_fdata()[
                        bbox1[0]:bbox2[0],
                        bbox1[1]:bbox2[1],
                        bbox1[2]:bbox2[2]]

    # We correct the coordinates, so first we have to convert ijk to xyz for
    # half block size and centroid
    cdg_xyz = apply_to.affine @ np.append(cdg_ijk, 1)
    halfs_xyz = apply_to.affine @ np.append(halfs, 1)
    padding_xyz = apply_to.affine @ np.append(tuple(padding_ijk), 1)
    offset_padding = apply_to.affine[:,3] - padding_xyz

    # on recopie la matrice affine de l'image, on va la modifier
    affine_out = deepcopy(apply_to.affine)

    # We shift the center of the image reference frame because we start from
    # less far (image is smaller)
    # And the center is no longer quite the same
    if negative_shift:
        affine_out[:, 3] = affine_out[:, 3] + (cdg_xyz - halfs_xyz) - offset_padding
    else:
        affine_out[:, 3] = affine_out[:, 3] + (cdg_xyz - halfs_xyz)

    # We write the result image
    cropped = nip.Nifti1Image(array_out, affine_out)

    return cropped, cdg_ijk, bbox1, bbox2


'''
def apply_mask(apply_to: nb.Nifti1Image,
               model: tf.keras.Model):

    image_tensor = tf.expand_dims(apply_to.get_fdata(), axis=0)
    brain_mask_array = image_tensor.numpy()
    prediction = model.predict(image_tensor)
       
    mask = prediction <= 0.5
    brain_mask_array = np.squeeze(brain_mask_array)
    mask = np.squeeze(mask)
    brain_mask_array[mask] = 0

    brain_mask = nip.Nifti1Image(brain_mask_array, apply_to.affine)

    return brain_mask
'''
               
               
