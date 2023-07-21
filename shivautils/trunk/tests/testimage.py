"""Unit tests for SHIVA preprocessing tools
"""
import unittest
import nibabel as nb
import nibabel.processing as nip

from shivautils.image import normalization, thresholding, crop


class TestImage(unittest.TestCase):
    """Test function of module image."""
    def setUp(self):
        self.img = nb.loadsave.load("C:/scratch/raw/01.nii")

    def test_normalize_run(self):
        """Check if value max of normalized image is less than img input
        """
        self.assertLessEqual(normalization(self.img).get_fdata().max(),
                             self.img.get_fdata().max())

    def test_crop_modified_shape(self):
        """Check modification of shape's image
        """
        resampled = nip.conform(self.img, out_shape=(256, 256, 256),
                                voxel_size=(1, 1, 1), order=3,
                                cval=0.0, orientation='RAS',
                                out_class=None)
        img_cropped, cdg_xyz, bbox1, bbox2 = crop(resampled,
                                                  dimensions=(160, 214, 176),
                                                  voxel_size=(1, 1, 1),
                                                  orientation='RAS')
        self.assertEqual((160, 214, 176), img_cropped.shape)

    def test_threshold_run(self):
        """Check if value max of thresholding image is less than img input
        """
        self.assertLessEqual(thresholding(self.img).get_fdata().mean(),
                             self.img.get_fdata().mean())


if __name__ == '__main__':
    unittest.main()
