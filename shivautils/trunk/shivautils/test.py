import matplotlib.pyplot as plt
import io
import base64
import nibabel as nb
import numpy as np

img = nb.load("/homes_unix/yrio/Documents/data/TestSetGlobal/PVS_WMH/T1-FLAIR-brainmask_preprocessing_gin/21/t1/21_T1_cropped.nii.gz")
array = img.get_fdata()
x = array.reshape(-1)
hist, edges = np.histogram(x, bins=100)

fig, ax = plt.subplots()
ax.hist(x, bins=edges, color=(0.3, 0.5, 0.8))  # Utilisation d'une nuance de bleu atténuée (0.3, 0.5, 0.8)
ax.set_yscale('log')
ax.set_title("Histogram of intensities voxel values")
ax.set_xlabel("Voxel Intensity")
ax.set_ylabel("Number of Voxels")

plt.savefig('hist.png', format='png')

