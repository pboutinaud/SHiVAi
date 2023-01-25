"""Script workflow"""
import os
import nibabel as nb
import numpy as np
from shivautils.workflows.vrs_preprocessing import genWorkflow
from shivautils.workflows.param_wf import genParamWf

from shivautils.stats import set_percentile

data_dir = os.path.normpath('/mnt/c/scratch/raw_2/BB_SWAN')
output_dir = '/mnt/c/scratch/nipype_out_3'

# This path iq used to retrieve the images of the cohort on which we
# want to perform the same voxel intensity normalization (ex: CMBDOU)
path_reference_cohort = '/mnt/c/scratch/preprocessed_2/BB_SWAN/'
images_path = os.listdir(path_reference_cohort)

list_images_cohort = []
for i in images_path:
    image = nb.loadsave.load(path_reference_cohort+i)
    list_images_cohort.append(image)


voxel_size = (1.0, 1.0, 1.0)
final_dimensions = (160, 214, 176)
bins = 100

test_percentile = [i for i in np.arange(95, 99, 0.25)]
subject_list = os.listdir(data_dir)
args_1 = {'SUBJECT_LIST': subject_list,
          'BASE_DIR': data_dir,
          'percentiles': test_percentile,
          'voxel_size': voxel_size,
          'final_dimensions': final_dimensions}

wf_p = genParamWf(**args_1)
wf_p.base_dir = output_dir
res = wf_p.run(plugin="MultiProc")
list_node = list(res.nodes)

percentile = set_percentile(list_node,
                            test_percentile,
                            bins,
                            list_images_cohort)

print(percentile)

args_2 = {'SUBJECT_LIST': subject_list,
          'BASE_DIR': data_dir,
          'percentile': percentile,
          'voxel_size': voxel_size,
          'final_dimensions': final_dimensions}

wf = genWorkflow(**args_2)
wf.base_dir = output_dir
wf.run(plugin="MultiProc")
