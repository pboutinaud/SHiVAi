Run containerized deep learning segmentation workloads
------------------------------------------------------

1. Download the model weights

2. Build the container image


Install Apptainer (formerly Singularity) :
https://apptainer.org/docs/user/main/quick_start.html

Nvidia container toolkit (GPU support):
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


From this folder, logged in as root (or sudo) use the following command to build the singularity image file:

singularity build preproc_tensorflow.sif singularity_tf.recipe

%files
    */shivautils /usr/local/src/shivautils

* path to your local 'shivautils' source code folder path 


3. Run the command line

In your python environment, install the *pyyaml* package with:

    pip install pyyaml

**run_shiva.py** command line arguments:


--in (path): path of the input dataset
--out (path): the path where the generated files will be output
--input_type (str):  Type of structure file, way to capture and manage nifti files : standard, BIDS or json
-- model (path): Path to model descriptor
--config (path): File with configuration options for workflow processing images, (path_default : /homes_unix/yrio/Documents/Script_predict/container_preproc_predict/config_wmh.yml)


Command line example : 

    python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~/shiva_Nagahama_test_preprocessing_dual  
    --input_type standard --config ~/.shiva/config_dual.yml

SLURM job manager command line example : 

    srun –gpus 1 python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~shiva_Nagahama_test_preprocessing_dual --input_type standard --config ~/.shiva/config_wmh.yml


Option configuration in yaml file :

    --model_path (path) : structure mount_file
    --singularity_image (path): path of singularity container file
    --brainmask_descriptor (path): path to brain_mask descriptor tensorflow model
    --WMH_descriptor (path): path of White Matter HyperIntensities descriptor tensorflow model
    --PVS_descriptor (path): path of Peri-Vascular Spaces descriptor tensorflow model
    --percentile (float) : Threshold value expressed as percentile
    --threshold (float) : Value of the treshold to apply to the image for brainmask, default value : 0.5
    --threshold_clusters (float) : Threshold to compute clusters metrics, default value : 0.2
    --final_dimensions (str) : Final image array size in i, j, k. Example : '160 214 176'.
    --voxels_size (str) : Voxel size of the final image, example : '1.0 1.0 1.0' --grab : data grabber
    --SWI (str) : if a second workflow for CMB is required, example : 'True' or 'False' 
    --interpolation (str): Way of upsamples images, default interpolation : 'WelchWindowedSinc', others interpolations  possibility : 'Linear', 'NearestNeighbor', 'CosineWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian', 'GenericLabel'


Example of BIDS input structure folders (see the BIDS staging specification):

    .
    ├── dataset_description.json
    └── rawdata
        └── sub-21
            └── anat
                ├── sub-21_FLAIR_raw.nii.gz
                └── sub-21_T1_raw.nii.gz
        └── sub-51
            └── anat
                ├── sub-51_FLAIR_raw.nii.gz
                └── sub-51_T1_raw.nii.gz


Example of Standard structure folders :

    .
    ├── 21
    │   ├── flair
    │   │   └── 21_FLAIR_raw.nii.gz
    │   └── t1
    │       └── 21_T1_raw.nii.gz
    └── 51
        ├── flair
        │   └── 51_FLAIR_raw.nii.gz
        └── t1
            └── 51_T1_raw.nii.gz


Example of JSON-structured input :

{
    "parameters": {
        "out_dir": "/mnt/data/output",
        "brainmask_descriptor": "/homes_unix/yrio/Documents/modele/ReferenceModels/model_info/brainmask/model_info.json",
        "WMH_descriptor": "/homes_unix/yrio/Documents/modele/ReferenceModels/model_info/T1.FLAIR-WMH/model_info.json",
        "PVS_descriptor": "/homes_unix/yrio/Documents/modele/ReferenceModels/model_info/T1.FLAIR-PVS/model_info.json",
        "percentile": 99.0,
        "final_dimensions": [
            160,
            214,
            176
        ],
        "voxels_size": [
            1.0,
            1.0,
            1.0
        ]
    },
    "files_dir": "/homes_unix/yrio/Documents/data/TestSetGlobal/PVS_WMH/T1-FLAIR_raw",
    "all_files": {
        "21": {
            "t1": "21_T1_raw.nii.gz",
            "flair": "21_FLAIR_raw.nii.gz"
        },
        "51": {
            "t1":"51_T1_raw.nii.gz",
            "flair": "51_FLAIR_raw.nii.gz"
        }
    }
}

Workflow description
-------------------- 

The SHIVA preprocessing for deep learning predictors is in charge of the resampling of a NIfTI-formatted 
structural  head image, followed by intensity normalization, and cropping centered on the brain.

A nipype workflow is used to preprocess images in batch. Predictions are sthen performed on the supplied images.

In the last step the image segmentations from the wmh and pvs models are analyzed and a report is generated.

Preprocessing steps
===================

1 - Conform : Resample image to 'final_dimensions with voxels of size 'voxels_size'

2 - Intensity Normalization : We remove values above the 99th 'percentile' (parameter default value) to avoid hot spots,
       set values below 0 to 0, set values above 1.3 to 1.3 and normalize the data between 0 and 1

3 - BrainMask Prediction : Run predict to segment brainmask from reformated structural images with Tensorflow model

4 - Threshold : Create a binarized brain_mask by putting all the values below the 'threshold' to 0

5 - Crop : adjust the real-world referential and crop image. If a mask is supplied, the procedure uses the center of mass of the mask as 
       a crop center. If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine referential coordinates origin. If set to 'ijk', the middle of the image is used.

6 - Coregistration : Register a FLAIR images on T1w images either through the full interface to the ANTs registration method.

Reporting
=========

Individual CSV file (path folder 'subject_{id}/metrics_predictions_{pvs/wmh}'):

In the 'metrics_predictions_pvs' folder or 'metrics_prediction_wmh' of the main folder, there will be a *.csv* file describing all cluster metrics for the perivascular spaces or white matter hyperintensities :
- Number of voxels
- Number of clusters
- Mean clusters size
- Median clusters size
- Minimal clusters size
- Maximal clusters size
- Predictions results in ventricles and deep white matters hyperintensities clusters (for wmh)
- Predictions results in basal ganglia and deep white matters clusters (for pvs)

Individual HTML / PDF Report (path folder 'subject_{id}/summary_report'):

- Summary of metrics clusters per subject
- Histogram of voxel intensity during main normalization
- Display of the cropping region on the conformed image
- T1 on FLAIR isocontour slides 
- Overlay of final brainmask over cropped main images
- Preprocessing workflow diagram

General *.csv* file (path folder 'metrics_predictions_{pvs/wmh}_generale'):

 - Sum of cluster metrics with one row per subject.
