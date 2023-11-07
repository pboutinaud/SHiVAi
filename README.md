# SHIVA preprocessing and deep learning segmentation workflows

This package includes a set of image analysis tools for the study of covert cerebrovascular diseases with structural Magnetic Resonance Imaging.

The SHIVA segmentation tools currently include cerebral microbleeds, Virchow Robin Spaces (perivascular spaces, PVS) and White Matter Hyperintensities (WMH). The 3D-Unet model weights are available separately at https://github.com/pboutinaud.

The tools cover preprocessing (image resampling and cropping to match the required size for the deep learning models, coregistration for multimodal segmentation tools), predictions (model weights available ), and reporting (QC and results).

The package includes an Apptainer (Singularity) container image recipe file. 

## Dependencies

The deep learning relies on Tensorflow 2.7.13, and a GPU with 16Gb of memory is necessary. The processing pipelines are implemented with Nipype and make use of ANTS for image registration. Quality control reporting uses DOG contours from: https://github.com/neurolabusc/PyDog.

## Package Installation

To deploy the python package, from the project directory (containing the 'pyproject.toml' file), use the following command line: 

```bash
python -m pip install .
```

The scripts should be available in the command line prompt.

## Apptainer image

The SHIVA application requires a Linux machine with a GPU (with 16GB of dedicated memory), and **AppTainer** intalled (previously known as **Singularity**):
https://apptainer.org/docs/user/main/quick_start.html

AppTainer is a container solution, meaning that you will need an AppTainer image (a file with the *.sif* extension) containing all the software and environment necessary to run SHIVA. You can get it from us (https://cloud.efixia.com/sharing/TAHaV6ZgZ) or build it yourself.

To build the AppTainer image, you need a machine on which you are *root* user. Then, from the **shivautils/singularity** directory, run the following command to build the SHIVA AppTainer container image :

```bash
apptainer build shiva.sif apptainer_tf.recipe
```
Then you can move the AppTainer image on any computer with AppTainer installed and run the processing even without being a root user on that machine.

Note that if you are on a **Windows** computer, you can use WSL (Windows Subsystem for Linux) to run AppTainer and build the image. You can find more info here https://learn.microsoft.com/windows/wsl/install. With WSL installed, open a command prompt, type `wsl` and you will have access to a Linux terminal where you can install and run AppTainer.
There are also similar options for **Mac** users (check the dedicated section from https://apptainer.org/docs/admin/main/installation.html).

### Other files

You will need to copy the `run_shiva.py` script (available in `shivautils/singularity/run_shiva.py`) to the computer that will run the process.
To use it (see below), you will also need a minimal python environment with the pyyaml library installed.

You will need to obtain the trained AI model accompanying the Shiva project. Let's consider that you stored it in `/myHome/myProject/Shiva_AI_models` for the following parts.

You now need to prepare a configuration file that will hold diverse parameters as well as the path to the AI model and to the apptainer image.
You can find the `config_example.yml` example configuration file in `shivautils/singularity`.

There, you should change the placeholder paths for `model_path` and `apptainer_image` with your own paths.
Normally you shouldn't have to modify the `parameters` part, except maybe the `SWI` parameter that you can swap to `False` if you don't have SWI acquisitions in your dataset. Let's say that you now have the config file prepared in `/myHome/myProject/myConfig.yml`.

## Running the process

### Requirements

To run the shiva process, you will need:
- The `run_shiva.py` script that you can find in the present repository (shivautils/singularity/run_shiva.py)
- The input dataset (see below for example of accepted file strucures)
- The apptainer image (*shiva.sif* above)
- The trained AI model (that we provide)
- A configuration file (.yml) that will contain all the options and various paths needed for the workflow

### Command line arguments (with `run_shiva.py`)

    --in (path): Path of the input dataset
    --out (path): Path to where the generated files will be saved
    --input_type (str): Type of structure file, way to capture and manage nifti files : standard, BIDS or json
    --config (path): File with configuration options for the workflow

### Command line examples

Locally running the processing (from the directory where you stored `run_shiva.py`): 

```bash
python run_shiva.py --in /myHome/myProject/MyDataset --out /myHome/myProject/shiva_results --input_type standard --config /myHome/myProject/myConfig.yml
```

Using SLURM to run on a grid (with GPU resources configured): 

```bash
srun –gpus 1 python run_shiva.py --in /myHome/myProject/MyDataset --out /myHome/myProject/shiva_results --input_type standard --config  /myHome/myProject/myConfig.yml
```
 
### Data structures accepted by `run_shiva.py`

Example of `BIDS` structure folders:

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


Example of `standard` structure folders:

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


Example of `json` structure input:
```json
{
    "parameters": {
        "out_dir": "/mnt/data/output",
        "brainmask_descriptor": "/myHome/myProject/Shiva_AI_models/model_info/brainmask/model_info.json",
        "WMH_descriptor": "/myHome/myProject/Shiva_AI_models/model_info/T1.FLAIR-WMH/model_info.json",
        "PVS_descriptor": "/myHome/myProject/Shiva_AI_models/model_info/T1.FLAIR-PVS/model_info.json",
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
    "files_dir": "/myHome/myProject/MyDataset",
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
```

### More info on the .yml configuration file:

Options configuration in yaml file:

    --model_path (path) : bind mount path for the tensorflow models directory
    --singularity_image (path): path of singularity container file
    --brainmask_descriptor (path): path to brain_mask tensorflow model descriptor
    --WMH_descriptor (path): path of White Matter HyperIntensities tensorflow model descriptor
    --PVS_descriptor (path): path of Peri-Vascular Spaces tensorflow model descriptor
    --CMB_descriptor (path) path of Cerebral MicroBleeds tensorflow model descriptor
    --percentile (float) : Threshold value expressed as percentile
    --threshold (float) : Value of the treshold to apply to the image for brainmask, default value : 0.5
    --threshold_clusters (float) : Threshold to compute clusters metrics, default value : 0.2
    --final_dimensions (str) : Final image array size in i, j, k. Example : '160 214 176'.
    --voxels_size (str) : Voxel size of the final image, example : '1.0 1.0 1.0'
    --grab : data grabber
    --SWI (str) : if a second workflow for CMB is required, example : 'True' or 'False' 
    --interpolation (str): image resampling method, default interpolation : 'WelchWindowedSinc', others ANTS interpolation possibilities : 'Linear', 'NearestNeighbor', 'CosineWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian', 'GenericLabel'

## Pipeline description

SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
preprocess a lot of images at the same time. In the last step the file segmentations with the wmh and 
pvs models are processed


### Preprocessing

1. **Conform**: Resample image to Predictor dimensions (currently 160x214x176) by adjusting the voxel size

2. **Intensity Normalization**: We remove values above the 99th 'percentile' (parameter default value) to avoid hot spots,
    set values below 0 to 0, set values above 1.3 to 1.3 and normalize the data between 0 and 1

3. **Pre BrainMask Prediction**: Run predict to segment brainmask from resampled structural images with Tensorflow model

4. **Threshold**: Create a binarized brain_mask by putting all the values below the 'threshold' to 0

5. **Crop**: adjust the real-world referential and crop image. With the pre brainmask prediction, the procedure uses the center of mass of the mask to center the bounding box. If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine referential coordinates origin. If set to 'ijk', the middle of the image is used.

6. **Final Brainmask**: Run predict to segment brainmask from cropping images with Tensorflow model. 

7. **Coregistration** : Register a FLAIR images on T1w images either through the full interface to the ANTs registration method.


### Reporting

Individual CSV file (path folder 'subject_{id}/prediction_metrics_{pvs/wmh}'):

- Cluster Threshold
- Cluster Filter
- Number of voxels
- Number of clusters 
- Mean clusters size
- Median clusters size
- Minimal clusters size 
- Maximal clusters size


Individual HTML / PDF Report (path folder 'subject_{id}/summary_report'):

- Summary of metrics clusters per subject
- Histogram of voxel intensity during main normalization
- Display of the cropping region on the conformed image
- T1 on FLAIR isocontour slides 
- Overlay of final brainmask over cropped main images
- Preprocessing workflow diagram

General CSV file (path folder 'prediction_metrics_{pvs/wmh}_all'):

- Sum of metrics clusters with one arrow per subjects.


