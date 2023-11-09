# SHIV-AI: SHIVA preprocessing and deep learning segmentation workflow

This package includes a set of image analysis tools for the study of covert cerebrovascular diseases with structural Magnetic Resonance Imaging.

The SHIVA segmentation tools currently include Cerebral MicroBleeds (CMB),  PeriVascular Spaces (PVS) (also known as Virchow Robin Spaces - VRS), and White Matter Hyperintensities (WMH). The 3D-Unet model weights are available separately at https://github.com/pboutinaud.

The tools cover preprocessing (image resampling and cropping to match the required size for the deep learning models, coregistration for multimodal segmentation tools), segmentation (model weights available ), and reporting (QC and results).

## Dependencies and hardware requirements

The SHIV-AI application requires a Linux machine with a GPU (with 16GB of dedicated memory).

The deep-learning models relies on Tensorflow 2.7.13. The processing pipelines are implemented with Nipype and make use of ANTs for image registration. Quality control reporting uses (among others) DOG contours from: https://github.com/neurolabusc/PyDog. Building and/or using the container image relies on Apptainer (https://apptainer.org). More details about Apptainer in the [Apptainer image](#apptainer-image) section.

## Package Installation

Depending on your situation you may want to deploy SHIV-AI in different ways:
- **Fully contained process**: The simplest approach. All the computation is done through the Apptainer image. It accounts for most of the local environment set-up, which simplifies the installation and ensure portability.
- **Traditional python install**: does not require singularity as all the dependencies will have to be installed locally. Useful for full control and development of the package, however it may lead to problems due to the finicky nature of TensorFlow and CUDA.
- *Mixed approach (NOT IMPLEMENTED YET)*: Local installation of the package without TensorFlow (and so without troubles), but using the Apptainer image to run the deep-learning processes (using TensorFlow).Ideal for parallelization of the processes and use on HPC clusters.

In all these situations, **you will also need** to obtain the trained deep-learning models you want to use (for PVS, WMH, and CMB segmentation). They are available at https://github.com/pboutinaud

Let's consider that you stored them in `/myHome/myProject/Shiva_AI_models` for the following parts.

>**/!\\** For the process too work, a `model_info.json` must be present in the folder containing the AI model files (the .h5 files). If it's not the case, see the [Create missing json file](#create-missing-json-file) section, and don't forget to update the config file (see [Fully contained process](#fully-contained-process)) if you use one.

### Fully contained process

1. You will need to have **Apptainer** installed (previously known as **Singularity**):
https://apptainer.org/docs/user/main/quick_start.html
    Let's assume you saved it in `/myHome/myProject/shiva.sif`

2. Download the Apptainer image (.sif file) from https://cloud.efixia.com/sharing/TAHaV6ZgZ.

From the shivautils repository (where you are reading this), go to the 'singularity' folder and download:

3. `run_shiva.py`

and

4. `config_example.yml`

    You now need to prepare this configuration file, it will hold diverse parameters as well as the path to the AI model and to the apptainer image.
    There, you should change the placeholder paths for `model_path` and `apptainer_image` with your own paths (e.g. `/myHome/myProject/Shiva_AI_models` and `/myHome/myProject/shiva.sif`). You may also have to set the model descriptors (like `PVS_descriptor` or `WMH_descriptor` with the path to the `model_info.json` file)
    Normally you shouldn't have to modify the `parameters` part, except if you need to change some specific settings like the  size filter (*min_\*_size*) for the different biomarkers.

    For the rest of this readme, let's assume that you now have the config file prepared in `/myHome/myProject/myConfig.yml`.

5. Finally, set-up a minimal Python virtual environment with the `pyyaml` package installed.

Next, see [Running a contained SHIV-AI](#running-a-contained-shiv-ai)

### Traditional python install

To deploy the python package, create a Python 3.9 virtual environment, clone or download the shivautils project and use the following command line from the project's directory (containing the 'pyproject.toml' file): 

```bash
python -m pip install .[TF]
```

The scripts should then be available from the command line prompt.

Optionally, you can download and prepare the `config_example.yml` like explained in the [Fully contained process](#fully-contained-process) section. This will ease the command call as a lot of arguments will be given through the yaml file (instead of manually entered with the command).

Next, see [Running SHIV-AI from direct package commands](#running-shiv-ai-from-direct-package-commands)

### Mixed approach (WIP)

*WORK IN PROGRESS*

## Running the process

### Running a contained SHIV-AI

To run the shiva process, you will need:
- The `run_shiva.py` (mentioned above)
- The input dataset (see [Data structures accepted by SHIV-AI](#data-structures-accepted-by-shiv-ai))
- The apptainer image (*shiva.sif* above)
- The trained AI model (that we provide and you should have downloaded)
- A configuration file (.yml) that will contain all the options and various paths needed for the workflow

**Command line arguments (with `run_shiva.py`):**

    --in: Path of the input dataset
    --out: Path to where the generated files will be saved
    --input_type: Type of structure file, way to capture and manage nifti files : standard, BIDS or json
    --prediction: Choice of the type of prediction (i.e. segmentation) you want to compute (PVS, PVS2, WMH, CMB, all). Give a combination of these labels separated by blanc spaces.
    --config: File with configuration options for the workflow

**Command line examples**

Locally running the processing (from the directory where you stored `run_shiva.py`): 

```bash
python run_shiva.py --in /myHome/myProject/MyDataset --out /myHome/myProject/shiva_results --input_type standard --prediction PVS CMB --config /myHome/myProject/myConfig.yml
```

Using SLURM to run on a grid (with GPU resources configured): 

```bash
srun –gpus 1 python run_shiva.py --in /myHome/myProject/MyDataset --out /myHome/myProject/shiva_results --input_type standard --prediction PVS CMB --config  /myHome/myProject/myConfig.yml
```

### Running SHIV-AI from direct package commands

From the virtual environment where you installed shivautils, run the command `shiva` (calling the `shiva.py` script).

To see the detailed help for this command, you can call:
```bash
shiva -h
```

Here is an example of a shiva call, using a config .yml file:
```bash
shiva --in /myHome/myProject/MyDataset --out /myHome/myProject/shiva_results --input_type standard --prediction PVS CMB --model_config /myHome/myProject/myConfig.yml --gpu 0 --retry 
```

## Results

The results will be stored in the `results` folder in your output folder (so `/myHome/myProject/shiva_results/results` in our example). There you will find the results for individual participants as well as a results_summary folder that contains grouped data like statistics about each subjects segmentation and quality control (QC).

You will also find a PDF report for each participant detailing statics about their segmentation and QC in `results/participantXXXX/report/_subject_id_participantXXXX/summary.pdf`


## Data structures accepted by SHIV-AI

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

## Additional info

### Create missing json file

In some cases, the model_info.json might be missing from the model folder you downloaded. To create it, you need to use the `prep_json.py` script, found in src/shivautils/scripts/prep_json.py.

Let's assume you downloaded the T1-PVS model (for PVS detection using only T1 images), you should now have it in `/myHome/myProject/Shiva_AI_models/T1-PVS` (or something close to this).

If you directly download `prep_json.py` it, you can run it with:
```bash
python prep_json.py --folder /myHome/myProject/Shiva_AI_models/T1-PVS
```

If you installed the shivautils package, you can directly run the command line:
```bash
prep_json --folder /myHome/myProject/Shiva_AI_models/T1-PVS
```

### Apptainer image

Apptainer is a container solution, meaning that you can run SHIV-AI using an Apptainer image (a file with the *.sif* extension) containing all the software and environment necessary. You can get the SHIV-AI image from us (https://cloud.efixia.com/sharing/TAHaV6ZgZ) or build it yourself.

To build the Apptainer image, you need a machine on which you are *root* user. Then, from the **shivautils/singularity** directory, run the following command to build the SHIVA Apptainer container image :

```bash
apptainer build shiva.sif apptainer_tf.recipe
```
Then you can move the Apptainer image on any computer with Apptainer installed and run the processing even without being a root user on that machine.

Note that if you are on a **Windows** computer, you can use WSL (Windows Subsystem for Linux) to run Apptainer and build the image. You can find more info here https://learn.microsoft.com/windows/wsl/install. With WSL installed, open a command prompt, type `wsl` and you will have access to a Linux terminal where you can install and run Apptainer.
There are also similar options for **Mac** users (check the dedicated section from https://apptainer.org/docs/admin/main/installation.html).

### More info on the .yml configuration file

- **model_path** (path) : bind mount path for the tensorflow models directory
- **singularity_image** (path): path of singularity container file
- **brainmask_descriptor** (path): path to brain_mask tensorflow model descriptor
- **PVS_descriptor** (path): path of Peri-Vascular Spaces tensorflow model descriptor
- **PVS2_descriptor**  (path): path of Peri-Vascular Spaces tensorflow model descriptor using t1 and flair together
- **WMH_descriptor** (path): path of White Matter HyperIntensities tensorflow model descriptor
- **CMB_descriptor** (path) path of Cerebral MicroBleeds tensorflow model descriptor
- **percentile** (float) : Threshold value for the intensity normalization, expressed as percentile
- **threshold** (float) : Value of the threshold used to binarize brain masks, default value : 0.5
- **threshold_clusters** (float) : Threshold to binarize clusters after the segmentation, default value : 0.2
- **min_pvs_size** (int): Filter size (in voxels) for detected PVS under which the cluster is discarded
- **min_wmh_size** (int): Filter size (in voxels) for detected WMH under which the cluster is discarded
- **min_cmb_size** (int): Filter size (in voxels) for detected CMB under which the cluster is discarded
- **final_dimensions** (list) : Final image array size in i, j, k. Example : [160, 214, 176].
- **voxels_size** (list) : Voxel size of the final image, example : [1.0, 1.0, 1.0]
- **interpolation** (str): image resampling method, default interpolation : 'WelchWindowedSinc', others ANTS interpolation possibilities : 'Linear', 'NearestNeighbor', 'CosineWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian', 'GenericLabel'

### Pipeline description

Performs resampling of a structural NIfTI brain image, followed by intensity normalization, and cropping centering on the brain. Then, PVS, WMH, and CMB are segmented (depending on the input) using a deep-learning model. Finally, quality control measures are generated and all th results are aggregated in a pdf report.   


### Preprocessing

1. **Conform**: Resample image to Predictor dimensions (currently 160x214x176) by adjusting the voxel size

2. **Intensity Normalization**: We remove values above the 99th 'percentile' (parameter default value) to avoid hot spots,
    set values below 0 to 0, set values above 1.3 to 1.3 and normalize the data between 0 and 1

3. **Pre BrainMask Prediction**: Run predict to segment brainmask from resampled structural images with Tensorflow model

4. **Threshold**: Create a binarized brain_mask by putting all the values below the 'threshold' to 0

5. **Crop**: adjust the real-world referential and crop image. With the pre brainmask prediction, the procedure uses the center of mass of the mask to center the bounding box. If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordinates of the affine referential coordinates origin. If set to 'ijk', the middle of the image is used.

6. **Final Brainmask**: Run predict to segment brainmask from cropping images with Tensorflow model. 

7. **Coregistration** : Register a FLAIR images on T1w images either through the full interface to the ANTs registration method.


