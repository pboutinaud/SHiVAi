# Run containerized deep learning segmentation workloads

## Apptainer Installation

## CUDA Toolkit Docker

## Synthseg Apptainer image

### Apptainer image

Apptainer is a container solution, meaning that you can run SHiVAi using an Apptainer image (a file with the *.sif* extension) containing all the software and environment necessary. You can get the SHiVAi image from us (https://cloud.efixia.com/sharing/TAHaV6ZgZ) or build it yourself.

To build the Apptainer image, you need a machine on which you are *root* user. Then, from the **shivai/apptainer** directory, run the following command to build the SHIVA Apptainer container image :

```bash
apptainer build shiva.sif apptainer_tf.recipe
```
Then you can move the Apptainer image on any computer with Apptainer installed and run the processing even without being a root user on that machine.

Note that if you are on a **Windows** computer, you can use WSL (Windows Subsystem for Linux) to run Apptainer and build the image. You can find more info here https://learn.microsoft.com/windows/wsl/install. With WSL installed, open a command prompt, type `wsl` and you will have access to a Linux terminal where you can install and run Apptainer.
There are also similar options for **Mac** users (check the dedicated section from https://apptainer.org/docs/admin/main/installation.html).

1. Download the model weights

2. Build the container image


Install Apptainer (formerly Singularity) :
https://apptainer.org/docs/user/main/quick_start.html

Nvidia container toolkit (for GPU support, **required**):
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


From this folder, logged in as root (or sudo) use the following command to build the singularity image file:

    singularity build preproc_tensorflow.sif singularity_tf.recipe


3. Run the command line

The container is launched via a wrapper script (**run_shiva.py**). In your python environment, first install the *pyyaml* package with:

    pip install pyyaml

**run_shiva.py** command line arguments:

    --in (path): path of the input dataset
    --out (path): the path where the generated files will be output
    --input_type (str):  Type of structure file, way to capture and manage nifti files : standard, BIDS or json
    --config (path): File with configuration options for workflow processing images

Command line example : 

    python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~/shiva_Nagahama_test_preprocessing_dual  
    --input_type standard --config ~/.shiva/config_dual.yml

SLURM job manager command line example : 

    srun –gpus 1 python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~shiva_Nagahama_test_preprocessing_dual --input_type standard --config ~/.shiva/config_wmh.yml


The 'fixed' options are set in the *--config* yaml file:


    --model_path (path) : the path to the model folder, which will be mounted by the container.
    --singularity_image (path): path of singularity image file
    --brainmask_descriptor (path): path to brain_mask descriptor tensorflow model
    --WMH_descriptor (path): path of White Matter HyperIntensities descriptor tensorflow model
    --PVS_descriptor (path): path of Peri-Vascular Spaces descriptor tensorflow model
    --percentile (float) : Threshold value expressed as percentile
    --threshold (float) : treshold to apply to the image to obtain a head mask, default value : 0.5
    --threshold_clusters (float) : Threshold to compute clusters metrics, default value : 0.2
    --final_dimensions (str) : Final image array size in i, j, k. Example : '160 214 176'.
    --voxels_size (str) : Voxel size of the final image, example : '1.0 1.0 1.0' --grab : data grabber
    --SWI (str) : if a second workflow for CMB is required, example : 'True' or 'False' 
    --interpolation (str): for image resampling with ANTS, default interpolation : 'WelchWindowedSinc', others interpolations  possibility : 'Linear',     
                           'NearestNeighbor', 'CosineWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian', 'GenericLabel'

