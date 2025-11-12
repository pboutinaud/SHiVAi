# Run containerized deep learning segmentation workloads

### Apptainer image

Apptainer is a container solution, meaning that you can run SHiVAi using an Apptainer image (a file with the *.sif* extension) containing all the software and environment necessary. You can get the SHiVAi image from us [https://cloud.efixia.com/sharing/t3jG8DICk](https://cloud.efixia.com/sharing/t3jG8DICk) or build it yourself.

To build the Apptainer image, you need a machine on which you are *root* user. Then, from the **shivai/apptainer** directory, run the following command to build the SHIVA Apptainer container image :

```bash
apptainer build shivai.sif apptainer_tf.recipe
```
Then you can move the Apptainer image on any computer with Apptainer installed and run the processing even without being a root user on that machine.

Note that if you are on a **Windows** computer, you can use WSL (Windows Subsystem for Linux) to run Apptainer and build the image. You can find more info here https://learn.microsoft.com/windows/wsl/install. With WSL installed, open a command prompt, type `wsl` and you will have access to a Linux terminal where you can install and run Apptainer.
There are also similar options for **Mac** users (check the dedicated section from https://apptainer.org/docs/admin/main/installation.html).

1. Download the model weights

2. Install Apptainer (formerly Singularity)

Follow the Apptainer documentation:
https://apptainer.org/docs/user/main/quick_start.html

Nvidia container toolkit (for GPU support) may also be required, see with you IT:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

3. (opt.) Build the container image (if you haven't downloaded it)

From this folder (shivai/apptainer/), logged in as root (or sudo) use the following command to build the singularity image file:

```singularity build shivai.sif apptainer.recipe```

4. Run the command line as [described in the main readme](../README.md/#running-shivai-from-a-container)

## Synthseg Apptainer image
To create the Synthseg container, you first need to download the Synthseg models from the UCL OnedDrive using [this link](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmappmb_ucl_ac_uk/EtlNnulBSUtAvOP6S99KcAIBYzze7jTPsmFk2_iHqKDjEw?e=rBP0RO). If the link doesn't work, refer to the one given directly on the Synthseg repository [here](https://github.com/BBillot/SynthSeg/tree/master?tab=readme-ov-file#installation).
Downloading the folder should give you a .zip file (with a name like *OneDrive_\*.zip*).
Rename this file as `synthseg_models.zip` and put it in the same folder as the `apptainer_synthseg_tf.recipe` file.

> To check if everything is as it should: When unzipped, `synthseg_models.zip` should yield a `synthseg models` folder containing the Synthseg models as .h5 files.

Then follow the same procedure as for the Shivai pipeline explained above, with:

```singularity build synthseg.sif apptainer_synthseg_tf.recipe```

Then add the path to the `synthseg.sif` image to the yaml config file in the dedicated place.

## Synthseg Docker image
First navigate to the the [apptainer](.) folder of the project (containing the [Sythseg dockerfile](./synthseg.Dockerfile)) and follow the same directions from the [Synthseg Apptainer image](#synthseg-apptainer-image) section regarding the `synthseg_models.zip` file needed before building the image.

Then, to build the image (you will need root privileges), run (replace `myId` by your username or something equivalent):

```
docker build --rm -f synthseg.Dockerfile -t myId/synthseg_shivai .
```
