# Run containerized deep learning segmentation workloads

### Apptainer image

Apptainer is a container solution, meaning that you can run SHiVAi using an Apptainer image (a file with the *.sif* extension) containing all the software and environment necessary. You can get the SHiVAi image from us (https://cloud.efixia.com/sharing/oUBzl7d0Z) or build it yourself.

To build the Apptainer image, you need a machine on which you are *root* user. Then, from the **shivai/apptainer** directory, run the following command to build the SHIVA Apptainer container image :

```bash
apptainer build shiva.sif apptainer_tf.recipe
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
To create the Synthseg container, follow the same procedure as for the Shivai pipeline explained above, with:

```singularity build synthseg.sif apptainer_synthseg_tf.recipe```

Then add the path to the `synthseg.sif` image to the yaml config file in the dedicated place.