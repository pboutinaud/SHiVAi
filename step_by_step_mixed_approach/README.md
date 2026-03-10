# SHiVAi Quick Start: Mixed Approach (Local Package + Apptainer, BIDS, Synthseg, PVS2/WMH/LAC)

This guide summarizes the essential steps to run SHiVAi using the **mixed approach**: the shivai package is installed locally (without TensorFlow, which avoids dependency conflicts), while the Apptainer image is used for the deep-learning steps that require TensorFlow and CUDA. This is the **recommended approach** as it supports parallelization of the pipeline steps (e.g. via SLURM on HPC clusters), unlike the fully contained approach.

Here we will run bi-modal PVS and WMH segmentation, using the Synthseg masking scheme, on BIDS-structured data. Other segmentations (lacunes, CMB, mono-modal PVS) follow exactly the same process and can be added by adapting the models and predictions described below.

In this guide, we will assume everything is stored in a folder called `~/myShivaiProject/` as an example.

> Note:\
> SHiVAi offers a lot of flexibility regarding inputs and processing, but for the sake of simplicity in this step-by-step README, we will only focus on basic commands.
---

## 1. Requirements

### System

- **Linux machine** with GPU (16GB memory recommended) — CPU-only processing is *possible* but very slow.

### Data

Depending on the cSVD segmentation you want to run, you will need different types of acquisition:

- PVS : T1w (and FLAIR optionally)
- WMH : T1w + FLAIR
- Lacunes : T1w + FLAIR
- CMB : SWI

For reproducible statistical analysis, a brain parcellation is also required.

> Note:\
> Technically, you don't need a brain parcellation to run SHiVAi, you only need a brain mask if you don't care about region-wise analysis of the detected cSVD markers. We also provide an AI model for automatic brain mask creation within the SHiVAi pipeline. You can see more about that [in the main README](../README.md/#brain-masking).

**Here we need to make a distinction between two cases:**

- If you have access to a Freesurfer-type parcellation of the brain (from the Freesurfer pipeline or Synthseg) for your dataset, or not. We will assume here that you have one and that it already is in the same space as the T1w image (or SWI image if you are running CMB segmentation alone). The parcellation **must** have a name of the form `*aparc+aseg.[nii | nii.gz | mgz]` to be properly recognized (e.g. `aparc+aseg.mgz` or `sub-01_aparc+aseg.nii.gz` would work).

- If you don't have such a parcellation on hand, you can:
  - Download, install, and run Freesurfer on your dataset (see the [Freesurfer website](https://surfer.nmr.mgh.harvard.edu/) for more info).
  - Or use our Synthseg integration that will compute everything for you. If you want to use our Synthseg integration with the Apptainer image, you will need to be a super-user (capable of doing `sudo` commands) on your computer during the build (so only once), as you will need to build a Synthseg Apptainer image. Everything related to this is described in the [Synthseg Apptainer image section](#synthseg-apptainer-image-optional).

### Software

- **Apptainer** installed ([Install Guide](https://apptainer.org/docs/user/main/quick_start.html))
- **Python 3.9** environment manager (e.g. `venv` or `conda`)
- **ANTs** toolbox ([original repository](http://stnava.github.io/ANTs/) or `conda install -c aramislab ants`)
- **Graphviz** ([classic install](https://graphviz.org/download/) or `conda install graphviz`)
- **dcm2niix** ([install guide](https://github.com/rordenlab/dcm2niix) or `conda install -c conda-forge dcm2niix`)
- Being an admin / super user on your machine during the Apptainer image build (one-time step).

---

## 2. Setting up SHiVAi

### SHiVAi Apptainer image

Download the `.sif` file from [cloud.efixia.com](https://cloud.efixia.com/sharing/bbWPx1QAZ) (about 4GB).

Let's assume you saved it as `~/myShivaiProject/shivai.sif`.

### Trained AI models

Download AI models from [pboutinaud GitHub](https://github.com/pboutinaud) as well as their JSON companion files.

- PVS:
  - `v1/T1-FLAIR.PVS`
  - `model_info_t1-flair-pvs-v3.json`
- WMH:
  - `v1/T1-FLAIR.WMH`
  - `model_info_t1-flair-wmh-v2.json`

> Notes:\
> Here we only mention the PVS and WMH models as an example; feel free to download all the models you need.

Place the model archives in a single folder (let's call it `shivai_models`). Extract the models from the downloaded archives and place each JSON file in its corresponding model folder (e.g. `model_info_t1-flair-wmh-v2.json` in `T1-FLAIR.WMH`).

You should then have something like this:

```txt
~/myShivaiProject/shivai_models/
                    ├── T1-FLAIR.PVS/
                    │   ├── model_info_t1-flair-pvs-v3.json
                    │   ├── ...
                    │   ...
                    ├── T1-FLAIR.WMH/
                    │   ├── model_info_t1-flair-wmh-v2.json
                    │   ├── ...
                    │   ...
                    ...
```

### Brain masking model (if needed)

If you do not have brain masks on hand, SHiVAi can create them automatically using a dedicated AI model. Download it from [cloud.efixia.com/sharing/Mn9LB5mIR](https://cloud.efixia.com/sharing/Mn9LB5mIR) and extract it into the same `shivai_models` folder.

### Synthseg Apptainer image (optional)

If you want to use our Synthseg parcellation pipeline, fully integrated in the process, follow the steps in this section. If you already have your own parcellation, you can skip this.

Create a folder for Synthseg-related files (e.g. `~/myShivaiProject/synthseg/`).

Download the [Synthseg Apptainer recipe](../apptainer/apptainer_synthseg_tf.recipe) and the [precomp_synthseg.py script](../apptainer/precomp_synthseg.py), and put them in `~/myShivaiProject/synthseg/`.

Download the Synthseg models from the [MIT file-sharing system](https://mitprod-my.sharepoint.com/:u:/g/personal/bbillot_mit_edu/Ebqxo6YgUmBJkOML0m8NSXgBrhaHG7iqClFXRXPinS6FGw?e=DzKf1p).

> Notes:\
> If this link fails, check [Benjamin Billot's repository on Github](https://github.com/BBillot/SynthSeg), and specifically [the issues page](https://github.com/BBillot/SynthSeg/issues) to see if others have encountered the same problem.

Put the model files in the synthseg folder. You should now have:

```txt
~/myShivaiProject/synthseg/
                    ├── apptainer_synthseg_tf.recipe
                    ├── precomp_synthseg.py
                    └── SynthSeg_models.zip
```

Build the Apptainer image. In a terminal, from the synthseg folder run:

```bash
apptainer build synthseg.sif apptainer_synthseg_tf.recipe
```

> Notes:\
> You will need to be an admin / super user to do this, so you may need `sudo` in front of the command.

This will generate the `synthseg.sif` image.

### Prepare the Configuration file

Download [`config_example.yml`](../apptainer/config_example.yml) to your project directory and optionally rename it (e.g. `config.yml`). Edit it with a text editor:

- Set `model_path` to your models folder (e.g. `~/myShivaiProject/shivai_models`)
- Set `apptainer_image` to the SHiVAi `.sif` file (e.g. `~/myShivaiProject/shivai.sif`)
- (Opt.) Set `synthseg_image` to the Synthseg `.sif` file (e.g. `~/myShivaiProject/synthseg/synthseg.sif`)
- Set descriptor paths for AI model `.json` files relative to the models folder

In our case, the file should look something like this:

```yml
model_path: ~/myShivaiProject/shivai_models
apptainer_image: ~/myShivaiProject/shivai.sif
synthseg_image: ~/myShivaiProject/synthseg/synthseg.sif

parameters:
  brainmask_descriptor:
  PVS_descriptor:
  PVS2_descriptor: "T1.FLAIR-PVS/model_info_t1-flair-pvs-v3.json"
  WMH_descriptor: "T1.FLAIR-WMH/model_info_t1-flair-wmh-v2.json"
  CMB_descriptor:
  LAC_descriptor:
  swi_echo: 1
  percentile: 99.0
  threshold: 0.5
  threshold_pvs: 0.5
  threshold_wmh: 0.2
  threshold_cmb: 0.5
  threshold_lac: 0.2
  min_pvs_size: 5
  min_wmh_size: 1
  min_cmb_size: 1
  min_lac_size: 3
  final_dimensions: [160, 214, 176]
  voxels_size: [1.0, 1.0, 1.0]
  voxels_tolerance: [0, 0, 0]
  interpolation: 'WelchWindowedSinc'
```

> The `apptainer_image` entry is **required** for the mixed approach because SHiVAi will delegate the TensorFlow-based steps (AI inference, and optionally brain masking) to the Apptainer container while running everything else locally.

### Install the SHiVAi Python Package

Create a Python 3.9 virtual environment, clone or download the SHiVAi project, and from the project directory (containing `pyproject.toml`) run:

```bash
python3.9 -m venv ~/shivai_env
source ~/shivai_env/bin/activate
pip install .
```

> Notes:\
> Unlike the traditional install, you do **not** need to install TensorFlow or CUDA locally — these are handled by the Apptainer image.\
> Feel free to use any Python environment manager (e.g. `conda`). The `venv` example above is common and lightweight.

The `shiva` command line tool will then be available from within the activated environment.

---

## 3. Running SHiVAi

### Organize Input Data

Structure your MRI data in **BIDS format**:

- Each subject in its own folder (e.g. `sub-01`, `sub-02`).
- Modalities (T1, FLAIR, etc.) in the `anat` subfolder.
- Expected file name format: `{subject id}_{acquisition type}*.nii(.gz)`.
- Each subject id must start with `sub-`.
- Each file must start with the subject id exactly as written in the folder name.
- Each file must have their modality written in **upper-case** in their name.

As explained earlier, if you are providing a FreeSurfer-type parcellation, it must be named `*aparc+aseg.[nii | nii.gz | mgz]` and placed in the same `anat` folder.

**Example folder structure (with available parcellation):**

```txt
~/myShivaiProject/BIDS_dataset/
                    ├── sub-01/
                    │   └── anat/
                    │       ├── aparc+aseg.mgz
                    │       ├── sub-01_T1w.nii.gz
                    │       └── sub-01_FLAIR.nii.gz
                    ├── sub-02/
                    │   └── anat/
                    │       ├── aparc+aseg.mgz
                    │       ├── sub-02_T1w.nii.gz
                    │       └── sub-02_FLAIR.nii.gz
                    └── ...
```

> Notes:\
> Alternatively, you can use the "standard" file structure, which is less stringent on filenames. See the [dedicated section in the main read-me](../README.md/#data-structures-accepted-by-shivai).

### Run SHiVAi Processing

Activate the environment and use the `shiva` command directly (no `run_shiva.py` script needed):

```bash
source ~/shivai_env/bin/activate
shiva --help   # optional, to verify the install and explore all options
```

**Example command:**

1. With available parcellation

```bash
shiva \
  --in ~/myShivaiProject/BIDS_dataset \
  --out ~/myShivaiProject/shivai_results \
  --input_type BIDS \
  --prediction PVS2 WMH \
  --brain_seg fs_precomp \
  --containerized_nodes \
  --config ~/myShivaiProject/config.yml
```

1. Using our Synthseg integration

```bash
shiva \
  --in ~/myShivaiProject/BIDS_dataset \
  --out ~/myShivaiProject/shivai_results \
  --input_type BIDS \
  --prediction PVS2 WMH \
  --brain_seg synthseg \
  --containerized_nodes \
  --config ~/myShivaiProject/config.yml
```

The `--containerized_nodes` flag tells SHiVAi to delegate the nodes that require TensorFlow, CUDA, ANTs, or niimath to the Apptainer image, while the orchestration and non-deep-learning steps run locally. The path to the Apptainer image is read from the `apptainer_image` entry in your config file.

Replace the paths with your actual folders.

> Notes:\
> These commands run linearly by default. See the [Parallelization](#parallelization-optional) section below to take advantage of the mixed approach's key benefit.\
> If you want to restrict processing to certain subjects, check the `--sub_list` or `--sub_names` options (run `shiva --help` to see all options).

### Parallelization (optional)

One of the main advantages of the mixed approach over the fully contained process is the ability to **parallelize** the pipeline steps. SHiVAi uses Nipype under the hood, which supports several parallel execution plugins.

**Using local multiprocessing** (when multiple GPUs or CPUs are available on the same machine):

```bash
shiva \
  --in ~/myShivaiProject/BIDS_dataset \
  --out ~/myShivaiProject/shivai_results \
  --input_type BIDS \
  --prediction PVS2 WMH \
  --brain_seg synthseg \
  --containerized_nodes \
  --config ~/myShivaiProject/config.yml \
  --run_plugin MultiProc
```

> Note: multiprocessing may cause issues on some systems.

**Using SLURM** (on HPC clusters):

```bash
shiva \
  --in ~/myShivaiProject/BIDS_dataset \
  --out ~/myShivaiProject/shivai_results \
  --input_type BIDS \
  --prediction PVS2 WMH \
  --brain_seg synthseg \
  --containerized_nodes \
  --config ~/myShivaiProject/config.yml \
  --run_plugin SLURM
```

In the SLURM case, the config file is **required** as it holds the path to the Apptainer image used by the compute nodes.

---

## 4. Results

Results will be saved in the `results/` folder inside your output directory (i.e. `~/myShivaiProject/shivai_results/results/`).

**Example:**

```txt
~/myShivaiProject/shivai_results/
                  └── results/
                    ├── results_summary/
                    │   ├── wf_graph/
                    │   │   └── graph.svg
                    │   ├── segmentations/
                    │   │   ├── pvs_metrics/
                    │   │   │   └── prediction_metrics.csv
                    │   │   └── wmh_metrics/
                    │   │       └── ...
                    │   └── preproc_qc/
                    │       ├── qc_metrics.csv
                    │       ├── qc_metrics_plot.svg
                    │       └── failed_qc.json
                    ├── shiva_preproc/
                    │   ├── t1_preproc/
                    │   ├── synthseg/
                    │   └── qc_metrics/
                    ├── segmentation/
                    │   ├── pvs_segmentation/
                    │   │   └── sub-01/
                    │   │       ├── pvs_map.nii.gz
                    │   │       ├── labelled_pvs.nii.gz
                    │   │       ├── pvs_census.csv
                    │   │       └── pvs_stats.csv
                    │   └── wmh_segmentation/
                    │       └── ...
                    └── report/
                        └── sub-01/
                            └── Shiva_report.pdf
```

Each subject will have a PDF report containing statistics and figures at `results/report/{participant_ID}/Shiva_report.pdf`.

The `segmentation` folder contains all segmented cSVD results and csv statistics files. The `*_map.nii.gz` files contain raw prediction maps (no filter, no binarization), while the `labelled_*.nii.gz` files contain the cleaned predictions with each cluster labeled by a number (which can be linked to the related census CSV file).

---

**Notes:**

- For the mixed approach, the `--containerized_nodes` flag is the key argument that routes GPU-intensive steps through the Apptainer image while keeping the rest local.
- For the fully contained (non-parallel) approach, see the [step_by_step README](../step_by_step/README.md).
- For advanced options or troubleshooting, see the full [README](../README.md).
