# SHiVAi Quick Start: Fully Contained Apptainer Workflow (BIDS, Synthseg, PVS2/WMH/LAC)

This guide summarizes the essential steps to run SHiVAi using Apptainer in a fully contained process, with BIDS data structure, for bi-modal PVS and WMH segmentation, using the Synthseg masking scheme. Other segmentation (lacunes, CMB, mono-modal PVS) follow exactly the same process and can be added by adapting what you will find below with the proper models, available at the same place as the ones we will describe here.

In this guide, we will assume we are using storing everying in a folder called `~/myShivaiProject/` as an example.

---

## 1. Requirements

- **Linux machine** with GPU (16GB memory recommended) - GPU-only processing is *possible* but very slow.
- **Apptainer** installed ([Install Guide](https://apptainer.org/docs/user/main/quick_start.html))
- Being an admin / super user on your machine during the installation.

## 2. Setting up Shivai

### SHiVAi Apptainer image

Download `.sif` file from [cloud.efixia.com](https://cloud.efixia.com/sharing/t3jG8DICk)

### SHiVAi Launcher script

Download the [run_shiva.py script](../apptainer/run_shiva.py).

### Trained AI models

Download AI models from [pboutinaud GitHub](https://github.com/pboutinaud) as well as the JSON companion file.

- PVS:
  - v1/T1-FLAIR.PVS
  - model_info_t1-flair-pvs-v3.json
- WMH
  - v1/T1-FLAIR.WMH
  - model_info_t1-flair-wmh-v2.json

> Notes:
> Here we only talk about the PVS and WMH models as an example for the current read-me, but feel free to download all the models you need

Place the model file archive in a single folder (let's call it `shivai_models`). Extract the models from the downloaded archives. Place the json file in their corresponding model folder (i.e. "model_info_t1-flair-wmh-v1.json" in "T1-FLAIR.PVS").

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

### Synthseg Apptainer image

Create of folder where we will put Synthseg-related files (e.g. `~/myShivaiProject/synthseg/`).

Download the [Synthseg Apptainer recipe](../apptainer/apptainer_synthseg_tf.recipe) and the [precomp_synthseg.py script](../apptainer/precomp_synthseg.py), and put them in their own folder, say `~/myShivaiProject/synthseg/`.

Download the Synthseg models from the [MIT file-sharing system](https://mitprod-my.sharepoint.com/:u:/g/personal/bbillot_mit_edu/Ebqxo6YgUmBJkOML0m8NSXgBrhaHG7iqClFXRXPinS6FGw?e=DzKf1p).

> Notes:
> If this link fails, check [Benjamin Billot's repository on Github](https://github.com/BBillot/SynthSeg), and more specifically [the issues page](https://github.com/BBillot/SynthSeg/issues) to see of other people have encountered the same problem.

Put the model files in the above-mentioned synthseg folder.

You should now have:

```txt
~/myShivaiProject/synthseg/
                    ├── apptainer_synthseg_tf.recipe
                    ├── precomp_synthseg.py
                    └── SynthSeg_models.zip
```

You can now build the Apptainer image. In a terminal, from the synthseg folder we just created, do:

```bash
apptainer build synthseg.sif apptainer_synthseg_tf.recipe
```

> Notes:
> Beware, you will need to be an admin / super user to do this. So you may need to add `sudo` in front of the command (and have the rights to do so).

This will generate the `synthseg.sif` image that we will use.

### Prepare Configuration file

- Download [`config_example.yml`](../apptainer/config_example.yml) to your project directory and optionally remane it (e.g. `config.yml`):
- Edit `config.yml` with a text editor:
  - Set `model_path` to your models folder (e.g., `~/myShivaiProject/shivai_models`)
  - Set `apptainer_image` to your pipeline `.sif` file (e.g., `~/myShivaiProject/shivai.sif`)
  - Set `synthseg_image` to the Synthseg `.sif` file (e.g.  `~/myShivaiProject/synthseg/synthseg.sif`)
  - Set descriptor paths for AI models `.json` files relative to the models folder (e.g. `"T1.FLAIR-PVS/model_info_t1-flair-pvs-v3.json"` for PVS2_descriptor)

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

## Set Up Python Environment

- Create a minimal Python virtual environment
- Install `pyyaml` package

**Example commands:**

```bash
python3 -m venv ~/shivai_env
source ~/shivai_env/bin/activate
pip install pyyaml
```

> Notes:
> Feel free to use any Python environment manager. The one I gave here, `venv`, is common and lightweigh, but this can be done using any other one.
> If you don't have any Python distribution already installed on your machine, there is plenty of material available online that will guide you through this (even more so now with the advances of LLM-based chat-bots)

## 3. Running Shivai

### Organize Input Data

- Structure your MRI data in **BIDS format**
  - Each subject in its own folder (e.g., `sub-01`, `sub-02`).
  - Modalities (T1, FLAIR, etc.) in the `anat` subfolder.
  - Expected file name format: `{subject id}_{aquisition type}*.nii(.gz)`.
  - Each subject id must start with `sub-`.
  - Each file must start with the subject id exactly as written in the folder name.
  - Each file must have their modality written in **upper-case** in their name.

**Example folder structure:**

```txt
~/myShivaiProject/BIDS_dataset/
                    ├── sub-01/
                    │   └── anat/
                    │       ├── sub-01_T1w.nii.gz
                    │       └── sub-01_FLAIR.nii.gz
                    ├── sub-02/
                    │   └──  anat/
                    │       ├── sub-02_T1w.nii.gz
                    │       └── sub-02_FLAIR.nii.gz
                    └── ...
```

> Notes:
> Alternatively, you can also use what we called the "standard" file structure, less stringent on the filenames. See the [dedicated section in the main read-me](../README.md/#data-structures-accepted-by-shivai).

### Run SHiVAi Processing

- Use the provided `run_shiva.py` script we [downloaded earlier](#shivai-launcher-script). So open a terminal and go to the folder where you stored the script (e.g. ``cd ~/myShivaiProject` if you stored it with the other parts of the project we have worked on so far).
- Activate the python environment you've set up for Shivai (e.g. by doing `source ~/shivai_env/bin/activate`, like above)
- Optionally, you can test if the environment is set up correctly by doing `python3 run_shiva.py --help`. This will display all the available options for running Shivai using the `run_shiva.py` script.

**Example command:**

```bash
python run_shiva.py \
  --in ~/myShivaiProject/BIDS_dataset \
  --out ~/myShivaiProject/shivai_results \
  --input_type BIDS \
  --prediction PVS2 WMH \
  --brain_seg synthseg \
  --config ~/myShivaiProject/config.yml
```

Replace the paths with your actual folders.

> Notes:
> This will run Shivai on all the subjects available in the input folder. If you want to restrict this, check the `--sub_list` or `--sub_names` options (run `python3 run_shiva.py --help` to see all options and their function)
> If you get an error about missing permissions or a file path being having a problem because "`<class 'str'> was specified`", check that you have access to all files and folders, cehck that you gave the proper paths (e.g. no typo) check that the data structure is correct.

---

## 7. Results

- Results will be saved in the output folder under `results/`
- Includes segmentation maps, metrics, and PDF reports for each subject

**Example:**

After running, you will find:

```txt
~/myShivaiProject/shivai_results/
                  └── results/
                    ├── results_summary/
                    ├── segmentation/
                    ├── shiva_preproc/
                    └── ...
```

Each subject will have a PDF report containing statistics and figures in `results/report/{participant_ID}/Shiva_report.pdf`.

The `segmentation` folder will contain all the segmented cSVD and csv files with statistics. The `*_map.nii.gz` files contain raw prediction maps (no filter, no binarization) while the `labeled_*.nii.gz` files contain the cleaned prediction, whith each cluster labeled with a number (which can be linked to their related census csv file).

---

**Note:**

- Only the fully contained Apptainer workflow is described here. I runs linearly, which make it potentially slower than other modes. For better performence, check the [Mixed approach](../README.md/#mixed-approach-recommended) from the main read-me. It requires specific python envivornment with more installs, so you will need to know how to do this kind of things if you want to go this way.
- For advanced options or troubleshooting, see the full [README](../README.md)
