Shivai commands
-------------------
These scripts can be run from the shell prompt after installation of the package.

* **shiva.py**: main command-line tool and entry point for the containers
* **predict.py**: perform prediction on a preprocessed image
* **run_shiva.py**: wrapper script to run full-containerized pipelines over batches of subjects
* **prep_json.py**: utility script for model-weights-descriptors generation
* **download_helper.py** (`shiva_download`): download and extract AI models and their JSON descriptors, and generate a configuration file. For example: `shiva_download -m PVS2 WMH -o /path/to/models`. To use an existing container without installing SHiVAi locally, use [download_models.py](../../../apptainer/download_models.py).