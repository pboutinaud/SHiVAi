Shivai commands
-------------------
These scripts can be run from the shell prompt after installation of the package.

* **predict.py**: perform prediction on preprocessed image
* **module_wf_script.py**: perform preprocessing, predictions (WMH and PVS) and postprocessing (PDF report), starting from a simple file staging structure: *subject_id*/*main* or *acc*/*file.nii* with *main* folder usually the t1 and the *acc* folder containing another image
* **preprocess.py**: perform quick, simple preprocessing: conforming to 1mm isotropic resolution, cropping centered on top of head (no actual brain mask), intensity normalization [0..1]
* **script_wf**: runs the SHIVA workflow allowing for different data staging strategies
* **slicer_run_preprocessing.py**: mean to be used from a 3D Slicer extension