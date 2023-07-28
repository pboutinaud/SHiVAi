SHIVA preprocessing and deep learning segmentation workflows
===========================================================


Installation
------------

The SHIVA application requires a Linux machine with a GPU (16Gb memory), with the following dependencies:

1. Singularity (now known as AppTainer):
https://apptainer.org/docs/user/main/quick_start.html

2. Nvidia Container Toolkit (for GPUs): 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Once the dependencies are installed, you can build the container image.

3. As *root* user, from the **shivautils/singularity** directory, run the following command to build the SHIVA singularity container image :

 singularity build preproc_tensorflow.sif singularity_tf.recipe

 

Command line arguments (with run_shiva.py)
=========================================

    --in (path): path of the input dataset
    --out (path): the path where the generated files will be output
    --input_type (str):  Type of structure file, way to capture and manage nifti files : standard, BIDS or json
    -- model (path): Path to model descriptor (path_default : '/homes_unix/yrio/Documents/modele/ReferenceModels')
    --config (path): File with configuration options for workflow processing images, (path_default : /homes_unix/yrio/Documents/utils_python/shivautils/singularity/config_wmh.yml)


Command line example : 

    python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~/shiva_Nagahama_test_preprocessing_dual 
    --input_type standard --config ~/.shiva/config_dual.yml

SLURM command line example (with GPU resources configured): 

    srun –gpus 1 python run_shiva.py --in ~/Documents/data/VRS/Nagahama/raw/dataset_test --out ~shiva_Nagahama_test_preprocessing_dual --input_type standard --config ~/.shiva/config_wmh.yml


Options configuration in yaml file :

    --cuda (file) : bind mount path for the CUDA libs
    --gcc (file) : bind mount path for the gcc
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
    --voxels_size (str) : Voxel size of the final image, example : '1.0 1.0 1.0' --grab : data grabber
    --SWI (str) : if a second workflow for CMB is required, example : 'True' or 'False' 
    --interpolation (str): image resampling method, default interpolation : 'WelchWindowedSinc', others ANTS interpolation possibilities : 'Linear', 'NearestNeighbor', 'CosineWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'BSpline', 'MultiLabel', 'Gaussian', 'GenericLabel'

 


Example of BIDS structure folders :

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


Example of Json structure input :

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







Pipeline description
====================


SHIVA preprocessing for deep learning predictors. Perform resampling of a structural NIfTI head image, 
followed by intensity normalization, and cropping centered on the brain. A nipype workflow is used to 
preprocess a lot of images at the same time. In the last step the file segmentations with the wmh and 
pvs models are processed


Preprocessing
-------------

1. **Conform**: Resample image to Predictor dimensions (currently 160x214x176) by adjusting the voxel size

2. **Intensity Normalization**: We remove values above the 99th 'percentile' (parameter default value) to avoid hot spots,
    set values below 0 to 0, set values above 1.3 to 1.3 and normalize the data between 0 and 1

3. **Pre BrainMask Prediction**: Run predict to segment brainmask from resampled structural images with Tensorflow model

4. **Threshold**: Create a binarized brain_mask by putting all the values below the 'threshold' to 0

5. **Crop**: adjust the real-world referential and crop image. With the pre brainmask prediction, the procedure uses the center of mass of the mask to center the bounding box. If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine referential coordinates origin. If set to 'ijk', the middle of the image is used.

6. **Final Brainmask**: Run predict to segment brainmask from cropping images with Tensorflow model. 

7. **Coregistration** : Register a FLAIR images on T1w images either through the full interface to the ANTs registration method.


Reporting
---------

Individual CSV file (path folder 'subject_{id}/metrics_predictions_{pvs/wmh}'):

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

General CSV file (path folder 'metrics_predictions_{pvs/wmh}_generale'):

- Sum of metrics clusters with one arrow per subjects.


Detailed about python package
-----------------------------

Use the following command line to deploy the python package with 'setup.py' file : 

    python -m build

All of the scripts in the package run nipype workflows which are implemented as follows : 
    https://nipype.readthedocs.io/en/latest/api/generated/nipype.pipeline.engine.workflows.html

The package is designed with nipype interfaces present in the file path 'shivautils/interfaces/image.py'

Example of personnalized Nipype Interface :

    class ConformInputSpec(BaseInterfaceInputSpec):

        img = traits.File(exists=True)

        dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                default=(256, 256, 256))

        voxel_size = traits.Tuple(float, float, float,
                                desc='resampled voxel size',
                                mandatory=False)

    class ConformOutputSpec(TraitedSpec):

        resampled = traits.File(exists=True,
                                desc='Image conformed to the required voxel size and shape.')


    class Conform(BaseInterface):

        input_spec = ConformInputSpec
        output_spec = ConformOutputSpec

        def _run_interface(self, runtime):

            fname = self.inputs.img
            img = nb.funcs.squeeze_image(nb.load(fname))

            voxel_size = self.inputs.voxel_size
            resampled = nip.conform(img, 
                                    out_shape=self.inputs.dimensions,
                                    voxel_size=voxel_size)

            # Save it for later use in _list_outputs
            _, base, _ = split_filename(fname)
            nb.save(resampled, base + 'resampled.nii.gz')

            return runtime

        def _list_outputs(self):
            """Just get the absolute path to the scheme file name."""
            outputs = self.output_spec().get()
            fname = self.inputs.img
            _, base, _ = split_filename(fname)
            outputs["resampled"] = os.path.abspath(base +'resampled.nii.gz')
            return outputs


