# Changelog

All notable changes to the SHiVAi project will be documented in this file.

## [v0.5.7] - 2026-03-25

### Added

- Basal ganglia + thalamus to deep WM for WMH statistics
- Initial transform to ANTs registration using center of mass
- Proper winner-takes-all logic for priority labels
- Batch size parameter to SHiVAi configuration
- Step-by-step README for the mixed approach
- Full mixed approach README

### Fixed

- FreeSurfer case handling for priority labels on WMH
- Sink node for workflow graph
- Recipes with new SynthSeg file name

### Changed

- Workflow graph is now optional (requires graphviz)

## [v0.5.6a] - 2026-03-09

### Changed

- Updated step-by-step README for fs_precomp
- Updated to use fs_precomp in fully contained mode
- Minor README updates
- Updated .sif download link

## [v0.5.6] - 2026-02-25

### Added

- T2 inversion support in SHiVAi
- New step-by-step README
- Controllable variable for bad affine threshold

### Fixed

- Improved affine correction to better generalize
- Affine correction taking into account dimension flips on any axis
- Debug fixes from Defacer: all possible orientation codes for Conform node, voxel size handling for morphology in threshold

### Changed

- Swapped Conform node to simpler CorrectAffine node for 2nd image prior to registration
- Reduced strictness of threshold for bad affine detection
- Updated SynthSeg recipe with new model info

## [v0.5.5] - 2026-01-08

### Fixed

- seg_cleaner when island is on the side of the image
- Interpolation in Conform cancelled when no voxel size change is needed
- FOV of FLAIR in dual preprocessing during Conform
- Wide CSV to include empty cells / missing values

### Added

- Conversion of label 77 to WM for FreeSurfer and SynthSeg

## [v0.5.4] - 2025-11-12

### Added

- Support for SavedModel-type models (folders instead of files) in prediction
- Fix from xu-boyan (GitHub issue) for BIDS support
- Version info in description

### Fixed

- Corrected pip call in Apptainer recipe
- Corrected broken link in README

## [v0.5.3] - 2025-06-04

### Added

- Voxel tolerance feature
- fs_precomp support in run_shiva
- Default "qc.csv" to SWOMed workflow
- Documentation for fs_precomp in README

### Fixed

- Binding for SynthSeg when using dcm2nii
- Minor debugging and refactoring

## [v0.5.2] - 2025-02-10

### Added

- FreeSurfer support for input segmentation
- Full Docker support for SynthSeg

### Fixed

- Pre-registered FLAIR by adding Resample_from_to node

### Changed

- Replaced SVG with PNG for PDF report compatibility inside containers
- Swapped os.path with pathlib.Path

## [v0.5.1] - 2025-02-03

### Added

- Support for pre-registered FLAIR images (prereg_flair argument)
- Docker support
- L/R orientation labels on brain image overlays in reports
- Documentation for Docker integration
- Full license text

### Changed

- SynthSeg volumes made optional
- Updated Dockerfile with new TensorFlow version
- Moved library requirements to external file (for Docker layers management)

## [v0.5.0] - 2025-01-20

### Changed

- **Major update**: now uses Keras 3 models
- Can run on CPU without GPU
- Updated library versions (TensorFlow and Keras)
- Updated predict_multi for Keras 3 models using pathlib.Path

### Fixed

- get_clusters_and_filter_image when filter has nothing to do
- Slurm sbatch_args
- CPU vs GPU in SHiVA brainmask node

### Added

- Wired SynthSeg QC
- Dummies for SingularityCommandLine

### Improved

- Morphology opening of binary masks for speed

## [v0.4.2] - 2024-11-26

### Added

- CPU inference with configurable thread count
- use_cpu option to predict and related methods

### Fixed

- bad_affine check for non-isotropic space

### Changed

- Each workflow instance runs in a different subject-specific directory
- Removed default values of args; args now override config.yml settings

## [v0.4.1] - 2024-11-05

### Added

- Option to run SynthSeg using local install even when containerized
- use_cpu argument to force CPU inference
- New predict_multi script and command line

### Fixed

- Realpath used in Singularity for path parsing
- nan_to_num added to intensity normalization
- predict_multi with Singularity

### Changed

- Updated container download link
- Set numpy version < 2.0
- License text updates with AGPL logo and acknowledgements
- Updated READMEs with structure corrections and details
- Removed problematic code in Conform interface

## [v0.4.0] - 2024-09-17

### Added

- New batch GPU prediction interface (Predict_Multi)
- Docker entrypoint and Dockerfile
- Suffix to output name for resample_from_to interface

### Changed

- Updated prediction workflow and main workflow with new prediction interface
- Removed old directories
- Updated download link for container

## [v0.3.13] - 2024-09-04

### Added

- SWOMed SynthSeg workflow (first complete version)
- Preprocessing-only mode
- Option to inverse image after intensity normalization
- SWI echo time configuration in config file
- File selector after dcm2nii for SWI (handles 2 generated files)
- All prediction combinations to SHiVAi node

### Fixed

- Crop when applying previous cdg
- Circular import issue
- Bug corrections for SWOMed integration

### Changed

- Renamed shivautils to shivai
- Improved external capsule segmentation
- Reorganized prep_json for importable function
- Major update for SWOMed: dicom/nifti handling, model descriptor management

## [v0.3.11] - 2024-04-26

### Added

- First version of SWOMed SHiVAi workflow
- All outputs to SHiVAi interface
- Conform can now ignore bad affine (e.g. for SHiVA masking)

### Changed

- Absent SHiVAi outputs stay undefined
