# Dummy bin folder

This folder contains dummy scripts that are necessary for [`SingularityCommandLine`](../../interfaces/singularity.py) nodes to work.
Indeed, these nodes inherit from Nipype's `CommandLine`, which checks if the involved command line is in the local `PATH`, but may fail to do so if the software called by the `SingularityCommandLine` is only available in the container (which will crash the pipeline).

The dummy scripts are made available to the singularity/appatainer nodes by adding the snglrt_dummy_bin folder at the end of the `PATH` (only in the environment of the node).

All `SingularityCommandLine` interfaces must thus have their own dummy here to ensure their proper function in any environment.

> When adding new dummy scripts, don't forget to make them executable (`chmod +x your_script`) before the commit!