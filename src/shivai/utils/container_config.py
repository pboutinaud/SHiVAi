"""Utility functions for configuring container nodes in workflows.

Provides a unified way to set container-specific inputs (bind mounts,
image, GPU flags) on workflow nodes regardless of the container runtime
(Singularity/Apptainer or Docker).
"""


def configure_container_node(node, runtime, image, binds, gpu=True, synthseg_image=None):
    """Set container-specific inputs on a workflow node.

    Parameters
    ----------
    node : nipype Node
        The workflow node to configure.
    runtime : str
        Container runtime type: 'singularity' or 'docker'.
    image : str
        Container image path (.sif for Singularity) or name:tag (for Docker).
    binds : list of tuples
        Bind mount specifications as (src, dest, mode) tuples.
    gpu : bool, optional
        Whether to enable GPU support. Default True.
    synthseg_image : str, optional
        If provided, use this image instead of the main image
        (used for SynthSeg nodes which may use a different container).
    """
    actual_image = synthseg_image if synthseg_image else image
    node.inputs.container_runtime = runtime
    node.inputs.container_image = actual_image
    node.inputs.container_bind = binds
    if gpu:
        node.inputs.container_enable_nvidia = True
