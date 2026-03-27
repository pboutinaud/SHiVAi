"""Docker command line runner interface class.

DEPRECATED: This module is kept for backward compatibility only.
All classes are now defined in shivai.interfaces.container.
"""

from shivai.interfaces.container import (  # noqa: F401
    ContainerInputSpec as DockerInputSpec,
    ContainerCommandLine as DockerCommandLine,
    realpath_binding,
)
