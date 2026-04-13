"""Unified container command line runner interface.

Provides ContainerInputSpec and ContainerCommandLine that handle both
Singularity/Apptainer and Docker through a single ``container_runtime``
switch, eliminating the need for separate Singularity and Docker class
hierarchies.
"""

import os
import re

from nipype.interfaces.base import (
    traits,
    CommandLine,
    CommandLineInputSpec,
)
from nipype.interfaces.base.traits_extension import isdefined

from shivai import __file__ as SHIVAINIT

SHIVALOC = os.path.dirname(os.path.abspath(SHIVAINIT))


def realpath_binding(path_list):
    """Resolve real paths for bind mount specifications, handling `pwd` placeholders."""
    path_list_bis = [os.getcwd() if path == '`pwd`' else path for path in path_list]
    return [os.path.realpath(path_list_bis[0]), os.path.realpath(path_list_bis[1]), path_list_bis[2]]


# ---------------------------------------------------------------------------
# Unified ContainerInputSpec — replaces SingularityInputSpec & DockerInputSpec
# ---------------------------------------------------------------------------

class ContainerInputSpec(CommandLineInputSpec):
    """Unified container attributes and options.

    The ``container_`` prefix is used as a namespace to distinguish
    container arguments from the actual command arguments.  The
    ``container_runtime`` trait selects which backend is used.
    """

    container_runtime = traits.Enum(
        'singularity', 'docker', 'apptainer',
        argstr='%s',
        desc='Container runtime to use.',
        mandatory=True)

    container_image = traits.Str(
        mandatory=True,
        argstr='%s',
        position=-1,
        desc='Container image: .sif file path (Singularity) or name:tag (Docker)')

    container_bind = traits.List(
        traits.Tuple((traits.Str,
                      traits.Str,
                      traits.Enum('ro', 'rw'))),
        desc="Bind / volume mount specification. List of (src, dest, 'ro'|'rw') tuples.",
        argstr='--bind %s',
        position=0,
        mandatory=False)

    container_enable_nvidia = traits.Bool(
        False,
        argstr='--nv',
        desc='Enable NVIDIA GPU support (--nv for Singularity, --gpus all for Docker)',
        mandatory=False)

    container_no_net = traits.Bool(
        False,
        argstr='--nonet',
        desc='Disable network (--nonet for Singularity, --network none for Docker)',
        mandatory=False)

    container_working_directory = traits.Directory(
        argstr='--workdir %s',
        mandatory=False,
        position=1,
        desc='Working directory inside the container.')

    # --- Singularity-specific ---
    container_home = traits.Directory(
        os.getenv('HOME'),
        exists=True,
        argstr='--home %s',
        desc='Home folder (Singularity only)',
        mandatory=False)

    container_scratch = traits.String(
        argstr='--scratch %s',
        desc='Scratch directory within the container (Singularity only)',
        mandatory=False)

    container_disable_cache = traits.Bool(
        False,
        argstr='--disable-cache',
        desc="Don't use cache (Singularity only)",
        mandatory=False)

    # --- Docker-specific ---
    container_user = traits.Str(
        argstr='--user %s',
        desc='User ID mapping uid:gid (Docker only)',
        mandatory=False)

    container_tmpfs = traits.Str(
        argstr='--tmpfs %s',
        desc='Mount a tmpfs inside the container (Docker only)',
        mandatory=False)

    container_platform = traits.Str(
        argstr='--platform %s',
        desc='Platform specification, e.g. "linux/amd64" (Docker only)',
        mandatory=False)

    container_environ = traits.Dict(
        key_trait=traits.Str,
        value_trait=traits.Str,
        argstr='-e %s',
        desc='Environment variables to pass into the container (Docker only)',
        mandatory=False)

    container_enable_mps = traits.Bool(
        False,
        argstr='-e PYTORCH_ENABLE_MPS_FALLBACK=1',
        desc='Enable Apple Metal Performance Shaders support (Docker only, experimental)',
        mandatory=False)


# ---------------------------------------------------------------------------
# Unified ContainerCommandLine — replaces SingularityCommandLine & DockerCommandLine
# ---------------------------------------------------------------------------

class ContainerCommandLine(CommandLine):
    """Unified container exec command interface.

    The runtime (Singularity vs Docker) is selected by the
    ``container_runtime`` input trait.  All container-prefixed inputs
    (``container_*``) are formatted into the appropriate command-line
    flags automatically.
    """

    _container_prefix = 'container_'

    @property
    def _container_base_cmd(self):
        """Return the container base command depending on runtime."""
        runtime = getattr(self.inputs, 'container_runtime', None)
        if not isdefined(runtime) or runtime is None:
            raise ValueError('container_runtime must be set to "singularity", "apptainer", or "docker"')
        if runtime in ['singularity', 'apptainer']:
            return f'{runtime} exec'
        return 'docker run --rm'

    @property
    def cmdline(self):
        """Build the full command line: container command + container args + actual command."""
        self._check_mandatory_inputs()
        result = [self._container_base_cmd] + self._container_parse_inputs()
        result.append(super(ContainerCommandLine, self).cmdline)
        return ' '.join(result)

    def _get_environ(self):
        """Override _get_environ to automatically add the dummy scripts to the PATH."""
        dummy_bin = os.path.join(SHIVALOC, 'scripts', 'snglrt_dummy_bin')
        env = getattr(self.inputs, "environ", {})
        if env and "PATH" in env:
            path_in = env.get("PATH")
        else:
            path_in = os.getenv("PATH", os.defpath)
        path = path_in + os.pathsep + dummy_bin
        env['PATH'] = path
        return env

    # ------------------------------------------------------------------
    # Runtime-aware argument formatting
    # ------------------------------------------------------------------

    def _container_format_arg(self, name, spec, value):
        """Format a single container-prefixed argument for the active runtime."""
        runtime = self.inputs.container_runtime

        # --- bind / volume mounts ---
        if name == 'container_bind':
            if runtime in ['singularity', 'apptainer']:
                return '--bind ' + ','.join(
                    ':'.join(realpath_binding(b)) for b in value)
            # Docker: one -v flag per mount
            parts = []
            for bind in value:
                resolved = realpath_binding(bind)
                mount_str = f'{resolved[0]}:{resolved[1]}'
                if resolved[2] == 'ro':
                    mount_str += ':ro'
                parts.append(f'-v {mount_str}')
            return ' '.join(parts)

        # --- GPU ---
        if name == 'container_enable_nvidia':
            if not value:
                return None
            return '--nv' if runtime in ['singularity', 'apptainer'] else '--gpus all'

        # --- network ---
        if name == 'container_no_net':
            if not value:
                return None
            return '--nonet' if runtime in ['singularity', 'apptainer'] else '--network none'

        # --- working directory ---
        if name == 'container_working_directory':
            flag = '--workdir' if runtime in ['singularity', 'apptainer'] else '-w'
            return f'{flag} {value}'

        # --- Singularity-only traits (skip silently for Docker) ---
        if name == 'container_scratch':
            if runtime not in ['singularity', 'apptainer']:
                return None
            return f'--scratch {value}'

        if name == 'container_disable_cache':
            if runtime not in ['singularity', 'apptainer'] or not value:
                return None
            return '--disable-cache'

        # container_home has no argstr — not emitted on the command line
        if name == 'container_home':
            return None

        # --- Docker-only traits (skip silently for Singularity) ---
        if name == 'container_user':
            if runtime != 'docker' or not value:
                return None
            return f'--user {value}'

        if name == 'container_tmpfs':
            if runtime != 'docker':
                return None
            return f'--tmpfs {value}'

        if name == 'container_platform':
            if runtime != 'docker':
                return None
            return f'--platform {value}'

        if name == 'container_environ':
            if runtime != 'docker':
                return None
            parts = [f'-e {k}={v}' for k, v in value.items()]
            return ' '.join(parts)

        if name == 'container_enable_mps':
            if runtime != 'docker' or not value:
                return None
            return '-e PYTORCH_ENABLE_MPS_FALLBACK=1'

        # container_runtime itself — not emitted
        if name == 'container_runtime':
            return None

        # fallback
        return super(ContainerCommandLine, self)._format_arg(name, spec, value)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _container_parse_inputs(self, skip=None):
        """Parse all container-prefixed inputs into command-line fragments.

        For Docker, auto-injects ``--user uid:gid`` when not explicitly set.
        """
        prefix_pattern = f'^{re.escape(self._container_prefix)}.*'
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            if not re.match(prefix_pattern, name):
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)

            if not isdefined(value):
                continue
            arg = self._container_format_arg(name, spec, value)
            if arg is None:
                continue
            pos = spec.position
            if pos is not None:
                if int(pos) >= 0:
                    initial_args[pos] = arg
                else:
                    final_args[pos] = arg
            else:
                all_args.append(arg)
        first_args = [el for _, el in sorted(initial_args.items())]
        last_args = [el for _, el in sorted(final_args.items())]
        result = first_args + all_args + last_args

        # Auto-inject --user uid:gid for Docker when not explicitly provided
        if self.inputs.container_runtime == 'docker':
            user_val = getattr(self.inputs, 'container_user', None)
            if user_val is None or not isdefined(user_val) or not user_val:
                try:
                    uid = os.getuid()
                    gid = os.getgid()
                    result.insert(0, f'--user {uid}:{gid}')
                except AttributeError:
                    pass  # os.getuid() not available (e.g. Windows)

        # Auto-inject working directory for Docker when not explicitly provided.
        # Some tools (e.g. quickshear) write outputs using relative paths;
        # without an explicit -w flag the container defaults to / where the
        # mapped user has no write permission.
        if self.inputs.container_runtime == 'docker':
            wd_val = getattr(self.inputs, 'container_working_directory', None)
            if wd_val is None or not isdefined(wd_val) or not wd_val:
                cwd = os.path.realpath(os.getcwd())
                result.insert(0, f'-w {cwd}')

        return result

    def _parse_inputs(self, skip=None):
        """Parse non-container inputs using their ``argstr`` format string.

        File paths are converted to their real path to avoid mount issues.
        """
        prefix_pattern = f'^{re.escape(self._container_prefix)}.*'
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            if re.match(prefix_pattern, name):
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)

            if not isdefined(value):
                continue
            if spec.is_trait_type(traits.File):
                value = os.path.realpath(os.getcwd() if value == '`pwd`' else value)
            arg = self._format_arg(name, spec, value)
            if arg is None:
                continue
            pos = spec.position
            if pos is not None:
                if int(pos) >= 0:
                    initial_args[pos] = arg
                else:
                    final_args[pos] = arg
            else:
                all_args.append(arg)
        first_args = [el for _, el in sorted(initial_args.items())]
        last_args = [el for _, el in sorted(final_args.items())]
        return first_args + all_args + last_args

    # Legacy aliases for backward compatibility
    def _singularity_format_arg(self, name, spec, value):
        return self._container_format_arg(name, spec, value)

    def _singularity_parse_inputs(self, skip=None):
        return self._container_parse_inputs(skip=skip)


# ---------------------------------------------------------------------------
# Backward-compatible aliases — these names are deprecated but still importable
# ---------------------------------------------------------------------------
SingularityInputSpec = ContainerInputSpec
SingularityCommandLine = ContainerCommandLine
DockerInputSpec = ContainerInputSpec
DockerCommandLine = ContainerCommandLine
