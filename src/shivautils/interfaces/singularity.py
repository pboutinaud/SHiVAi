"""Singularity command line runner interface class.

Created on 15 October 2019

@author: Pierre-Yves Herve
"""

import os
from nipype.interfaces.base import (
    traits,
    CommandLineInputSpec,
    CommandLine,
)
import re
try:
    # more recent nipype versions
    from nipype.interfaces.base.traits_extension import isdefined
except ModuleNotFoundError:
    from nipype.interfaces.traits_extension import isdefined

from nipype.interfaces.dcm2nii import (Dcm2niiInputSpec, Dcm2nii,
                                       Dcm2niiOutputSpec)


class SingularityInputSpec(CommandLineInputSpec):
    """Singularity attributes and options.

    Adds a field for the singularity image to be used.
    The "snglrt_" prefix is used as a namespace, so as to distinguish
    from arguments of the command line that will be executed through
    the singularity image. Those arguments should be added in
    child classes.
    """

    snglrt_image = traits.File(
        mandatory=True,
        argstr='%s',
        postion=-1,
        exists=True,
        desc='Name of the singularity image file to use')

    snglrt_bind = traits.List(
        traits.Tuple((traits.Str,
                      traits.Str,
                      traits.Enum('ro', 'rw'))),
        desc="a user-bind path specification.  A list of tuples of the form: "
             "2 strings and an enumeration, with src path, dest path and options 'ro' or 'rw'",
        argstr='--bind %s',
        position=0,
        mandatory=False)

    snglrt_home = traits.Directory(os.getenv('HOME'),
                                   exists=True,
                                   desc="home folder",
                                   mandatory=False)

    snglrt_no_net = traits.Bool(False,
                                argstr="--nonet",
                                mandatory=False,
                                desc="diasble network conection")

    snglrt_enable_nvidia = traits.Bool(False,
                                       argstr='--nv',
                                       desc='enable nvidia support',
                                       mandatory=False)

    snglrt_working_directory = traits.Directory(
        argstr='--workdir %s',
        mandatory=False,
        position=1,
        desc='Initial working directory for payload processing inside '
             'the container.')

    snglrt_scratch = traits.String(
        argstr='--scratch %s',
        desc='Two strings ; include a scratch directory within the container '
             'that is linked to a temporary dir (use -W to force location)',
        mandatory=False)

    snglrt_disable_cache = traits.Bool(
        False,
        argstr='--disable-cache',
        desc="dont use cache, and don't create cache",
        mandatory=False)


class SingularityCommandLine(CommandLine):
    """Singularity exec command base interface.

    This is meant to be extended by command line specific interfaces.

    """

    @property
    def cmdline(self):
        """Add a 'singularity exec' command to begining of actual command."""
        self._check_mandatory_inputs()
        result = ['singularity exec'] + self._singularity_parse_inputs()
        result.append(super(SingularityCommandLine, self).cmdline)
        return ' '.join(result)

    def _singularity_format_arg(self, name, spec, value):
        """Custom argument formating for singularity."""
        if name == 'snglrt_bind':
            return spec.argstr % (','.join([':'.join(b) for b in value]))
        return super(SingularityCommandLine, self)._format_arg(name,
                                                               spec,
                                                               value)

    def _singularity_parse_inputs(self, skip=None):
        """Parse all singularity inputs using the ``argstr`` format string.

        Any inputs that are assigned (not the default_value) are formatted
        to be added to the command line.
        Returns
        -------
        all_args : list
            A list of all inputs formatted for the command line.

        """
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            if not re.match('^snglrt_.*', name):
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)

            if not isdefined(value):
                continue
            arg = self._singularity_format_arg(name, spec, value)
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

    def _parse_inputs(self, skip=None):
        """Parse non-singularity inputs using their ``argstr`` format string.

        Any inputs that are assigned (not the default_value) are formatted
        to be added to the command line.

        Returns
        -------
        all_args : list
            A list of all inputs formatted for the command line.

        """
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            if re.match('^snglrt_.*', name):
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)

            if not isdefined(value):
                continue
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


class SgDcm2niiInputSpec(Dcm2niiInputSpec, SingularityInputSpec):
    """Double inheritance so as to create a singularity Dcm2nii class."""
    pass

# Example of double inheritance as mentionned just above...
# create a new singularity class that inherits from both singularity and the preexisting 
# command line wrapper, et voilà...
class SgDcm2nii(Dcm2nii, SingularityCommandLine):
    """To run dcm2nii through singularity."""

    def __init__(self):
        """Call parent constructor."""
        super(SgDcm2nii, self).__init__()

    input_spec = SgDcm2niiInputSpec
    output_spec = Dcm2niiOutputSpec
    _cmd = Dcm2nii._cmd
