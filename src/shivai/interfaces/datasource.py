from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec,
                                    BaseInterface,
                                    isdefined)



class Direct_File_Provider_InputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Str(mandatory=True, desc='Dummy argument')
    img1 = traits.Any(exists=True, mandatory=True)  # Using any as can be File (for nifti) or Directory (for dcm)
    img2 = traits.Any(exists=True, mandatory=False)
    img3 = traits.Any(exists=True, mandatory=False)
    seg = traits.File(exists=True, mandatory=False)
    synthseg_vol = traits.File(exists=True, mandatory=False)
    synthseg_qc = traits.File(exists=True, mandatory=False)


class Direct_File_Provider_OutputSpec(TraitedSpec):
    img1 = traits.Any(exists=True, mandatory=True)
    img2 = traits.Any(exists=True, mandatory=False)
    img3 = traits.Any(exists=True, mandatory=False)
    seg = traits.File(exists=True, mandatory=False)
    synthseg_vol = traits.File(exists=True, mandatory=False)
    synthseg_qc = traits.File(exists=True, mandatory=False)


class Direct_File_Provider(BaseInterface):
    """Pass the input path to the ouput in order to replace a datagrabber when
    you want to directly pass the full path to the images (typically with SWOMed)"""

    input_spec = Direct_File_Provider_InputSpec
    output_spec = Direct_File_Provider_OutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for interface_trait in outputs:
            if isdefined(getattr(self.inputs, interface_trait)):
                outputs[interface_trait] = getattr(self.inputs, interface_trait)
        return outputs
