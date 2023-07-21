"""Custom nipype interfaces for image resampling/cropping and
other preliminary tasks"""
import os
import nibabel.processing as nip
import nibabel as nb
import numpy as np
import csv
import weasyprint
import matplotlib.pyplot as plt
from bokeh.io import export_png

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, TraitedSpec
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename
from shivautils.image import normalization, crop, threshold, reverse_crop, make_offset, apply_mask
from shivautils.stats import metrics_prediction, make_report, get_mask_regions
from shivautils.quantification_WMH_Ventricals_Maps import create_distance_map


class ConformInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply conform function on the
    image """
    img = traits.File(exists=True, desc='NIfTI formated input image to conform to agiven shape',
                      mandatory=True)

    dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                              default=(256, 256, 256),
                              usedefault=True,
                              desc='The minimal array dimensions for the'
                              'intermediate conformed image.')
    
    order = traits.Int(3, desc="Order of spline interpolation", usedefault=True)

    voxel_size = traits.Tuple(float, float, float,
                              desc='resampled voxel size',
                              mandatory=False)

    orientation = traits.Enum('RAS', 'LAS',
                              'RPS', 'LPS', 
                              'RAI', 'LPI', 
                              'RPI', 'LAP',
                              'RAP',
                              desc="orientation of image volume brain",
                              usedefault=True)


class ConformOutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nb.Nifti1Image): transformed image
    """
    resampled = traits.File(exists=True,
                            desc='Image conformed to the required voxel size and shape.')


class Conform(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            NIfTI image file to process
            dimensions (int, int, int): minimal dimension
            voxel_size (float, float, float): Voxel size of final image
            orientation (string): orientation of the volume brain
        output_spec (nb.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            conform image to desired voxel sizes and dimensions
    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
                               function
        Return: runtime
        """
        fname = self.inputs.img
        img = nb.funcs.squeeze_image(nb.load(fname))
        if not (isdefined(self.inputs.voxel_size)):
            # resample so as to keep FOV
            voxel_size = np.divide(np.multiply(img.header['dim'][1:4], img.header['pixdim'][1:4]).astype(np.double),
                                   self.inputs.dimensions)
        else:
            voxel_size = self.inputs.voxel_size
        resampled = nip.conform(img, 
                                out_shape=self.inputs.dimensions,
                                voxel_size=voxel_size, 
                                order=self.inputs.order,
                                cval=0.0,
                                orientation=self.inputs.orientation,
                                out_class=None)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(resampled, base + 'resampled.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["resampled"] = os.path.abspath(base +
                                               'resampled.nii.gz')
        return outputs


class IntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply normalization to the nifti
    image"""
    input_image = traits.File(exists=True, desc='NIfTI image input.',
                              mandatory=True)

    percentile = traits.Float(exists=True, desc='value to threshold above this'
                              'percentile',
                              mandatory=True)
                              
    brain_mask = traits.File(desc='brain_mask to adapt normalization to'
    			             'the greatest number', mandatory=False)


class IntensityNormalizationOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): file img IRM nifti transformed
    """
    intensity_normalized = traits.File(exists=True,
                                       desc='Intensity normalized image')

    report = traits.File(exists=True, 
                         desc='html file with histogram voxel value,'
                         'mode and percentile value')

    mode = traits.Float(exists=True,
                        desc='Most frequent value of intensities voxel'
                        'histogram in an interval given')


class Normalization(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image): NIfTI image input
        output_spec (nb.Nifti1Image): Intensity-normalized image

    Methods:
        _run_interface(runtime):
            transformed an image into another with specified arguments
    """
    input_spec = IntensityNormalizationInputSpec
    output_spec = IntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # conform image to desired voxel sizes and dimensions
        fname = self.inputs.input_image
        img = nb.load(fname)
        if self.inputs.brain_mask:
            brain_mask = nb.load(self.inputs.brain_mask)
        else:
            brain_mask = self.inputs.brain_mask
        img_normalized, report, mode = normalization(img,
                                                     self.inputs.percentile,
                                                     brain_mask)
        
        # Save it for later use in _list_outputs
        setattr(self, 'mode', mode)
        with open('report.html', 'w', encoding='utf-8') as fid:
            fid.write(report)

        _, base, _ = split_filename(fname)
        setattr(self, 'report', os.path.abspath('report.html'))
        nb.save(img_normalized, base + '_img_normalized.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        outputs['report'] = getattr(self, 'report')
        fname = self.inputs.input_image
        _, base, _ = split_filename(fname)
        outputs['mode'] = getattr(self, 'mode')
        outputs["intensity_normalized"] = os.path.abspath(base +
                                                          '_img_normalized.nii.gz'
                                                          )
        return outputs


class ThresholdInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    img = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    threshold = traits.Float(0.5, exists=True, mandatory=True,
                             desc='Value of the treshold to apply to the image'
                            )
    
    sign = traits.Enum('+', '-',
                       usedefault=True,
                       desc='Whether to keep data above threshold or below threshold.')

    binarize = traits.Bool(False, exists=True,
                            desc='Binarize image')


class ThresholdOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): file img IRM nifti transformed
    """
    thresholded = traits.File(exists=True,
                              desc='Thresholded image')


class Threshold(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            file img Nifti
            Threshold for the brain mask
        output_spec (nb.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            transformed an image into brain mask
    """
    input_spec = ThresholdInputSpec
    output_spec = ThresholdOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        fname = self.inputs.img
        img = nb.funcs.squeeze_image(nb.load(fname))
        thresholded = threshold(img,
                                self.inputs.threshold,
                                sign=self.inputs.sign,
                                binarize=self.inputs.binarize)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(thresholded, base + '_thresholded.nii.gz')

        return runtime

    def _list_outputs(self):
        """
        Just gets the absolute path to the scheme file name
        """
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["thresholded"] = os.path.abspath(base +
                                                 '_thresholded.nii.gz')
        return outputs


class CropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply cropping to the
    nifti image"""
    roi_mask = traits.File(exists=True,
                           desc='Mask for computation of center of gravity and'
                           'cropping coordinates', mandatory=False)

    apply_to = traits.File(exists=True, 
                           desc='Image to crop', mandatory=True)

    final_dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                    default=(160, 214, 176),
                                    usedefault=True,
                                    desc='Final image array size in i, j, k.')
                                   
    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
    			           desc='center of gravity of nifti image cropped with first' 
    			           'voxel intensities normalization', mandatory=False)
    
    default = traits.Enum("ijk", "xyz", usedefault=True, desc="Default crop center strategy (voxels or world).")


class CropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): Cropped image
    """
    cropped = traits.File(exists=True,
                          desc='nb.Nifti1Image: preprocessed image')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc='center of gravity brain_mask')

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')


class Crop(BaseInterface):
    """Transform an image to desired dimensions."""
    input_spec = CropInputSpec
    output_spec = CropOutputSpec

    def _run_interface(self, runtime):
        """Run crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        if  isdefined(self.inputs.roi_mask):
            mask = nb.load(self.inputs.roi_mask)
        else:
            # get from ijk
            mask = None
        if not isdefined(self.inputs.cdg_ijk):
            cdg_ijk = None
        else: 
            cdg_ijk = self.inputs.cdg_ijk
        target = nb.load(self.inputs.apply_to)

        # process
        cropped, cdg_ijk, bbox1, bbox2 = crop(
            mask,
            target,
            self.inputs.final_dimensions,
            cdg_ijk,
            self.inputs.default,
            safety_marger=5)
        cdg_ijk = tuple([cdg_ijk[0], cdg_ijk[1], cdg_ijk[2]])
        bbox1 = tuple([bbox1[0], bbox1[1], bbox1[2]])
        bbox2 = tuple([bbox2[0], bbox2[1], bbox2[2]])

        # Save it for later use in _list_outputs
        setattr(self, 'cdg_ijk_attr', cdg_ijk)
        setattr(self, 'bbox1_attr', bbox1)
        setattr(self, 'bbox2_attr', bbox2)
        _, base, _ = split_filename(self.inputs.apply_to)
        nb.save(cropped, base + '_cropped.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.apply_to
        outputs['cdg_ijk'] = getattr(self, 'cdg_ijk_attr')
        outputs['bbox1'] = getattr(self, 'bbox1_attr')
        outputs['bbox2'] = getattr(self, 'bbox2_attr')
        _, base, _ = split_filename(fname)
        outputs["cropped"] = os.path.abspath(base + '_cropped.nii.gz')

        return outputs
        
        
class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    apply_to = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    model = traits.Any(exists=True,
                    	desc='tensor flow model of pretrained brain_mask')


class ApplyMaskOutputSpec(TraitedSpec):
    """Output class

    Args:
        brain_mask (nb.Nifti1Image): brain mask of input NIfTI image
    """
    brain_mask = traits.File(exists=True,
                             desc='Segmented brain mask of NIfTI image ')


class ApplyMask(BaseInterface):
    """Main class

    Attributes:
        input_spec (nb.Nifti1Image):
            file img Nifti
            tensorflow model of pretrained brain mask
        output_spec (nb.Nifti1Image): brain mask for nifti image given

    Methods:

        _run_interface(runtime):
            apply a model of brain mask on a nifti image
    """
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        fname = self.inputs.apply_to
        apply_to = nb.load(fname)
        model = self.inputs.model
        # brain_mask = apply_mask(apply_to, model)
        brain_mask = None

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nb.save(brain_mask, base + '_brain_mask.nii.gz')

        return runtime

    def _list_outputs(self):
        """Get the absolute path to the output brain mask file."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        _, base, _ = split_filename(fname)
        outputs["brain_mask"] = os.path.abspath(base + '_brain_mask.nii.gz')

        return outputs   


class ReverseCropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply reverse cropping to the
    nifti cropped image"""
    original_img = traits.File(exists=True,
                               desc='reference image in the original space to modify the cropped image', 
                               mandatory=False)

    apply_to = traits.File(exists=True, 
                           desc='Image to move in the original shape ', mandatory=True)

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')


class ReverseCropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nb.Nifti1Image): Cropped image
    """
    reverse_crop = traits.File(exists=True,
                               desc='nb.Nifti1Image: image cropped in the original space')



class ReverseCrop(BaseInterface):
    """Re-transform an image into the originel dimensions."""
    input_spec = CropInputSpec
    output_spec = CropOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        original_img = nb.load(self.inputs.original_img)
        apply_to = nb.load(self.inputs.apply_to)

        # process
        reverse_img = reverse_crop(
            original_img,
            apply_to,
            self.inputs.bbox1,
            self.inputs.bbox2)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(self.inputs.apply_to)
        nb.save(reverse_img, base + '_reverse_img.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().get()
        fname = self.inputs.apply_to
        _, base, _ = split_filename(fname)
        outputs["reverse_img"] = os.path.abspath(base + '_reverse_img.nii.gz')

        return outputs     




class DataGrabberSlicerInputSpec(BaseInterfaceInputSpec):
    """Input parameter to datagrabber for 3D slicer workflow"""
    args = traits.Any(desc='dictionary with arguments for workflow',
                      mandatory=False
                     )

    subject = traits.String(desc='name of subject')


class DataGrabberSlicerOutputSpec(TraitedSpec):
    """Output class

    Args:
        raw (str): path of raw image
        seg (str): path of segmentation brain mask image
    """
    raw = traits.File(exists=True,
                      desc='path for raw image')
    seg = traits.File(exists=True,
                      desc='path for segmentation brain mask image')



class DataGrabberSlicer(BaseInterface):
    """DataGrabber to handle corresponding image for 3D slicer workflow"""
    input_spec = DataGrabberSlicerInputSpec
    output_spec = DataGrabberSlicerOutputSpec

    def _run_interface(self, runtime):
        """Run DataGrabber 3D Slicer

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        args = self.inputs.args
        subject = self.inputs.subject
        raw = os.path.join(args['files_dir'], args['all_files'][subject]['raw'])
        seg = os.path.join(args['files_dir'], args['all_files'][subject]['seg'])
        setattr(self, 'raw', raw)
        setattr(self, 'seg', seg)



    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().get()
        outputs['raw'] = getattr(self, 'raw')
        outputs['seg'] = getattr(self, 'seg')

        return outputs  
    


class MakeOffsetInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply shifting image with
    a random offset"""
    img = traits.File(exists=True,
                      desc='image use to shifted with a random offset', 
                      mandatory=False)
    
    offset = traits.Any(False,
                          desc="offset apply to images if specified")


class MakeOffsetOutputSpec(TraitedSpec):
    """Output class

    Args:
        shifted_img (nb.Nifti1Image): nifti image shifted with a random offset
    """
    shifted_img = traits.File(exists=True,
                            desc='nb.Nifti1Image: image shifted with a random offset')

    offset_number = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                 desc="voxel shift in image applied")


class MakeOffset(BaseInterface):
    """Re-transform an image into the originel dimensions."""
    input_spec = MakeOffsetInputSpec
    output_spec = MakeOffsetOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        img = nb.load(self.inputs.img)
        
        # process
        shifted_img, offset_number = make_offset(img, offset=self.inputs.offset)

        # Save it for later use in _list_outputs
        setattr(self, 'offset_number', offset_number)
        _, base, _ = split_filename(self.inputs.img)
        nb.save(shifted_img, base + '_shifted_img.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        outputs['offset_number'] = getattr(self, 'offset_number')
        _, base, _ = split_filename(fname)
        outputs["shifted_img"] = os.path.abspath(base + '_shifted_img.nii.gz')

        return outputs  
    


class MetricsPredictionsInputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    img = traits.File(exists=True,
                       desc='image use to shifted with a random offset', 
                       mandatory=False)
    
    subject_id = traits.Any(desc="id for each subject")

    nb_latventricles_clusters = traits.Int(mandatory=False,
                              desc="""Number of clusters around lateral ventricles (optional)""")


class MetricsPredictionsOutputSpec(TraitedSpec):
    """Output class

    Args:
        metrics_prediction_csv (csv): csv file with metrics about the cluster prediction
        file
    """
    metrics_predictions_csv = traits.Any(exists=True,
                            desc='csv file with metrics about each prediction')


class MetricsPredictions(BaseInterface):
    """Get cluster metrics about one prediction file"""
    input_spec = MetricsPredictionsInputSpec
    output_spec = MetricsPredictionsOutputSpec

    def _run_interface(self, runtime):
        """Run processing cluster metrics

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        path_images = self.inputs.img
        subject_id = self.inputs.subject_id
        nb_latventricles_clusters = None
        if self.inputs.nb_latventricles_clusters is not None:
            nb_latventricles_clusters = self.inputs.nb_latventricles_clusters

        # Création du fichier CSV dans le Sink
        with open("metrics_predictions.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Écriture de l'en-tête du fichier CSV
            writer.writerow(["Subject id", "Number of voxels", "Number of clusters", 
                             "Mean clusters size", "Median clusters size", "Minimal clusters size", 
                             "Maximal clusters size", "Number Lateral Ventricles Clusters"])

            img = nb.load(path_images)
            array_img = img.get_fdata()
            results = metrics_prediction(array_img, 0.5)
            # results = (sum_voxel_segmented, number_of_cluster, mean_size_cluster, 
            #            median_size_cluster, min_size_cluster, max_size_cluster)

            # Ajout des lignes de données
            writer.writerow([subject_id, results[0], results[1], 
                            results[2], results[3], results[4], 
                            results[5], nb_latventricles_clusters])

        setattr(self, 'metrics_predictions_csv', os.path.abspath("metrics_predictions.csv"))


    def _list_outputs(self):
        """File in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['metrics_predictions_csv'] = getattr(self, 'metrics_predictions_csv')
        return outputs
    

class JoinMetricsPredictionsInputSpec(BaseInterfaceInputSpec):
    """Input parameter to get metrics of prediction file"""
    csv_files = traits.List(exists=True,
                       desc='image use to shifted with a random offset', 
                       mandatory=False)


class JoinMetricsPredictionsOutputSpec(TraitedSpec):
    """Output class

    Args:
        metrics_prediction_csv (csv): csv file with metrics about each prediction
    """
    metrics_predictions_csv = traits.Any(exists=True,
                            desc='csv file with metrics about each prediction')


class JoinMetricsPredictions(BaseInterface):
    """Get metrics about each prediction file"""
    input_spec = JoinMetricsPredictionsInputSpec
    output_spec = JoinMetricsPredictionsOutputSpec

    def _run_interface(self, runtime):
        """Run join of all cluster metrics in
        one csv file

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        path_csv_files = self.inputs.csv_files

        # Create CSV file
        for i in path_csv_files:
            with open(i, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                # Read second row
                try:
                    next(reader)
                    second_row = next(reader)
                except Exception as e:
                    print("There is no secon row in source file.")
                    second_row = None

                # Close file source
                file.close()

            if os.path.isfile('metrics_predictions.csv'):
                exists = True

            else:
                exists = False
            
            with open('metrics_predictions.csv', mode='a', newline='', encoding='utf-8') as file:

                writer = csv.writer(file)

                if exists == False:
                    # Write header of CSV file
                    writer.writerow(["Subject id", "Number of voxels", "Number of clusters", 
                                    "Mean clusters size", "Median clusters size", "Minimal clusters size", 
                                    "Maximal clusters size", "Number Lateral Ventricles Clusters"])

                writer.writerow(second_row)

        setattr(self, 'metrics_predictions_csv', os.path.abspath("metrics_predictions.csv"))


    def _list_outputs(self):
        """File in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['metrics_predictions_csv'] = getattr(self, 'metrics_predictions_csv')
        return outputs  
    


class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    """Delete prediction outside space of brainmask"""
    segmentation = traits.File(exists=True,
                      desc='prediction file to change', 
                      mandatory=False)
    
    brainmask = traits.Any(False,
                          desc="brainmask reference to delete prediction voxels")


class ApplyMaskOutputSpec(TraitedSpec):
    """Output class

    Args:
        shifted_img (nb.Nifti1Image): prediction file filtered with brainmask file
    """
    segmentation_filtered = traits.File(exists=True,
                            desc='prediction file filtered with brainmask file')



class ApplyMask(BaseInterface):
    """Re-transform an image into the originel dimensions."""
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        segmentation = nb.load(self.inputs.segmentation)
        brainmask = nb.load(self.inputs.brainmask)
        
        # process
        segmentation_filtered = apply_mask(segmentation, brainmask)

        # Save it for later use in _list_outputs
        setattr(self, 'segmentation_filtered', segmentation_filtered)
        _, base, _ = split_filename(self.inputs.segmentation)
        nb.save(segmentation_filtered, base + 'segmentation_filtered.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.segmentation
        _, base, _ = split_filename(fname)
        outputs["segmentation_filtered"] = os.path.abspath(base + 'segmentation_filtered.nii.gz')

        return outputs  
    


class SummaryReportInputSpec(BaseInterfaceInputSpec):
    """Make summary report file in pdf format"""
    histogram_intensity = traits.Any(exists=True,
                      desc='histogram of final voxels intensity', 
                      mandatory=False)
    
    isocontour_slides_FLAIR_T1 = traits.File(False,
                          mandatory=False,
                          desc="quality control of coregistration isocontour slides FLAIR on T1")
    
    isocontour_slides_brainmask_T1 = traits.File(False,
                          mandatory=False,
                          desc="quality control of coregistration isocontour slides brainmask on T1")
    
    metrics_clusters = traits.File(desc="pandas array of metrics clusters")

    metrics_clusters_2 = traits.File(mandatory=False, 
                                     desc="""optionnal metris clusters csv file if necessary 
                                     (example : dualpreprocessing T1-FLAIR with pvs and wmh clusters metrics)""")


class SummaryReportOutputSpec(TraitedSpec):
    """Output class

    Args:
        summary_report (html): summary report for each subject
        summary_report (pdf): summary report for each subject
    """
    summary_report = traits.Any(exists=True,
                            desc='summary html report')
    
    summary = traits.Any(exists=True,
                            desc='summary pdf report')



class SummaryReport(BaseInterface):
    """Make a summary report of preprocessing and prediction"""
    input_spec = SummaryReportInputSpec
    output_spec = SummaryReportOutputSpec

    def _run_interface(self, runtime):
        """Run reverse crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        histogram_intensity = self.inputs.histogram_intensity
        isocontour_slides_FLAIR_T1 = self.inputs.isocontour_slides_FLAIR_T1
        isocontour_slides_brainmask_T1 = self.inputs.isocontour_slides_brainmask_T1
        metrics_clusters = self.inputs.metrics_clusters
        metrics_clusters_2 = None
        if self.inputs.metrics_clusters_2:
            metrics_clusters_2 = self.inputs.metrics_clusters_2
        
        # process
        summary_report = make_report(histogram_intensity, 
                                     isocontour_slides_FLAIR_T1,
                                     isocontour_slides_brainmask_T1, 
                                     metrics_clusters,
                                     metrics_clusters_2)

        with open('summary_report.html', 'w', encoding='utf-8') as fid:
            fid.write(summary_report)

        # Convertir le fichier HTML en PDF
        weasyprint.HTML('summary_report.html').write_pdf('summary.pdf')

        setattr(self, 'summary_report', os.path.abspath('summary_report.html'))
        setattr(self, 'summary', os.path.abspath('summary.pdf'))

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        outputs['summary_report'] = getattr(self, 'summary_report')
        outputs['summary'] = getattr(self, 'summary')
        
        return outputs  
    





class MaskRegionsInputSpec(BaseInterfaceInputSpec):
    """Filter voxels according to a specific regions"""
    img = traits.File(exists=True,
                      desc='Nifti file with segmented regions', 
                      mandatory=False)
    
    list_labels_regions = traits.Any(False,
                          desc="list of indices or labels regions")


class MaskRegionsOutputSpec(TraitedSpec):
    """Output class

    Args:
        mask_regions (nb.Nifti1Image): prediction file filtered with brainmask file
    """
    mask_regions = traits.File(exists=True,
                               desc='Nifti mask file filtered with specific regions')



class MaskRegions(BaseInterface):
    """Filter voxels according to a specific regions of brain"""
    input_spec = MaskRegionsInputSpec
    output_spec = MaskRegionsOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        img = nb.load(self.inputs.img)
        list_labels_regions = self.inputs.list_labels_regions
        
        # process
        mask_regions = get_mask_regions(img, list_labels_regions)

        # Save it for later use in _list_outputs
        setattr(self, 'mask_regions', mask_regions)
        _, base, _ = split_filename(self.inputs.img)
        nb.loadsave.save(mask_regions, base + 'mask_regions.nii.gz')


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["mask_regions"] = os.path.abspath(base + 'mask_regions.nii.gz')

        return outputs
    

class QuantificationWMHVentricalsInputSpec(BaseInterfaceInputSpec):
    """Compute metrics about prediction around lateral ventricles"""
    WMH = traits.File(exists=True,
                      desc='Nifti file prediction', 
                      mandatory=False)

    subject_id = traits.Any(desc="id for each subject")
    
    Leventrical_distance_maps = traits.Any(False,
                          desc="distance maps in millimeter to lateral ventrical")


class QuantificationWMHVentricalsOutputSpec(TraitedSpec):
    """Output class

    Args:
        csv_clusters_localization (csv): metrics of clusters around lateral ventricles
    """
    csv_clusters_localization = traits.File(exists=True,
                               desc='metrics of clusters in lateral ventricles')
    
    nb_latventricles_clusters = traits.Int(exists=True,
                                            desc='number of clusters localized 2 millimeters around lateral ventricles')



class QuantificationWMHVentricals(BaseInterface):
    """Compute metrics about prediction in lateral ventricles"""
    input_spec = QuantificationWMHVentricalsInputSpec
    output_spec = QuantificationWMHVentricalsOutputSpec

    def _run_interface(self, runtime):
        """Run mask specific regions

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        WMH = nb.load(self.inputs.WMH)
        Leventrical_distance_maps = nb.load(self.inputs.Leventrical_distance_maps)
        subject_id = self.inputs.subject_id
        
        # process
        dataframe_clusters_localization, nb_latventricles_clusters = create_distance_map(Leventrical_distance_maps, WMH, subject_id)        

        out_csv = 'WMH_clusters_localization.csv'
        with open(out_csv, 'w') as f:
            dataframe_clusters_localization.to_csv(f, index=False)
        
        # Save it for later use in _list_outputs
        setattr(self, 'csv_clusters_localization', os.path.abspath("WMH_clusters_localization.csv"))
        setattr(self, 'nb_latventricles_clusters', nb_latventricles_clusters)
        


    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.WMH
        _, base, _ = split_filename(fname)
        outputs["csv_clusters_localization"] = getattr(self, 'csv_clusters_localization')
        outputs["nb_latventricles_clusters"] = getattr(self, 'nb_latventricles_clusters')

        return outputs