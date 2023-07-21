import os

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function

from shivautils.interfaces.image import MetricsPredictions, MaskRegions, QuantificationWMHVentricals
from shivautils.interfaces.shiva import SynthSeg
from shivautils.interfaces.interfaces_post_processing import MakeDistanceVentricleMap
from shivautils.quantification_WMH_Ventricals_Maps import create_distance_map


dummy_args = {'SUBJECT_LIST': ['BIOMIST::SUBJECT_LIST'],
              'BASE_DIR': os.path.normpath(os.path.expanduser('~'))}


def genWorkflow(**kwargs) -> Workflow:
    """Generate a nipype workflow

    Returns:
        workflow
    """
    workflow = Workflow("test_report_fonctionnalities")
    workflow.base_dir = kwargs['BASE_DIR']

    # get a list of subjects to iterate on
    subject_list = Node(IdentityInterface(
        fields=['subject_id'], mandatory_inputs=True), name="subject_list")
    subject_list.iterables = ('subject_id', kwargs['SUBJECT_LIST'])

    # file selection
    datagrabber = Node(DataGrabber(infields=['subject_id'],
                                   outfields=['t1_preproc', 'prediction']),
                       name='dataGrabber')
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True

    workflow.connect(subject_list, 'subject_id', datagrabber, 'subject_id')

    if kwargs['SYNTHSEG']:
        synthseg = Node(SynthSeg(), name='synthseg')
        synthseg.inputs.out_filename = 'segmentation_regions.nii.gz'

        workflow.connect(datagrabber, 't1_preproc', synthseg, 'i')

        mask_Latventrical_regions = Node(MaskRegions(), name='mask_Latventrical_regions')
        mask_Latventrical_regions.inputs.list_labels_regions = [4, 5, 43, 44]
        workflow.connect(synthseg, 'segmentation', mask_Latventrical_regions, 'img')

        # Creating a distance map for each ventricle mask
        MakeDistanceLeventricalMap_ = Node(MakeDistanceVentricleMap(), name="MakeDistanceVentricleMap")                
        MakeDistanceLeventricalMap_.inputs.out_file = 'distance_map.nii.gz'
        workflow.connect(mask_Latventrical_regions, 'mask_regions', MakeDistanceLeventricalMap_ , "im_file")

        WMH_Quantification_Leventrical = Node(QuantificationWMHVentricals(), name='WMH_Quantification_Leventrical')
        workflow.connect(datagrabber, 'prediction', WMH_Quantification_Leventrical, 'WMH')
        workflow.connect(MakeDistanceLeventricalMap_, 'out_file', WMH_Quantification_Leventrical, 'Leventrical_distance_maps')
        workflow.connect(subject_list, 'subject_id', WMH_Quantification_Leventrical, 'subject_id')


    metrics_predictions = Node(MetricsPredictions(subject_id=kwargs['SUBJECT_LIST']),
                                   joinsource = 'subjectList',
                                   joinfield= 'imgs',
                                   name="metrics_predictions")

    workflow.connect(datagrabber, 'prediction', metrics_predictions, 'img')
    workflow.connect(subject_list, 'subject_id', metrics_predictions, 'subject_id')
    if kwargs['SYNTHSEG']:
        workflow.connect(WMH_Quantification_Leventrical, 'nb_latventricles_clusters', metrics_predictions, 'nb_latventricles_clusters')


    datasink = Node(DataSink(), name='sink')
    datasink.inputs.base_directory = 'output'
    workflow.connect(WMH_Quantification_Leventrical, 'csv_clusters_localization', datasink, 'csv_clusters_localization')
    

    return workflow