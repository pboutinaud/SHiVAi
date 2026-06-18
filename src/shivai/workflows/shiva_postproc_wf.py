"""
Run the postprocessing from Shiva on arbitrary input data.
Typically expects a binarised cluster mask (from some predictor), a brain parcellation (or mask),
and the type of prediction corresponding to the cluster mask (e.g. WMH, PVS, CMB, LAC).
The postprocessing will then compute the different statistics for the clusters in the mask and output a CSV file with the results.
"""
import os
from nipype.pipeline import Workflow, Node
from nipype.interfaces.io import DataGrabber
from shivai.interfaces.datasink import DataSink_CSV_and_PDF_safe
from shivai.interfaces.image import (Threshold, Normalization, CorrectAffine,
                                     Conform, Crop, Resample_from_to,
                                     Parc_from_Synthseg, Segmentation_Cleaner,
                                     Regionwise_Prediction_metrics,
                                     Brain_Seg_for_biomarker,Label_clusters)

def genWorkflow(**kwargs) -> Workflow:
    workflow = Workflow(name="shiva_postproc_wf")
    workflow.base_dir = kwargs['BASE_DIR']
    datagrabber = Node(DataGrabber(
        infields=['subject_id'],
        outfields=['pred', 'seg']),
        name='datagrabber')
    datagrabber.inputs.base_directory = kwargs['DATA_DIR']
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s/%s/*.nii*'
    datagrabber.inputs.field_template = {'pred': '%s/pred/*.nii.gz',
                                         'seg': '%s/seg/*.nii.gz'}
    datagrabber.inputs.template_args = {'pred': [['subject_id']],
                                        'seg': [['subject_id']]}
    
    correct_pred = Node(CorrectAffine(), name="correct_pred")
    workflow.connect(datagrabber, 'pred', correct_pred, 'img')
    
    correct_seg = Node(CorrectAffine(), name="correct_seg")
    workflow.connect(datagrabber, 'seg', correct_seg, 'img')
    
    seg_to_pred_resample = Node(Resample_from_to(), name="seg_to_pred_resample")
    seg_to_pred_resample.inputs.spline_order = 0
    seg_to_pred_resample.inputs.out_name = 'seg_to_pred_resampled.nii.gz'
    
    workflow.connect(correct_seg, 'corrected_img', seg_to_pred_resample, 'moving_image')
    workflow.connect(correct_pred, 'corrected_img', seg_to_pred_resample, 'fixed_image')
    
    segtype = kwargs['BRAIN_SEG']  # Type: str
    pred = kwargs['PREDICTION']  # Type: str
    lpred = pred.lower()
    
    # Est-ce que la segmentation donnée pour filtrer est correctment appliquée si l'on donne un parcelisation?
    cluster_labelling = Node(Label_clusters(),
                                     name=f'cluster_labelling_{lpred}')
    cluster_labelling.inputs.thr_cluster_val = kwargs['THRESHOLD']
    cluster_labelling.inputs.thr_cluster_size = kwargs['MIN_SIZE'] - 1  # "- 1 because thr removes up to given value"
    cluster_labelling.inputs.out_name = f'labelled_{lpred}.nii.gz'
    
    prediction_metrics = Node(Regionwise_Prediction_metrics(),
                                      name=f"prediction_metrics_{lpred}")
    prediction_metrics.inputs.biomarker_type = lpred
    prediction_metrics.inputs.brain_seg_type = segtype
    
    # workflow.connect(prediction_metrics, 'biomarker_stats_csv', summary_report, f'{lpred}_metrics_csv')
    # workflow.connect(prediction_metrics, 'biomarker_census_csv', summary_report, f'{lpred}_census_csv')
    sink_node = Node(DataSink_CSV_and_PDF_safe(), name='sink_node')
    sink_node.inputs.base_directory = os.path.join(kwargs['BASE_DIR'], 'results')
    sink_node.inputs.substitutions = [
        ('_subject_id_', ''),
    ]
    # if segtype in ['synthseg', 'freesurfer', 'custom', 'brain_mask']:
    
    if segtype in ['synthseg', 'freesurfer']:
        seg_cleaning = Node(Segmentation_Cleaner(), name='seg_cleaning')
        shiva_parc = Node(Parc_from_Synthseg(), name='shiva_parc')
        custom_parc = Node(Brain_Seg_for_biomarker(), name='custom_parc')
        custom_parc.inputs.custom_parc = lpred if lpred in ('pvs', 'wmh') else 'mars'
        workflow.connect(seg_to_pred_resample, 'resampled_image', seg_cleaning, 'input_seg')
        workflow.connect(seg_cleaning, 'ouput_seg', shiva_parc, 'brain_seg')
        workflow.connect(shiva_parc, 'brain_parc', custom_parc, 'brain_seg')
        workflow.connect(custom_parc, 'brain_seg', cluster_labelling, 'brain_seg')
        workflow.connect(custom_parc, 'region_dict', prediction_metrics, 'region_dict')
        
        workflow.connect(shiva_parc, 'brain_parc', sink_node, f'{lpred}_segmentation.@shiva_brain_seg')
        workflow.connect(custom_parc, 'brain_seg', sink_node, f'{lpred}_segmentation.@custom_brain_seg')
        workflow.connect(custom_parc, 'region_dict_json', sink_node, f'{lpred}_segmentation.@region_dict_json')
    elif segtype == 'custom':
        prediction_metrics.inputs.region_dict = kwargs['CUSTOM_LUT']
        workflow.connect(seg_to_pred_resample, 'resampled_image', cluster_labelling, 'brain_seg')
    else:
        prediction_metrics.inputs.region_list = ['Whole brain']
        workflow.connect(seg_to_pred_resample, 'resampled_image', cluster_labelling, 'brain_seg')
    
    workflow.connect(cluster_labelling, 'labelled_biomarkers', prediction_metrics, 'brain_seg')
    
    workflow.connect(cluster_labelling,'labelled_biomarkers', sink_node, f'{lpred}_segmentation.@labelled')
    workflow.connect(prediction_metrics,'biomarker_stats_csv', sink_node, f'{lpred}_segmentation.@metrics')
    workflow.connect(prediction_metrics,'biomarker_stats_wide_csv', sink_node, f'{lpred}_segmentation.@metrics_wide')
    workflow.connect(prediction_metrics,'biomarker_census_csv', sink_node, f'{lpred}_segmentation.@census')

    return workflow