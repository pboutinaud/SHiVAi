''' 
A script to quantify PVS segmentation in two regions: Basal ganglia (BG) and deep white matter (DWM), using the winner's rule.

INPUTS: PVS segmentations, basal ganglia binary mask.
You must also specify the threshold applied to PVS segmentations (usually = 0.5) and cluster_filter applied to PVS clusters.

OUTPUT: csv is an individual global table containing information for each subject:

Threshold - the threshold that was applied to the PVS segmentation
Cluster_filters - cluster filter that was applied to PVS clusters
Total_num_clusters - total number of clusters (in BG + DWM regions)
Total_num_voxels - total number of voxels (in BG + DWM regions)
    
DWM_num_clusters - Number of clusters in DWM region
DWM_num_voxels - Number of voxels in DWM region
BG_num_clusters - Number of clusters in the BG region
BG_num_voxels - Number of voxels in the BG region

@autor: iastafeva
@date: 19/06/2023
'''

# %%
from nipype import config, logging  
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function
import os
import os.path as op
import json
import os


SUBJECTFILE =  '/extra/SHIVA/scripts/swoomed/subjects.txt'

config_json = '/extra/SHIVA/scripts/swoomed/PVS_config.json'


# %%
def quantify_clusters(img, bg_mask, thr, cl_filter ,out_csv='cluster_summary_pvs.csv'):
    '''
    Given the image, count the number of clusters
    and voxels in the img in BG region and DWM regions

    Returns out_csv if file name is provided.
    '''
    import os.path as op
    import numpy as np
    import pandas as pd
    import nibabel
    from skimage import measure
    
    # load WMH predictions    
    img = img.get_fdata()
    img = img.reshape(img.shape[0:3])
    
    bg_mask = bg_mask.get_fdata()
    bg_mask = bg_mask.reshape(bg_mask.shape[0:3])
    
    #Threshold & Binarize
    img[img<thr] = 0
    img[img>=thr] = 1

    # Start quantification
    quant_data = {}

    all_clust, num_clust = measure.label(img, return_num=True, connectivity=3)
    unique, counts = np.unique(all_clust[all_clust!=0], return_counts=True)
    
    quant_data['Threshold'] = [thr]
    quant_data['Cluster filter DWM'] = [cl_filter]
    
    quant_data['DWM num clusters'] = 0
    quant_data['DWM num voxels'] =0

    quant_data['BG num clusters'] = 0   
    quant_data['BG num voxels'] =0
    
    # DWM with cluster filter
    for c in unique[counts >cl_filter]:
 
        unique_DWM, counts = np.unique(bg_mask[all_clust==c], return_counts=True)
        winner = unique_DWM[counts == counts.max()]
        
        # assign random label if more than one winner
        if winner.size > 1:
            winner = np.array([np.random.choice(winner)])   
        
        if winner ==0:
            quant_data['DWM num clusters']+=1
            quant_data['DWM num voxels'] += len(all_clust[all_clust==c])

            
    # BG without cluster filter    
    for c in unique:    
        unique_BG, counts = np.unique(bg_mask[all_clust==c], return_counts=True)
        winner = unique_BG[counts == counts.max()]
        # assign random label if more than one winner
        if winner.size > 1:
            winner = np.array([np.random.choice(winner)])   
        
        if winner !=0:
            quant_data['BG num clusters']+=1    
            quant_data['BG num voxels'] += len(all_clust[all_clust==c])
            
    quant_data['Total num clusters'] = [quant_data['DWM num clusters'] + quant_data['BG num clusters']]
    quant_data['Total num voxels'] = [quant_data['DWM num voxels'] + quant_data['BG num voxels']]
     
    out_df = pd.DataFrame(quant_data)
   
    return out_df