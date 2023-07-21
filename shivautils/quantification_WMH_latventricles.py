"""
Methods to quantify WMH per subject using lateral ventricles distance masks and PostHorn mask.

Lateral ventricles distance maps were obtained using: niimath (niimath /path/to/latventricle_mask -binv -edt /path/to/save_directory/latventricle_mask
PostHorn mask obtained using the WMH atlas threshold in standard space and the 0.005 threshold in native space.

Each individual table has the following columns:

sub_ID - subject identifier.
thr - WMH prediction threshold
WMH_cluster - cluster ID 
Nb_voxels - Nb voxels per cluster
Max_distance_to_Latventricle_mask - in mm, using Latventricle distance masks
Min_distance_to_Latventricle_mask - in mm, using Latventricle distance masks.      

DWMH - Cluster in Min_distance_to_Latventricle_mask > 2 mm 
PVWMH - Cluster in Min_distance_to_Latventricle_mask <= 2 mm 

@author: iastafeva
@date: 2022-12-06
"""
from nipype import config, logging  
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function


def metrics_clusters_latventricles(latventricle_distance_maps, wmh, subject_id, threshold=0.2):

    from skimage import measure
    import numpy as np
    import pandas as pd

    array_wmh = wmh.get_fdata(dtype=np.float32)[..., np.newaxis]
    latventricle_distance_maps = latventricle_distance_maps.get_fdata(dtype=np.float32)[..., np.newaxis]

    #Threshold & Binarize
    array_wmh[array_wmh < threshold] = 0
    array_wmh[array_wmh >= threshold] = 1
    
    # quantify nb of WMH predictions  
    all_clust, _ = measure.label(array_wmh.reshape(array_wmh.shape[0:3]), return_num=True, connectivity=3)
    if np.unique(all_clust)[0] == 0:
            clusters = np.unique(all_clust)[1:]
    else:   clusters =  np.unique(all_clust)

    
    data = {'Sub_ID': [],'thr': [],'WMH_cluster': [], 'Nb_voxels': [], 'Max_distance_to_Latventricle_mask': [], 
            'Min_distance_to_Latventricle_mask': [], 'DWMH': [],'PVWMH': []}
    

    nb_voxels_lateral_ventricles = 0
    nb_voxels_DWMH = 0

    if len(clusters)<1:
        data['Sub_ID'].append(subject_id)
        data['WMH_cluster'].append(np.nan)
        data['Nb_voxels'].append(np.nan)
        data['thr'].append(threshold)
        data['Max_distance_to_Latventricle_mask'].append(np.nan)
        data['Min_distance_to_Latventricle_mask'].append(np.nan)   
        
        data['DWMH'].append(np.nan)
        data['PVWMH'].append(np.nan) 
    
    
    for cl in clusters:
        #binarize first
        data['Sub_ID'].append(subject_id)
        data['WMH_cluster'].append(cl)
        data['Nb_voxels'].append(len(all_clust[all_clust ==cl]))
        data['thr'].append(threshold)
        data['Max_distance_to_Latventricle_mask'].append((max(latventricle_distance_maps[all_clust ==cl]))[0])
        data['Min_distance_to_Latventricle_mask'].append((min(latventricle_distance_maps[all_clust ==cl]))[0])

        if (min(latventricle_distance_maps[all_clust ==cl]))[0]<= 2:
                data['PVWMH'].append(1)
                nb_voxels_lateral_ventricles += len(all_clust[all_clust == cl])
        else:     data['PVWMH'].append(0) 
        
        if (min(latventricle_distance_maps[all_clust ==cl]))[0]> 2:
                data['DWMH'].append(1)
                nb_voxels_DWMH += len(all_clust[all_clust == cl])
        else:    data['DWMH'].append(0)
    

    nb_cluster_lateral_ventricles = data['PVWMH'].count(1)
    nb_cluster_dwmh = data['DWMH'].count(1)

    data = pd.DataFrame(data)
    return data, nb_cluster_lateral_ventricles, nb_cluster_dwmh, nb_voxels_lateral_ventricles, nb_voxels_DWMH, threshold
