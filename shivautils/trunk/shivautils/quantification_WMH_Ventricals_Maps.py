'''
A script to quantify WMH per subject using Leventricalr distance masks and PostHorn mask.

Leventrical distance maps were obtained using: niimath (niimath /path/to/Leventrical_mask -binv -edt /path/to/save_directory/Leventrical_mask
PostHorn mask obtained using the WMH atlas threshold in standard space and the 0.005 threshold in native space.

Each individual table has the following columns:

sub_ID - subject identifier.
thr - WMH prediction threshold
WMH_cluster - cluster ID 
Nb_voxels - Nb voxels per cluster
Max_distance_to_Leventrical_mask - in mm, using Leventrical distance masks
Min_distance_to_Leventrical_mask - in mm, using Leventrical distance masks.      

DWMH - Cluster in Min_distance_to_Leventrical_mask > 2 mm 
PVWMH - Cluster in Min_distance_to_Leventrical_mask <= 2 mm 

@author: iastafeva
@date: 2022-12-06
'''


# %%
from nipype import config, logging  
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
import os
import os.path as op
import json
import os
# %%

SUBJECTFILE =  '/homes_unix/iastafeva/dev/1_subjects.txt'

config_json = '/homes_unix/iastafeva/dev/config_DM_Ventricals.json'


def create_distance_map(Leventrical_distance_maps, WMH, subject_id):
        import os.path as op
        import os
        from skimage import measure
        import nibabel
        import numpy as np
        import pandas as pd

        array_WMH = WMH.get_fdata(dtype=np.float32)[..., np.newaxis]
        Leventrical_distance_maps = Leventrical_distance_maps.get_fdata(dtype=np.float32)[..., np.newaxis]

        #Threshold & Binarize
        array_WMH[array_WMH<0.2] = 0
        array_WMH[array_WMH>=0.2] = 1
        
        # quantify nb of WMH predictions  
        all_clust, num_clust = measure.label(array_WMH.reshape(array_WMH.shape[0:3]), return_num=True, connectivity=3)
        if np.unique(all_clust)[0]==0:
                clusters = np.unique(all_clust)[1:]
        else:   clusters =  np.unique(all_clust)

        
        data = {'Sub_ID':  [],'thr': [],'WMH_cluster': [], 'Nb_voxels': [], 'Max_distance_to_Leventrical_mask': [], 
                'Min_distance_to_Leventrical_mask': [], 'DWMH': [],'PVWMH': []}
        
        if len(clusters)<1:
                data['Sub_ID'].append(subject_id)
                data['WMH_cluster'].append(np.nan)
                data['Nb_voxels'].append(np.nan)
                data['thr'].append(0.2)
                data['Max_distance_to_Leventrical_mask'].append(np.nan)
                data['Min_distance_to_Leventrical_mask'].append(np.nan)   
                
                data['DWMH'].append(np.nan)
                data['PVWMH'].append(np.nan) 
        
        
        for cl in clusters:
                #binarize first
                data['Sub_ID'].append(subject_id)
                data['WMH_cluster'].append(cl)
                data['Nb_voxels'].append(len(all_clust[all_clust ==cl]))
                data['thr'].append(0.2)
                data['Max_distance_to_Leventrical_mask'].append((max(Leventrical_distance_maps[all_clust ==cl]))[0])
                data['Min_distance_to_Leventrical_mask'].append((min(Leventrical_distance_maps[all_clust ==cl]))[0])
        
        
        
                if (min(Leventrical_distance_maps[all_clust ==cl]))[0]<= 2:
                        data['PVWMH'].append(1)
                else:     data['PVWMH'].append(0) 
                
                if (min(Leventrical_distance_maps[all_clust ==cl]))[0]> 2:
                        data['DWMH'].append(1)
                else:    data['DWMH'].append(0)
        

        nb_cluster_lateral_ventricles = data['PVWMH'].count(1)

        data = pd.DataFrame(data)
        return data, nb_cluster_lateral_ventricles