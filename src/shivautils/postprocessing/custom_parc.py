'''
Custom parcellation functions for each cSVD biomarker
Each function requires our Synthseg-derived parcellation
'''

import numpy as np


def seg_for_pvs(parc: np.ndarray) -> tuple[np.ndarray, dict]:
    '''
    Will segment PVS into:  L / R
        - Deep:             1 / 6
        - Basal Ganglia:    2 / 7
        - Hippocampal:      3 / 8
        - Cerebellar:       4 / 9
        - Ventral DC:       5 / 10
        - Brainstem:          11

    '''
    seg_vals = {  # See src/shivautils/postprocessing/lobarseg.py for labels
        # Left
        'Left Deep WM': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 46],  # With cortical GM for now
        'Left Basal Ganglia': [17, 40, 41, 44, 45],
        'Left Hippocampal': [43],
        'Left Cerebellar': [47],
        'Left Ventral DC': [42],
        # Right
        'Right Deep WM': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 56],
        'Right Basal Ganglia': [37, 50, 51, 54, 55],
        'Right Hippocampal': [53],
        'Right Cerebellar': [57],
        'Right Ventral DC': [52],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Deep WM': 1,
        'Left Basal Ganglia': 2,
        'Left Hippocampal': 3,
        'Left Cerebellar': 4,
        'Left Ventral DC': 5,
        # Right
        'Right Deep WM': 6,
        'Right Basal Ganglia': 7,
        'Right Hippocampal': 8,
        'Right Cerebellar': 9,
        'Right Ventral DC': 10,
        #
        'Brainstem': 11
    }

    pvs_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        pvs_seg[np.isin(parc, vals)] = seg_labels[region]

    return pvs_seg, seg_labels


def seg_for_wmh(parc: np.ndarray) -> tuple[np.ndarray, dict]:
    '''
    Will segment WMH into:  L / R
        - Shallow:          1 / 5  (includes the cortex, i.e. GM)
        - Deep:             2 / 6
        - Perivetricular:   3 / 7
        - Cerebellar:       4 / 8
        - Brainstem:          9
    '''
    seg_vals = {  # See src/shivautils/postprocessing/lobarseg.py for labels
        # Left
        'Left Shallow WM': [1, 2, 5, 6, 9, 10, 13, 14, 17, 45],  # With cortical GM
        'Left Deep WM': [3, 7, 11, 15, 44, 46],
        'Left PV WM': [4, 8, 12, 16],
        'Left Cerebellar': [47],
        # Right
        'Right Shallow WM': [21, 22, 25, 26, 29, 30, 33, 34, 37, 55],  # With cortical GM
        'Right Deep WM': [23, 27, 31, 35, 54, 56],
        'Right PV WM': [24, 28, 32, 36],
        'Right Cerebellar': [57],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Shallow WM': 1,  # With cortical GM
        'Left Deep WM': 2,
        'Left PV WM': 3,
        'Left Cerebellar': 4,
        # Right
        'Right Shallow WM': 5,  # With cortical GM
        'Right Deep WM': 6,
        'Right PV WM': 7,
        'Right Cerebellar': 8,
        #
        'Brainstem': 9
    }
    wmh_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        wmh_seg[np.isin(parc, vals)] = seg_labels[region]
    return wmh_seg, seg_labels
