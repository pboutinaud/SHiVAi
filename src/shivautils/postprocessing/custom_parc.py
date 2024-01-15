'''
Custom parcellation functions for each cSVD biomarker
Each function requires our Synthseg-derived parcellation
'''

import numpy as np


def seg_for_pvs(parc: np.ndarray) -> tuple[np.ndarray, dict]:
    '''
    Will segment PVS into:    L / R
        - Deep              : 1 / 4
        - Basal Ganglia     : 2 / 7
        - Hippocampal       : 3 / 8
        - Cerebellar        : 4 / 9
       (- Ventral-diencephalic : 5 / 10)
    '''
    seg_vals = {
        # Left
        'Deep WM (L)': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 46],
        'Basal Ganglia (L)': [17, 40, 41, 44, 45],
        'Hippocampal (L)': [43],
        'Cerebellar (L)': [47],
        'Ventral DC (L)': [42],
        # Right
        'Deep WM (R)': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 56],
        'Basal Ganglia (R)': [37, 50, 51, 54, 55],
        'Hippocampal (R)': [53],
        'Cerebellar (R)': [57],
        'Ventral DC (R)': [52],
        #
        'Brainstem': 50
    }
    seg_labels = {
        # Left
        'Deep WM (L)': 1,
        'Basal Ganglia (L)': 2,
        'Hippocampal (L)': 3,
        'Cerebellar (L)': 4,
        'Ventral DC (L)': 5,
        # Right
        'Deep WM (R)': 6,
        'Basal Ganglia (R)': 7,
        'Hippocampal (R)': 8,
        'Cerebellar (R)': 9,
        'Ventral DC (R)': 10,
        #
        'Brainstem': 11
    }

    pvs_seg = np.zeros(parc.shape, int)
    for region, vals in seg_vals.items():
        pvs_seg[np.isin(parc, vals)] = seg_labels[region]

    return pvs_seg, seg_labels
