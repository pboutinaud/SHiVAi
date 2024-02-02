'''
Custom parcellation functions for each cSVD biomarker
Each function requires our Synthseg-derived parcellation
'''

import numpy as np
from typing import Tuple


def seg_for_pvs(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
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


def seg_for_wmh(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
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


def seg_from_mars(parc: np.ndarray) -> Tuple[np.ndarray, dict]:
    '''
    Based on the Microbleed Anatomical Rating Scale (MARS)
    Will segment CMBs into: L / R
        - Frontal:          1 / 15
        - Parietal:         2 / 16
        - Temporal:         3 / 17
        - Occipital:        4 / 18
        - Insula:           5 / 19 (just the GM)
        - Basal Ganglia:    6 / 20
        - Thalamus:         7 / 21
        - Ventral DC:       8 / 22
        - Hippocampus:      9 / 23
        - Int. Capsule:    10 / 24
        - Ext. Capsule:    11 / 25
        - Corp. Call.:     12 / 26
        - Deep and PV WM:  13 / 27
        - Cerebellum:      14 / 28
        - Brainstem:          29

    '''
    seg_vals = {  # See src/shivautils/postprocessing/lobarseg.py for labels
        # Left
        'Left Frontal': [1, 2],
        'Left Parietal': [5, 6],
        'Left Temporal': [9, 10],
        'Left Occipital': [13, 14],
        'Left Insula': [17],
        'Left Basal Ganglia': [40],
        'Left Thalamus': [41],
        'Left Ventral DC': [42],
        'Left Hippocampus': [43],
        'Left Int. Capsule': [44],
        'Left Ext. Capsule': [45],
        'Left Corp. Call': [46],
        'Left Deep and PV WM': [3, 4, 7, 8, 11, 12, 15, 16],
        'Left Cerebellum': [47],
        # Right
        'Right Frontal': [21, 22],
        'Right Parietal': [25, 26],
        'Right Temporal': [29, 30],
        'Right Occipital': [33, 34],
        'Right Insula': [37],
        'Right Basal Ganglia': [50],
        'Right Thalamus': [51],
        'Right Ventral DC': [52],
        'Right Hippocampus': [53],
        'Right Int. Capsule': [54],
        'Right Ext. Capsule': [55],
        'Right Corp. Call': [56],
        'Right Deep and PV WM': [23, 24, 27, 28, 31, 32, 35, 36],
        'Right Cerebellum': [57],
        #
        'Brainstem': 60
    }
    seg_labels = {
        # Left
        'Left Frontal': 1,
        'Left Parietal': 2,
        'Left Temporal': 3,
        'Left Occipital': 4,
        'Left Insula': 5,
        'Left Basal Ganglia': 6,
        'Left Thalamus': 7,
        'Left Ventral DC': 8,
        'Left Hippocampus': 9,
        'Left Int. Capsule': 10,
        'Left Ext. Capsule': 11,
        'Left Corp. Call': 12,
        'Left Deep and PV WM': 13,
        'Left Cerebellum': 14,
        # Right
        'Right Frontal': 15,
        'Right Parietal': 16,
        'Right Temporal': 17,
        'Right Occipital': 18,
        'Right Insula': 19,
        'Right Basal Ganglia': 20,
        'Right Thalamus': 21,
        'Right Ventral DC': 22,
        'Right Hippocampus': 23,
        'Right Int. Capsule': 24,
        'Right Ext. Capsule': 25,
        'Right Corp. Call': 26,
        'Right Deep and PV WM': 27,
        'Right Cerebellum': 28,
        #
        'Brainstem': 29
    }
    cmb_seg = np.zeros(parc.shape, 'int16')
    for region, vals in seg_vals.items():
        cmb_seg[np.isin(parc, vals)] = seg_labels[region]
    return cmb_seg, seg_labels
