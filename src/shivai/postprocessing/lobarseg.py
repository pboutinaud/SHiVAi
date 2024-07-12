"""
Lobar segmentation from a SynthSeg parc
1001    ctx-lh-bankssts                     Temporal
    1002    ctx-lh-caudalanteriorcingulate      Frontal (Limbic) #
1003    ctx-lh-caudalmiddlefrontal          Frontal
1005    ctx-lh-cuneus                       Occipital
1006    ctx-lh-entorhinal                   Temporal
1007    ctx-lh-fusiform                     Temporal
1008    ctx-lh-inferiorparietal             Parietal
1009    ctx-lh-inferiortemporal             Temporal
    1010    ctx-lh-isthmuscingulate             Parietal (Limbic) #
1011    ctx-lh-lateraloccipital             Occipital
1012    ctx-lh-lateralorbitofrontal         Frontal
1013    ctx-lh-lingual                      Occipital
1014    ctx-lh-medialorbitofrontal          Frontal
1015    ctx-lh-middletemporal               Temporal
1016    ctx-lh-parahippocampal              Temporal (Limbic)
    1017    ctx-lh-paracentral                  Frontal (Parietal?) #
1018    ctx-lh-parsopercularis              Frontal (Insula)
1019    ctx-lh-parsorbitalis                Frontal
1020    ctx-lh-parstriangularis             Frontal
1021    ctx-lh-pericalcarine                Occipital
    1022    ctx-lh-postcentral                  Parietal #
    1023    ctx-lh-posteriorcingulate           Parietal (Limbic) #
    1024    ctx-lh-precentral                   Frontal #
1025    ctx-lh-precuneus                    Parietal
    1026    ctx-lh-rostralanteriorcingulate     Frontal (Limbic) #
1027    ctx-lh-rostralmiddlefrontal         Frontal
1028    ctx-lh-superiorfrontal              Frontal
1029    ctx-lh-superiorparietal             Parietal
1030    ctx-lh-superiortemporal             Temporal
1031    ctx-lh-supramarginal                Parietal
1032    ctx-lh-frontalpole                  Frontal
1033    ctx-lh-temporalpole                 Temporal
1034    ctx-lh-transversetemporal           Temporal
1035    ctx-lh-insula                       Insula
"""
from scipy.ndimage import (distance_transform_edt,
                           binary_erosion,
                           binary_dilation,
                           binary_closing,
                           binary_opening)
from scipy.spatial import Delaunay, ConvexHull
import nibabel as nib
import numpy as np

# %%
lobar_vals_L = {
    'Frontal': [1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1027, 1028, 1032, 1024, 1026],
    'Pariental': [1008, 1010, 1025, 1029, 1031, 1022, 1023],
    'Temporal': [1001, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034],
    'Occipital': [1005, 1011, 1013, 1021],
}
lobar_vals_R = {k: [v+1000 for v in val] for k, val in lobar_vals_L.items()}

other_vals_L = {
    'Insula': [1035],
    'Basal Ganglia': [11, 12, 13, 26],  # incuding N.Acc
    'Thalamus': [10],
    'Ventral DC': [28],
    'Hippocampus': [17, 18],  # including amygdala
    'Cerebellum': [7, 8]
}
other_vals_R = {
    'Insula': [2035],
    'Basal Ganglia': [50, 51, 52, 58],  # incuding N.Acc
    'Thalamus': [49],
    'Ventral DC': [60],
    'Hippocampus': [53, 54],  # including amygdala
    'Cerebellum': [46, 47]
}
brainstem_val = 16

# Excluding cortical areas for lobe parcellation to avoid a "jigsaw puzzle" effect
to_excluded_L = [1002, 1017, 1026, 1010, 1023]
to_excluded_R = [val+1000 for val in to_excluded_L]

cingulate_vals_L = [1002, 1010, 1023, 1026]
cingulate_vals_R = [2002, 2010, 2023, 2026]

lobar_labels_L = {
    'Frontal': 1,  # 2, 3, 4 for wm (jxtc/deep/prvc)
    'Pariental': 5,  # 6, 7, 8
    'Temporal': 9,  # 10, 11, 12
    'Occipital': 13,  # 14, 15, 16
}
other_labels_L = {
    'Insula': 17,
    'Basal Ganglia': 40,
    'Thalamus': 41,
    'Ventral DC': 42,
    'Hippocampus': 43,  # including amygdala
    'Internal capsule': 44,
    'Ext. capsule': 45,
    'Corpus callosum': 46,
    'Cerebellum': 47
}

lobar_labels_R = {k: val+20 for k, val in lobar_labels_L.items()}
other_labels_R = {k: val+10 for k, val in other_labels_L.items()}
other_labels_R['Insula'] = other_labels_L['Insula'] + 20
brainstem_label = 60

# %%


def expand_label_masked(label_image, mask):
    '''
    Re-implementation of expand_label with a mask to fill instead of a distance
    '''
    nearest_label_coords = distance_transform_edt(
        label_image == 0, return_distances=False, return_indices=True
    )
    labels_out = np.zeros(label_image.shape, dtype='int16')
    masked_nearest_label_coords = [
        dimension_indices[mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[mask] = nearest_labels
    return labels_out


def lobar_seg(seg):
    '''
    First rough re-segmentation, with "lobar_labels" voxels and "other_labels" voxels
    '''
    wm_L = (seg == 2)
    wm_R = (seg == 41)

    # Also fill excluded cortical areas
    wm_L = wm_L | np.isin(seg, to_excluded_L)
    wm_R = wm_R | np.isin(seg, to_excluded_R)

    # Divide cortex in lobes
    vol_lobar_L = np.zeros(seg.shape, dtype='int16')
    vol_lobar_R = np.zeros(seg.shape, dtype='int16')
    for lob in lobar_vals_L.keys():
        vals_L = list(set(lobar_vals_L[lob]) - set(to_excluded_L))
        vals_R = list(set(lobar_vals_R[lob]) - set(to_excluded_R))
        vol_lobar_L[np.isin(seg, vals_L)] = lobar_labels_L[lob]
        vol_lobar_R[np.isin(seg, vals_R)] = lobar_labels_R[lob]

    # Associate WM to lobes (and add the cortical parts)
    vol_lobar_L_exp = expand_label_masked(vol_lobar_L, wm_L) + vol_lobar_L
    vol_lobar_R_exp = expand_label_masked(vol_lobar_R, wm_R) + vol_lobar_R

    # Add the other labels
    for parc in other_vals_L:
        parc_vox = np.isin(seg, other_vals_L[parc])
        vol_lobar_L_exp[parc_vox] = other_labels_L[parc]
    for parc in other_vals_R:
        parc_vox = np.isin(seg, other_vals_R[parc])
        vol_lobar_R_exp[parc_vox] = other_labels_R[parc]

    vol_lobar_exp = vol_lobar_L_exp + vol_lobar_R_exp
    vol_lobar_exp[seg == brainstem_val] = brainstem_label
    return vol_lobar_exp


# %%


def fill_hull(brain_regions):
    '''
    Create a convex hull around all the points (ore brain regions) given
    then fill the hull to give one big all-encompassing blob
    '''
    points = np.argwhere(brain_regions)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(brain_regions.shape), axis=-1)
    bg_idx = np.nonzero(deln.find_simplex(idx) + 1)
    filled_vol = np.zeros(brain_regions.shape, dtype=bool)
    filled_vol[bg_idx] = True
    return filled_vol


def internal_caps(seg):
    '''
    Find the internal capsule by creating a convex hull around the basal ganglia
    getting its internal volume, doing some erosion to avoid voxels that could
    be around the BG (instead of between) and keeping only WM labeled voxels
    '''
    bg_labels_L = [10, 11, 12, 13]  # With thalamus
    bg_labels_R = [49, 50, 51, 52]

    wm_L = (seg == 2)
    wm_R = (seg == 41)

    bg_L = np.isin(seg, bg_labels_L)
    bg_R = np.isin(seg, bg_labels_R)

    ic_L = binary_erosion(fill_hull(bg_L), iterations=2) & wm_L
    ic_R = binary_erosion(fill_hull(bg_R), iterations=2) & wm_R

    return ic_L, ic_R


def ex_capsule(seg, exclusion_wm):
    '''
    External and extreme capsule, roughly between insula and putamen
    exclusion_wm should be the juxta-cortical WM and the internal capsule
    '''
    wm_L = (seg == 2)
    wm_R = (seg == 41)

    hipp_vdc = np.isin(seg, [17, 53, 28, 60])  # hippocampus and ventral DC
    hipp_vdc_dil = binary_dilation(hipp_vdc, iterations=5)  # To prevent the ec from growing too low

    exclusion_area = exclusion_wm | hipp_vdc_dil

    putamen_L = (seg == 12)
    insula_L = (seg == 1035)
    putamen_R = (seg == 51)
    insula_R = (seg == 2035)

    xcap_L_raw = (fill_hull(insula_L | putamen_L).astype(bool) &
                  binary_dilation(putamen_L, iterations=10) &
                  binary_dilation(insula_L, iterations=10))
    xcap_R_raw = (fill_hull(insula_R | putamen_R).astype(bool) &
                  binary_dilation(putamen_R, iterations=10) &
                  binary_dilation(insula_R, iterations=10))

    xcap_L = xcap_L_raw & wm_L & ~exclusion_area
    xcap_R = xcap_R_raw & wm_R & ~exclusion_area

    # final cleaning
    xcap_L = binary_closing(xcap_L | insula_L | putamen_L, iterations=3) & ~(insula_L | putamen_L) & wm_L
    xcap_R = binary_closing(xcap_R | insula_R | putamen_R, iterations=3) & ~(insula_R | putamen_R) & wm_R

    return xcap_L, xcap_R


# def external_caps(seg, ec_thickness=2):  # jc = juxtacortical
#     '''
#     Create a mask of the external capsule, based on its location bewteen the putamen
#     and the insula, stuck to the putamen with a thickness of about 2mm (?)
#     '''
#     wm_L = (seg == 2)
#     wm_R = (seg == 41)

#     putamen_L = (seg == 12)
#     insula_L = (seg == 1035)
#     putamen_R = (seg == 51)
#     insula_R = (seg == 2035)

#     wm_area_L = fill_hull(putamen_L + insula_L)*wm_L
#     wm_area_R = fill_hull(putamen_R + insula_R)*wm_R

#     put_dil_L = binary_dilation(putamen_L, interations=ec_thickness)
#     put_dil_R = binary_dilation(putamen_R, interations=ec_thickness)

#     ec_L = (put_dil_L & wm_area_L)
#     ec_R = (put_dil_R & wm_area_R)

#     return ec_L, ec_R

# %%


def juxtacortical_wm(seg, thickness=3):
    '''
    Juxtacortical white matter, excluding the insula
    '''
    wm_L = (seg == 2)
    wm_R = (seg == 41)

    cortex_vals_L = []
    for _, vals in lobar_vals_L.items():
        cortex_vals_L += vals
    cortex_vals_R = []
    for _, vals in lobar_vals_R.items():
        cortex_vals_R += vals
    cortex_vals = cortex_vals_L + cortex_vals_R
    cortex = np.isin(seg, cortex_vals)
    cortex_dil = binary_dilation(cortex, iterations=thickness)
    jxtc_L = cortex_dil & wm_L
    jxtc_R = cortex_dil & wm_R
    return jxtc_L, jxtc_R


def periventricular_wm(seg, thickness=2):
    wm_L = (seg == 2)
    wm_R = (seg == 41)

    vent_L = np.isin(seg, [4, 5])
    vent_R = np.isin(seg, [43, 44])

    pvwm_L = binary_dilation(vent_L, iterations=thickness) * wm_L
    pvwm_R = binary_dilation(vent_R, iterations=thickness) * wm_R

    return pvwm_L, pvwm_R


# %%

def corpus_cal(seg):
    wm_L = (seg == 2)
    wm_R = (seg == 41)

    wm = wm_L + wm_R

    # Exclusion areas for wm
    #   Fornix
    vent_1_2 = np.isin(seg, [4, 5, 43, 44])
    vent_3_dil6 = binary_dilation(seg == 14, iterations=6)
    vent_123_raw = vent_1_2 + vent_3_dil6
    vent_123 = binary_closing(vent_123_raw, iterations=6)
    #   Cingulum
    cing_L = binary_closing(np.isin(seg, cingulate_vals_L), iterations=5)
    cing_R = binary_closing(np.isin(seg, cingulate_vals_R), iterations=5)
    cing = cing_L + cing_R
    #   Total
    exclusion_area = vent_123 + cing
    wm[exclusion_area] = False

    dil_wm_L = binary_dilation(wm_L)*wm
    dil_wm_R = binary_dilation(wm_R)*wm

    seed_cc = dil_wm_L & dil_wm_R

    cc_raw = binary_dilation(seed_cc, iterations=10)
    cc_filtered = binary_opening(cc_raw & wm, iterations=2)  # To hopefully remove stray extensions (in cingulate)

    cc_L = (cc_filtered & wm_L)
    cc_R = (cc_filtered & wm_R)
    return cc_L, cc_R

# %%


def lobar_and_wm_segmentation(seg):
    '''
    seg (numpy 3D array): Synthseg segmentation

    Takes a Synthseg parcellation and returns derived segmentation with lobar labels for both
    GM and WM, and juxtacortical/deep/periventricular WM subdivisions, as well as internal capsule
    and external/extreme capsule.

    Left hemisphere:
                    gm   jxct deep prvc
        Frontal:    1    2    3    4 
        Pariental:  5    6    7    8
        Temporal:   9    10   11   12
        Occipital:  13   14   15   16

        Insula:         17 (gm only)
        Basal Ganglia:  40
        Thalamus:       41
        Ventral DC:     42
        Hippocampus:    43  (including amygdala)
        Int. capsule:   44
        Ext. capsule:   45  (external + extreme capsules)
        Corpus call.:   46
        Cerebellum :    47

    Right hemisphere:
                    gm   jxct deep prvc
        Frontal:    21   22   23   24 
        Pariental:  25   26   27   28
        Temporal:   29   30   31   32
        Occipital:  33   34   35   36

        Insula:         37 (gm only)
        Basal Ganglia:  50
        Thalamus:       51
        Ventral DC:     52
        Hippocampus:    53  (including amygdala)
        Int. capsule:   54
        Ext. capsule:   55  (external + extreme capsules)
        Corpus call.:   56
        Cerebellum':    57

    Brainstem: 60

    '''

    seg_lobar = lobar_seg(seg)

    ic_L, ic_R = internal_caps(seg)
    jxtc_L, jxtc_R = juxtacortical_wm(seg, 3)
    prvc_L, prvc_R = periventricular_wm(seg, 2)

    ec_L, ec_R = ex_capsule(seg, (jxtc_L | jxtc_R | ic_L | ic_R))
    cc_L, cc_R = corpus_cal(seg)

    wm_L = (seg == 2)
    wm_R = (seg == 41)

    for _, vals in lobar_labels_L.items():
        lobar = (seg_lobar == vals) & wm_L
        jxtc = lobar & jxtc_L
        prvc = lobar & prvc_L
        seg_lobar[lobar] = vals + 2  # deep wm, set first to all wm as default
        seg_lobar[jxtc] = vals + 1  # juxtacortical wm
        seg_lobar[prvc] = vals + 3  # periventricular wm

    for _, vals in lobar_labels_R.items():
        lobar = (seg_lobar == vals) & wm_R
        jxtc = lobar & jxtc_R
        prvc = lobar & prvc_R
        seg_lobar[lobar] = vals + 2  # deep wm, set first to all wm as default
        seg_lobar[jxtc] = vals + 1  # juxtacortical wm
        seg_lobar[prvc] = vals + 3  # periventricular wm

    seg_lobar[ic_L] = other_labels_L['Internal capsule']
    seg_lobar[ic_R] = other_labels_R['Internal capsule']
    seg_lobar[ec_L] = other_labels_L['Ext. capsule']
    seg_lobar[ec_R] = other_labels_R['Ext. capsule']
    seg_lobar[cc_L] = other_labels_L['Corpus callosum']
    seg_lobar[cc_R] = other_labels_R['Corpus callosum']

    return seg_lobar

# %%
