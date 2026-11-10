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
                           distance_transform_cdt,
                           binary_erosion,
                           binary_dilation,
                           binary_closing,
                           binary_opening,
                           label as nd_label)
from scipy.spatial import Delaunay, ConvexHull
from skimage.morphology import ball as _skball
import numpy as np
from shivai.utils.misc import fisin


def _dilate(mask, r):
    """Binary dilation by r voxels (taxicab/L1 metric), single O(V) CDT call.
    Semantically identical to binary_dilation(mask, iterations=r) with 6-conn footprint.
    """
    return distance_transform_cdt(~np.asarray(mask, dtype=bool), metric='taxicab') <= r


def _erode(mask, r):
    """Binary erosion by r voxels (taxicab/L1 metric), single O(V) CDT call.
    Semantically identical to binary_erosion(mask, iterations=r) with 6-conn footprint.
    """
    return distance_transform_cdt(np.asarray(mask, dtype=bool), metric='taxicab') >= r


def _close(mask, r):
    """Binary closing by r voxels (taxicab/L1 metric). Two CDT calls instead of 2r iterations."""
    return _erode(_dilate(mask, r), r)


# %%  Values in the original Freesurfer/Synthseg parcellation for the new labels
LOBAR_VALS_L = {
    'Frontal': [1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1027, 1028, 1032, 1024, 1026],
    'Pariental': [1008, 1010, 1025, 1029, 1031, 1022, 1023],
    'Temporal': [1001, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034],
    'Occipital': [1005, 1011, 1013, 1021],
}
LOBAR_VALS_R = {k: [v+1000 for v in val] for k, val in LOBAR_VALS_L.items()}

OTHER_VALS_L = {
    'Insula': [1035],
    'Basal Ganglia': [11, 12, 13, 26],  # incuding N.Acc
    'Thalamus': [10],
    'Ventral DC': [28],
    'Hippocampus': [17, 18],  # including amygdala
    'Cerebellum': [7, 8]
}
OTHER_VALS_R = {
    'Insula': [2035],
    'Basal Ganglia': [50, 51, 52, 58],  # incuding N.Acc
    'Thalamus': [49],
    'Ventral DC': [60],
    'Hippocampus': [53, 54],  # including amygdala
    'Cerebellum': [46, 47]
}
BRAINSTEM_VAL = 16

# Excluding cortical areas for lobe parcellation to avoid a "jigsaw puzzle" effect
TO_EXCLUDE_L = [1002, 1017, 1026, 1010, 1023]
TO_EXCLUDE_R = [val+1000 for val in TO_EXCLUDE_L]

CINGULATE_VALS_L = [1002, 1010, 1023, 1026]
CINGULATE_VALS_R = [2002, 2010, 2023, 2026]

LAT_VENT_VALS_L = [4, 5, 31]  # with choroid plexus
LAT_VENT_VALS_R = [43, 44, 63]

WM_VAL_L = 2
WM_VAL_R = 41

# %% Labels for the final parcellation

LOBAR_GM_LABELS_L = {
    'Frontal': 1,  # 2, 3, 4 for wm (jxtc/deep/prvc)
    'Pariental': 5,  # 6, 7, 8
    'Temporal': 9,  # 10, 11, 12
    'Occipital': 13,  # 14, 15, 16
}
LOBAR_WM_LABELS_L = {lobe: [val+i for i in range(1, 4)] for lobe, val in LOBAR_GM_LABELS_L.items()}

OTHER_LABELS_L = {
    'Insula': 17,
    'Basal Ganglia': 40,
    'Thalamus': 41,
    'Ventral DC': 42,
    'Hippocampus': 43,  # including amygdala
    'Internal capsule': 44,
    'Ext. capsule': 45,
    'Corpus callosum': 46,
    'Cerebellum': 47,
    'BG prvc': 78,
    'Thal. prvc': 79,
    'In. cap. prvc': 80
}

LOBAR_GM_LABELS_R = {k: val+20 for k, val in LOBAR_GM_LABELS_L.items()}
LOBAR_WM_LABELS_R = {k: [v+20 for v in val] for k, val in LOBAR_WM_LABELS_L.items()}

OTHER_LABELS_R = {k: val+10 for k, val in OTHER_LABELS_L.items()}
OTHER_LABELS_R['Insula'] = OTHER_LABELS_L['Insula'] + 20

BRAINSTEM_LABEL = 60

# %%


def expand_label_masked(label_image, mask):
    '''
    Re-implementation of expand_label with a mask to fill instead of a distance
    '''
    nearest_label_coords = distance_transform_cdt(
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
    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    # Also fill excluded cortical areas
    wm_L = wm_L | fisin(seg, TO_EXCLUDE_L)
    wm_R = wm_R | fisin(seg, TO_EXCLUDE_R)

    # Divide cortex in lobes
    vol_lobar_L = np.zeros(seg.shape, dtype='int16')
    vol_lobar_R = np.zeros(seg.shape, dtype='int16')
    for lob in LOBAR_VALS_L.keys():
        vals_L = list(set(LOBAR_VALS_L[lob]) - set(TO_EXCLUDE_L))
        vals_R = list(set(LOBAR_VALS_R[lob]) - set(TO_EXCLUDE_R))
        vol_lobar_L[fisin(seg, vals_L)] = LOBAR_GM_LABELS_L[lob]
        vol_lobar_R[fisin(seg, vals_R)] = LOBAR_GM_LABELS_R[lob]

    # Associate WM to lobes (and add the cortical parts)
    vol_lobar_L_exp = expand_label_masked(vol_lobar_L, wm_L) + vol_lobar_L
    vol_lobar_R_exp = expand_label_masked(vol_lobar_R, wm_R) + vol_lobar_R

    # Add the other labels
    for parc in OTHER_VALS_L:
        parc_vox = fisin(seg, OTHER_VALS_L[parc])
        vol_lobar_L_exp[parc_vox] = OTHER_LABELS_L[parc]
    for parc in OTHER_VALS_R:
        parc_vox = fisin(seg, OTHER_VALS_R[parc])
        vol_lobar_R_exp[parc_vox] = OTHER_LABELS_R[parc]

    vol_lobar_exp = vol_lobar_L_exp + vol_lobar_R_exp
    vol_lobar_exp[seg == BRAINSTEM_VAL] = BRAINSTEM_LABEL
    return vol_lobar_exp


# %%


def fill_hull(brain_regions):
    '''
    Create a convex hull around all the points (ore brain regions) given
    then fill the hull to give one big all-encompassing blob
    '''
    filled_vol = np.zeros(brain_regions.shape, dtype=bool)
    if not np.any(brain_regions):
        return filled_vol
    points = np.argwhere(brain_regions)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(brain_regions.shape), axis=-1)
    bg_idx = np.nonzero(deln.find_simplex(idx) + 1)
    filled_vol[bg_idx] = True
    return filled_vol


def internal_caps(seg) -> tuple[np.ndarray[tuple[int, int, int], np.dtype[np.bool_]], np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]]:
    '''
    Find the internal capsule by creating a convex hull around the basal ganglia
    getting its internal volume, doing some erosion to avoid voxels that could
    be around the BG (instead of between) and keeping only WM labeled voxels
    '''
    bg_labels_L = [10, 11, 12, 13]  # With thalamus
    bg_labels_R = [49, 50, 51, 52]

    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    bg_L = fisin(seg, bg_labels_L)
    bg_R = fisin(seg, bg_labels_R)

    ic_L = binary_erosion(fill_hull(bg_L), iterations=2) & wm_L
    ic_R = binary_erosion(fill_hull(bg_R), iterations=2) & wm_R

    return ic_L, ic_R


def ex_capsule(seg, exclusion_wm):
    '''
    External and extreme capsule, roughly between insula and putamen
    exclusion_wm should be the juxta-cortical WM and the internal capsule
    '''
    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    hipp_vdc = fisin(seg, [17, 53, 28, 60])  # hippocampus and ventral DC
    hipp_vdc_dil = _dilate(hipp_vdc, 5)  # To prevent the ec from growing too low

    exclusion_area = exclusion_wm | hipp_vdc_dil

    putamen_L = (seg == 12)
    insula_L = (seg == 1035)
    putamen_R = (seg == 51)
    insula_R = (seg == 2035)

    # Pre-compute dilations once (reused in xcap_raw and closing below)
    put_L_dil = _dilate(putamen_L, 10)
    ins_L_dil = _dilate(insula_L, 10)
    put_R_dil = _dilate(putamen_R, 10)
    ins_R_dil = _dilate(insula_R, 10)

    xcap_L_raw = fill_hull(insula_L | putamen_L).astype(bool) & put_L_dil & ins_L_dil
    xcap_R_raw = fill_hull(insula_R | putamen_R).astype(bool) & put_R_dil & ins_R_dil

    xcap_L = xcap_L_raw & wm_L & ~exclusion_area
    xcap_R = xcap_R_raw & wm_R & ~exclusion_area

    # final cleaning
    xcap_L = _close(xcap_L | insula_L | putamen_L, 3) & ~(insula_L | putamen_L) & wm_L
    xcap_R = _close(xcap_R | insula_R | putamen_R, 3) & ~(insula_R | putamen_R) & wm_R

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
    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    cortex_vals_L = []
    for _, vals in LOBAR_VALS_L.items():
        cortex_vals_L += vals
    cortex_vals_R = []
    for _, vals in LOBAR_VALS_R.items():
        cortex_vals_R += vals
    cortex_vals = cortex_vals_L + cortex_vals_R
    cortex = fisin(seg, cortex_vals)
    cortex_dil = _dilate(cortex, thickness)
    jxtc_L = cortex_dil & wm_L
    jxtc_R = cortex_dil & wm_R
    return jxtc_L, jxtc_R


def periventricular(seg, thickness=2):
    vent = fisin(seg, LAT_VENT_VALS_L + LAT_VENT_VALS_R)  # we ignore the third ventricle for now (label 14)
    vent_dil = binary_dilation(vent, iterations=thickness).astype(bool)
    pv = vent_dil & ~vent
    return pv


def periventricular_wm(seg, thickness=2):
    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    vent_L = fisin(seg, LAT_VENT_VALS_L)
    vent_R = fisin(seg, LAT_VENT_VALS_R)

    pvwm_L = binary_dilation(vent_L, iterations=thickness) * wm_L
    pvwm_R = binary_dilation(vent_R, iterations=thickness) * wm_R

    return pvwm_L, pvwm_R


# %%

def convert_hyp_to_wm(seg: np.ndarray) -> np.ndarray:
    '''
    Convert "white matter hypo/hyper-intensities" (label 77) to corresponding
    latteralized WM (2 for left, 41 for right) using connected component analysis.

    '''
    hyp_vox = (seg == 77)

    if not np.any(hyp_vox):
        return seg  # Early exit if no hyp voxels

    # Label connected components of hyp voxels
    labeled_hyp, num_clusters = nd_label(hyp_vox)

    if num_clusters == 0:
        return seg

    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    # Dilate each WM region by 1 voxel to ensure contact detection
    wm_L_dilated = binary_dilation(wm_L, iterations=1)
    wm_R_dilated = binary_dilation(wm_R, iterations=1)

    # For each cluster, count contacts with left and right WM
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = (labeled_hyp == cluster_id)

        # Count voxels in this cluster that touch each WM region
        touches_L = np.sum(cluster_mask & wm_L_dilated)
        touches_R = np.sum(cluster_mask & wm_R_dilated)

        # Assign entire cluster to the WM region it touches most
        if touches_L > touches_R:
            seg[cluster_mask] = WM_VAL_L
        elif touches_R > touches_L:
            seg[cluster_mask] = WM_VAL_R
        else:
            # If equal contact (rare), check which WM region is larger overall
            # or default to left
            if np.sum(wm_L) >= np.sum(wm_R):
                seg[cluster_mask] = WM_VAL_L
            else:
                seg[cluster_mask] = WM_VAL_R

    return seg


def corpus_cal(seg):
    cc_labels = [251, 252, 253, 254, 255]
    if fisin(seg, cc_labels).any():
        cc = fisin(seg, cc_labels)
        cc_mask = binary_dilation(cc)
        # Assign each CC voxel to the nearest WM half (2=left, 41=right) via EDT
        seed = np.zeros(seg.shape, dtype='int16')
        wm_in_mask = cc_mask & fisin(seg, [WM_VAL_L, WM_VAL_R])
        seed[wm_in_mask] = seg[wm_in_mask]
        assigned = expand_label_masked(seed, cc_mask)
        cc_L = (assigned == WM_VAL_L) & cc_mask
        cc_R = (assigned == WM_VAL_R) & cc_mask
        return cc_L, cc_R
    else:
        wm_L = (seg == WM_VAL_L)
        wm_R = (seg == WM_VAL_R)

        wm = wm_L + wm_R

        # Exclusion areas for wm
        #   Fornix
        vent_1_2 = fisin(seg, [4, 5, 43, 44])
        vent_3_dil6 = _dilate(seg == 14, 6)
        vent_123_raw = vent_1_2 | vent_3_dil6
        vent_123 = _close(vent_123_raw, 6)
        #   Cingulum
        cing_L = _close(fisin(seg, CINGULATE_VALS_L), 5)
        cing_R = _close(fisin(seg, CINGULATE_VALS_R), 5)
        cing = cing_L | cing_R
        #   Total
        exclusion_area = vent_123 | cing
        wm[exclusion_area] = False

        dil_wm_L = binary_dilation(wm_L)*wm
        dil_wm_R = binary_dilation(wm_R)*wm

        seed_cc = dil_wm_L & dil_wm_R

        cc_raw = _dilate(seed_cc, 10)
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
    and external/extreme capsule. Also convert "white matter hypo/hyper-intensities" (label 77) to
    corresponding latteralized WM.

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
        Caudate prvc:   78 (Caudate periventricular voxels)
        Thal. prvc:     79 (Thalamus periventricular voxels)
        In. cap. prvc:  80 (Internal capsule periventricular voxels)


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
        Cerebellum :    57
        Caudate prvc:   88 (Caudate periventricular voxels)
        Thal. prvc:     89 (Thalamus periventricular voxels)
        In. cap. prvc:  90 (Internal capsule periventricular voxels)

    Brainstem: 60

    '''
    import time
    # t = time.time()
    seg = convert_hyp_to_wm(seg)
    # print(f"Time for convert_hyp_to_wm: {time.time() - t:.2f} seconds")
    # t = time.time()
    seg_lobar = lobar_seg(seg)
    # print(f"Time for lobar_seg: {time.time() - t:.2f} seconds")
    # t = time.time()
    ic_L, ic_R = internal_caps(seg)
    # print(f"Time for internal_caps: {time.time() - t:.2f} seconds")
    # t = time.time()
    jxtc_L, jxtc_R = juxtacortical_wm(seg, 3)
    # print(f"Time for juxtacortical_wm: {time.time() - t:.2f} seconds")
    # t = time.time()
    pv_voxels = periventricular(seg, 2)
    # print(f"Time for periventricular: {time.time() - t:.2f} seconds")
    prvc_L = pv_voxels & (seg == WM_VAL_L)
    prvc_R = pv_voxels & (seg == WM_VAL_R)

    bgpv_L = pv_voxels & (seg_lobar == 40)
    thalpv_L = pv_voxels & (seg_lobar == 41)
    icpv_L = pv_voxels & ic_L
    bgpv_R = pv_voxels & (seg_lobar == 50)
    thalpv_R = pv_voxels & (seg_lobar == 51)
    icpv_R = pv_voxels & ic_R

    ic_L = ic_L & ~prvc_L
    ic_R = ic_R & ~prvc_R
    # t = time.time()
    ec_L, ec_R = ex_capsule(seg, (jxtc_L | jxtc_R | ic_L | ic_R))
    # print(f"Time for ex_capsule: {time.time() - t:.2f} seconds")
    # t = time.time()
    cc_L, cc_R = corpus_cal(seg)

    wm_L = (seg == WM_VAL_L)
    wm_R = (seg == WM_VAL_R)

    for lobe, vals in LOBAR_GM_LABELS_L.items():
        lobar = (seg_lobar == vals) & wm_L
        jxtc = lobar & jxtc_L
        prvc = lobar & prvc_L
        seg_lobar[lobar] = LOBAR_WM_LABELS_L[lobe][1]  # deep wm, set first to all wm as default
        seg_lobar[jxtc] = LOBAR_WM_LABELS_L[lobe][0]  # juxtacortical wm
        seg_lobar[prvc] = LOBAR_WM_LABELS_L[lobe][2]  # periventricular wm

    for _, vals in LOBAR_GM_LABELS_R.items():
        lobar = (seg_lobar == vals) & wm_R
        jxtc = lobar & jxtc_R
        prvc = lobar & prvc_R
        seg_lobar[lobar] = LOBAR_WM_LABELS_R[lobe][1]  # deep wm, set first to all wm as default
        seg_lobar[jxtc] = LOBAR_WM_LABELS_R[lobe][0]  # juxtacortical wm
        seg_lobar[prvc] = LOBAR_WM_LABELS_R[lobe][2]  # periventricular wm

    seg_lobar[ic_L] = OTHER_LABELS_L['Internal capsule']
    seg_lobar[ic_R] = OTHER_LABELS_R['Internal capsule']
    seg_lobar[ec_L] = OTHER_LABELS_L['Ext. capsule']
    seg_lobar[ec_R] = OTHER_LABELS_R['Ext. capsule']
    seg_lobar[cc_L] = OTHER_LABELS_L['Corpus callosum']
    seg_lobar[cc_R] = OTHER_LABELS_R['Corpus callosum']
    seg_lobar[bgpv_L] = OTHER_LABELS_L['BG prvc']
    seg_lobar[bgpv_R] = OTHER_LABELS_R['BG prvc']
    seg_lobar[icpv_L] = OTHER_LABELS_L['In. cap. prvc']
    seg_lobar[icpv_R] = OTHER_LABELS_R['In. cap. prvc']
    seg_lobar[thalpv_L] = OTHER_LABELS_L['Thal. prvc']
    seg_lobar[thalpv_R] = OTHER_LABELS_R['Thal. prvc']

    return seg_lobar

# %%

def hemispheric_seg(fs_seg, lobar_seg):
    """Return masks for left and right hemispheres (with lateral ventricles), left and right cerebellum, and brainstem."""
    new_seg = np.zeros_like(fs_seg)
    
    left_hemi_labels = [v for v in LOBAR_GM_LABELS_L.values()] + [v for k, v in OTHER_LABELS_L.items() if k != 'Cerebellum'] + [v for vals in LOBAR_WM_LABELS_L.values() for v in vals]
    right_hemi_labels = [v for v in LOBAR_GM_LABELS_R.values()] + [v for k, v in OTHER_LABELS_R.items() if k != 'Cerebellum'] + [v for vals in LOBAR_WM_LABELS_R.values() for v in vals]
    cereb_L = OTHER_LABELS_L['Cerebellum']
    cereb_R = OTHER_LABELS_R['Cerebellum']
    
    new_seg[fisin(lobar_seg, left_hemi_labels)] = 1
    new_seg[fisin(fs_seg, LAT_VENT_VALS_L)] = 1
    new_seg[fisin(lobar_seg, right_hemi_labels)] = 2
    new_seg[fisin(fs_seg, LAT_VENT_VALS_R)] = 2
    new_seg[fisin(lobar_seg, cereb_L)] = 3
    new_seg[fisin(lobar_seg, cereb_R)] = 4
    new_seg[lobar_seg == BRAINSTEM_LABEL] = 5
    return new_seg

#%%