from statistics import mean
import numpy as np
import pandas as pd
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import median
import os
from scipy.spatial.transform import Rotation
from scipy.io import loadmat


def get_mode(hist: np.array,
             edges: np.array,
             bins: int):
    """Get most frequent value in an numpy histogram composed by
    frequence (hist) and values (edges)

    Args:
        hist (np.array): frequence of values
        edges (np.array): different values possible in histogram
        bins (int): number of batchs

    Returns:
        mode (int): most frequent value
    """
    inf = int(0.2 * bins)
    sup = int(0.9 * bins)
    index = np.where(hist[inf:sup] == hist[inf:sup].max())
    mode = edges[inf+index[0]][0]

    return mode


def set_percentile(list_node: list,
                   list_percentile: list,
                   bins: int,
                   list_img_cohort: list) -> int:
    """Compared mode distribution for different percentile
    modalities in normalization and return percentile value most adapted
    at the given cohort image

    Args:
        list_node (list): list of nodes obtained after
        workflow with normalization
        list_percentile (list): lists of different percentile values to test
        bins (int) : number of batchs in cohorte image histogram
        list_img_cohort : list of nibabel image to compare with preprocessed image

    Returns:
        int: percentile value most adapted at the mode distribution cohort image
    """
    list_mode = []
    for i in list_node:
        # filters only nodes resulting from normalization
        if i.name == 'normalization':
            list_mode.append(i.result.outputs.mode)

    mode_by_percentile = []
    for i in range(len(list_percentile)):
        # create a table, create inside a table for each percentile value,
        # put the mode values of each image for the percentile value in
        # the corresponding table
        mode_by_percentile.append([])
        for j in range(i, len(list_mode), len(list_percentile)):
            mode_by_percentile[i].append(list_mode[j])

    for i in mode_by_percentile:
        mode_by_percentile[mode_by_percentile.index(i)] = mean(i)

    list_array_cohort = []
    for i in list_img_cohort:
        list_array_cohort.append(i.get_fdata().reshape(-1))

    list_mode_cohort = []
    for i in list_array_cohort:
        hist, edges = np.histogram(i, bins)
        mode = get_mode(hist, edges, bins)
        list_mode_cohort.append(mode)
    mode_mean_cohort = mean(list_mode_cohort)

    mode_fit = mode_by_percentile[min(range(len(mode_by_percentile)), key=lambda i: abs(mode_by_percentile[i]-mode_mean_cohort))]
    percentile_fit = list_percentile[mode_by_percentile.index(mode_fit)]

    return percentile_fit


def prediction_metrics(clusters_vol, brain_seg_vol,
                       region_dict={'Whole brain': [-1]},
                       prio_labels=[]):
    """Get metrics on sgmented biomarkers by brain region

    Args:
        clusters_vol (array): Labelled biomarker array (from the nifti file)
        brain_seg_vol (array): Brain segmentation (or brain mask), should be filled with integers
        region_dict (dict): Dict pairing the brain regions' name (key) with the numeric label (as int) to use when
                            counting the biomarkers.
                            '-1' denotes the region encompassing the whole brain segmentation > 0
                            (must be associated with the 'Whole brain' key)
    Returns:
        Dataframe: Labels and size of each cluster
        Dataframe: Summary metrics of the clusters 
            (sum voxel segmented, number of cluster, mean size cluster, 
            median size cluster, min size cluster, max size cluster)
        Array: Labelled clusters
    """

    if len(clusters_vol.shape) > 3:
        clusters_vol = clusters_vol.squeeze()
    if len(brain_seg_vol.shape) > 3:
        brain_seg_vol = brain_seg_vol.squeeze()

    # Associate a label with a region name
    swaped_region_dict = {val: reg for reg, val in region_dict.items()}
    if prio_labels:
        prio_dict = {reg: region_dict[reg] for reg in prio_labels}

    clust_labels, clust_size = np.unique(clusters_vol[clusters_vol > 0], return_counts=True)

    cluster_measures = pd.DataFrame(
        {'Biomarker labels': clust_labels,
         'Biomarker size': clust_size})

    regions = []
    biom_num = []
    biom_tot = []
    biom_mean = []
    biom_med = []
    biom_std = []
    biom_min = []
    biom_max = []
    # Default case: get whole brain metrics
    if 'Whole brain' in region_dict.keys():
        del region_dict['Whole brain']
        regions.append('Whole brain')
        biom_num.append(cluster_measures.shape[0])
        biom_tot.append(cluster_measures['Biomarker size'].sum())
        biom_mean.append(cluster_measures['Biomarker size'].mean())
        biom_med.append(cluster_measures['Biomarker size'].median())
        biom_std.append(cluster_measures['Biomarker size'].std())
        biom_min.append(cluster_measures['Biomarker size'].min())
        biom_max.append(cluster_measures['Biomarker size'].max())

    # Attribution of one region per cluster (i.e. the most represented region in each cluster = winner-takes-all)
    if len(region_dict):
        clust_reg = []
        for clust in clust_labels:
            seg_clust = brain_seg_vol[clusters_vol == clust]
            reg_in_clust, reg_count = np.unique(seg_clust, return_counts=True)  # There shouldn't be any 0 here
            max_count = reg_count.max()
            seg_attributed_label = np.random.choice(reg_in_clust[reg_count == max_count])  # takes equalities into account
            if prio_labels:  # Overriding the w-t-a approach for priority labels
                prio_ind = np.isin(reg_in_clust, list(prio_dict.values()))
                if prio_ind.any():
                    seg_attributed_label = np.random.choice(reg_in_clust[prio_ind])  # takes equalities into account

            if seg_attributed_label in swaped_region_dict.keys():
                seg_attributed = swaped_region_dict[seg_attributed_label]
            else:  # keeping the raw label if it's not part of the investigated regions
                if seg_attributed_label:  # Non-zero label
                    seg_attributed = f'Label_{seg_attributed_label}'
            clust_reg.append(seg_attributed)

        cluster_measures['Biomarker region'] = clust_reg

        regions_seg = [reg for reg in region_dict.keys() if reg in clust_reg]  # Sorted like in the input dict
        regions += regions_seg

        for reg in regions_seg:
            clusts_in_reg = cluster_measures.loc[cluster_measures['Biomarker region'] == reg]
            biom_num.append(clusts_in_reg.shape[0])
            biom_tot.append(clusts_in_reg['Biomarker size'].sum())
            biom_mean.append(clusts_in_reg['Biomarker size'].mean())
            biom_med.append(clusts_in_reg['Biomarker size'].median())
            biom_std.append(clusts_in_reg['Biomarker size'].std())
            biom_min.append(clusts_in_reg['Biomarker size'].min())
            biom_max.append(clusts_in_reg['Biomarker size'].max())

    cluster_stats = pd.DataFrame(
        {'Region': regions,
         'Number of biomarkers': biom_num,
         'Total Biomarker volume': biom_tot,
         'Mean Biomarker volume': biom_mean,
         'Median Biomarker volume': biom_med,
         'StD Biomarker volume': biom_std,
         'Min Biomarker volume': biom_min,
         'Max Biomarker volume': biom_max})

    return cluster_measures, cluster_stats


def extreme_val(series: pd.Series) -> float:
    '''
    Returns the threshold define the extreme values in a pandas series.
    Extreme values are equal to:
        either 3 times the inter-quartile range from the upper quartile (arbitrary)
        or the 95th percentile
    depending on which is bigger
    '''
    return max(
        series.quantile(0.75) + 3*(series.quantile(0.75) - series.quantile(0.25)),
        series.quantile(0.95)
    )


def violinplot_from_census(census_csv: str, resolution: tuple, pred: str):
    census_df = pd.read_csv(census_csv)
    scale = resolution[0] * resolution[1] * resolution[2]
    census_df['Biomarker size ($mm^3$)'] = census_df['Biomarker size'] * scale
    save_name = f'{pred}_census_plot.svg'
    plt.ioff()
    if 'Biomarker region' not in census_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        # sns.stripplot(census_df, y='Biomarker size')  # replaced swamplot
        sns.histplot(census_df, x='Biomarker size ($mm^3$)', ax=ax)
        plt.title(f'{pred} size distribution')
        plt.tight_layout()
        plt.savefig(save_name, format='svg')
        plt.close(fig)
    else:
        # Change all non-identified regions by "Other"
        FS_id = ['FreeSurfer_' in reg for reg in census_df['Biomarker region']]
        census_df.loc[FS_id, 'Biomarker region'] = 'Other'
        reg_nb = census_df['Biomarker region'].nunique()
        fig, ax = plt.subplots(figsize=(2+0.5*reg_nb, 6))

        # Detect outliers and display them as dots
        brain_regions = census_df['Biomarker region'].unique()
        extreme_dict = {
            reg: extreme_val(census_df.loc[census_df['Biomarker region'] == reg, 'Biomarker size ($mm^3$)']) for reg in brain_regions
        }
        census_df['isXtreme'] = census_df.apply(lambda row: row['Biomarker size ($mm^3$)'] > extreme_dict[row['Biomarker region']], axis=1)

        # Detect regions with low biomarker count to display as swarmplot
        reg_swarm = [reg for reg in brain_regions if (census_df['Biomarker region'] == reg).sum() < 10]
        census_df['swarm'] = census_df.apply(lambda row: row['Biomarker region'] in reg_swarm, axis=1)

        # Set the palette so that the dots and the violins are the same color
        colors = sns.color_palette("hls", len(brain_regions))
        my_palette = {reg: colors[i] for i, reg in enumerate(brain_regions)}

        # Plots
        sns.violinplot(census_df.loc[~census_df['isXtreme'] & ~census_df['swarm']], y='Biomarker size ($mm^3$)', x='Biomarker region',
                       hue='Biomarker region', cut=0, bw_adjust=0.7, palette=my_palette, log_scale=True,
                       ax=ax)
        sns.swarmplot(census_df.loc[census_df['isXtreme']], y='Biomarker size ($mm^3$)', x='Biomarker region',
                      hue='Biomarker region', alpha=0.8, palette=my_palette, log_scale=True,
                      ax=ax)
        sns.swarmplot(census_df.loc[census_df['swarm']], y='Biomarker size ($mm^3$)', x='Biomarker region',
                      hue='Biomarker region', palette=my_palette, log_scale=True,
                      ax=ax)
        plt.title(f'{pred} size distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_name, format='svg')
        plt.close(fig)
    return os.path.abspath(save_name)


def get_mask_regions(img: nb.Nifti1Image,
                     list_labels_regions: list):
    """Filter in a regions segmented images regions of interest specified
    in 'list_labels_regions'

    Args:
        img (nb.Nifti1Image): image segmented by regions 
        list_labels_regions (list): list of int corresponding Fresurfer numerotation

    Returns:
        nb.Nifti1Image:segmented image with specific regions
    """

    import nibabel as nb
    import numpy as np

    pred_img_array = img.get_fdata()
    mask = np.isin(pred_img_array, list_labels_regions).astype(int)
    mask_regions = nb.Nifti1Image(mask, img.affine, img.header)

    return mask_regions


def transf_from_affine(mat_file: str):
    """
    Import the affine transformation from a .mat file generated by an ANTs registration
    and output the norm of its rotation (in degrees) and translation
    """
    # import numpy as np
    # from scipy.spatial.transform import Rotation
    # from scipy.io import loadmat

    mat = loadmat(mat_file)
    key_name = [k for k in mat if 'AffineTransform_' in k][0]  # AffineTransform_*_3_3
    rot = mat[key_name][0:9].reshape((3, 3))
    trans = mat[key_name][9:12]
    r = Rotation.from_matrix(rot)
    r_tot = np.abs(r.as_euler('xyz', degrees=True)).sum()
    t_norm = np.linalg.norm(trans)
    return r_tot, t_norm
