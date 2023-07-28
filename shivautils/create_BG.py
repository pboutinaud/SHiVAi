'''
A script for creating BGs using Freesurfer segmentations. The BG is made by creating a box around the BG up to the insula.
INPUT: You must provide the createBGsliceMask function with:

-The path to the nifti image segmented by Freesurfer (e.g. aparc+aseg_out.nii.gz).
 
OUTPUT:
- Binary map of Basal Ganglia.
 
@author: atsuchida, iastafeva
@date: 2023-06-08
'''

# %%
from nipype import config, logging  
from nipype.pipeline.engine import  Node
from nipype.interfaces.utility import  Function


def createBGsliceMask(path_to_freesurfer_segm, out_fname_root=''):
    '''
    A function that takes freesurfer segmentations to create 
    BG slice mask (for VRS region classification). 
    
    Note that it will include some regions overlapping with hipp/midbrain.
    '''
    import os.path as op
    import numpy as np
    import nibabel as nib
    #from workflows.wf_utils.computations import _get_3d_img
    
    # First create BG & Insula 
    #segm_im =  _get_3d_img(path_to_freesurfer_segm)
    segm_dat = path_to_freesurfer_segm.get_fdata()
    bg_dat = segm_dat.copy()
    insula_dat = segm_dat.copy()
    # Regions that need to create BG and Insula:
    regions_to_create_bg = np.array([10,11,12,13,17,26,49,50,51,52,53,58])
    regions_to_create_insula = np.array([2035,1035])

    for i in regions_to_create_bg:
                                
        bg_dat[bg_dat==i] = 1             
    bg_dat[bg_dat!=1]=0
    
    for i in regions_to_create_insula:
        
        insula_dat[insula_dat==i] = 1
    
    insula_dat[insula_dat!=1]=0
    
    # Create box of BG           
    bg_box = np.zeros_like(bg_dat)
    bg_nonzero = np.nonzero(bg_dat)
    bg_ymin, bg_ymax = np.min(bg_nonzero[1]), np.max(bg_nonzero[1])
    bg_zmin, bg_zmax = np.min(bg_nonzero[2]), np.max(bg_nonzero[2])
    bg_box[:, bg_ymin: bg_ymax+1, bg_zmin: bg_zmax+1] = 1
               
    # create insula wall to confine bg box
    # Zero everything lower/higher than R/L insular mask in x direction
    for hemi in ['R', 'L']:
        ins_wall = np.zeros_like(insula_dat)
        hemi_ins = insula_dat.copy()
        # Zero everything on the other hemi
        mid_x = int(insula_dat.shape[0]/2)
        if hemi == 'R':
            hemi_ins[:mid_x, :, :] = 0
        else:
            hemi_ins[mid_x:, :, :] = 0
        ins_nonzero = np.nonzero(hemi_ins)
    
        # modify ins_wall to create a wall
        ins_ymin, ins_ymax = np.min(ins_nonzero[1]),np.max(ins_nonzero[1])
        for y in range(ins_ymin, ins_ymax + 1):
            z_vals = ins_nonzero[2][ins_nonzero[1] == y]
            if z_vals.size == 0:
                zmin, zmax = 0, 0
            else :
                zmin, zmax = np.min(z_vals), np.max(z_vals)
            for z in range(zmin, zmax + 1):
                x_vals_at_yz = ins_nonzero[0][(ins_nonzero[1] == y) & (ins_nonzero[2] == z)]
                if x_vals_at_yz.size != 0: # if empty reuse previous vals
                    x_min, x_max = np.min(x_vals_at_yz), np.max(x_vals_at_yz)

                if y == ins_ymin:
                    if z == zmin: # draw a box below/behind
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, :(y + 1), :(z + 1)] = 1
                        else:
                            ins_wall[:x_min, :(y + 1), :(z + 1)] = 1
                    if z == zmax: # draw a box above/behind
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, :(y + 1), z:] = 1
                        else:
                            ins_wall[:x_min, :(y + 1), z:] = 1
                    else: # draw a plane behind
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, :(y + 1), z] = 1
                        else:
                            ins_wall[:x_min, :(y + 1), z] = 1
                elif y == ins_ymax:
                    if z == zmin: # draw a box below/in front
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y:, :(z + 1)] = 1
                        else:
                            ins_wall[:x_min, y:, :(z + 1)] = 1
                    if z == zmax: # draw a box above/ in front
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y:, z:] = 1
                        else:
                            ins_wall[:x_min, y:, z:] = 1
                    else: # draw a plane in front
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y:, z] = 1
                        else:
                            ins_wall[:x_min, y:, z] = 1

                else:
                    if z == zmin: # draw a plane below
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y, :(z + 1)] = 1
                        else:
                            ins_wall[:x_min, y, :(z + 1)] = 1
                    if z == zmax: # draw a plane above
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y, z:] = 1
                        else:
                            ins_wall[:x_min, y, z:] = 1
                    else: # draw a line
                        if hemi == 'R':
                            ins_wall[(x_max + 1):, y, z] = 1
                        else:
                            ins_wall[:x_min, y, z] = 1

        # Use wall to zero every voxel in the wall
        bg_box[ins_wall == 1] = 0

    bg_roi_nii = nib.Nifti1Image(bg_box, path_to_freesurfer_segm.affine, path_to_freesurfer_segm.header)
    out_fname = '{}_BG_slice.nii.gz'.format(out_fname_root) if out_fname_root else 'BG_slice.nii.gz'

    return bg_roi_nii
   