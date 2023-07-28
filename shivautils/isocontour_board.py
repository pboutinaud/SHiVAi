'''
A script to create the edges based on: https://github.com/neurolabusc/PyDog/blob/main/dog.py to check the coregistration.

INPUT: You must provide the create_edges function with:
 
 -Path to the image that was coregistered (the image must be cropped and the intensity normalized)
 -Path to the reference image on which the images have been coregistered (the reference image must be cropped and its intensity normalized)
 -Cropped brain mask (in the same space as reference image)
 -Optional: number of slices (default 8) for PNG output
 
 OUTPUT: the create_edges function will produce one PNG per subject with the reference image in the background 
 and the edges of the given image on top.

EXAMPLE: If you want to check the FLAIR coregistration on T1w, your image will be cropped and the intensity normalized FLAIR image, 
reference image - cropped and intensity normalized T1w image
Brain mask in T1w space and cropped.

@autor: iastafeva
@date: 24-04-2023
'''

def create_edges(path_image, path_ref_image, path_brainmask, nb_of_slices=8):
    
    import nibabel as nib
    from scipy.ndimage import (
        gaussian_filter1d,
        distance_transform_edt
    )
    import numpy as np
    import os.path as op
    import warnings
    import math
    #skimage package is "scikit-image"
    import skimage  
    from scipy import ndimage
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Creating edges using: https://github.com/neurolabusc/PyDog/blob/main/dog.py
    
    def dehaze(img,brainmask, level, verbose=0):
        """use Otsu to threshold https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
            n.b. threshold used to mask image: dark values are zeroed, but result is NOT binary
        Parameters
        ----------
        img : Niimg-like object
            Image(s) to run DoG on (see :ref:`extracting_data`
            for a detailed description of the valid input types).
        level : int
            value 1..5 with larger values preserving more bright voxels
            dark_classes/total_classes
                1: 3/4
                2: 2/3
                3: 1/2
                4: 1/3
                5: 1/4
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """
        fdata = img.get_fdata()
        brainmask = brainmask.get_fdata()
        fdata = fdata*brainmask 
        level = max(1, min(5, level))
        n_classes = abs(3 - level) + 2
        dark_classes = 4 - level
        dark_classes = max(1, min(3, dark_classes))
        thresholds = skimage.filters.threshold_multiotsu(fdata, n_classes)
        thresh = thresholds[dark_classes - 1]
    
        fdata[fdata < thresh] = 0
        return fdata
    
    def _fast_smooth_array(arr):
        """Simple smoothing which is less computationally expensive than
        applying a Gaussian filter.
        Only the first three dimensions of the array will be smoothed. The
        filter uses [0.2, 1, 0.2] weights in each direction and use a
        normalisation to preserve the local average value.
        Parameters
        ----------
        arr : :class:`numpy.ndarray`
            4D array, with image number as last dimension. 3D arrays are
            also accepted.
        Returns
        -------
        :class:`numpy.ndarray`
            Smoothed array.
        Notes
        -----
        Rather than calling this function directly, users are encouraged
        to call the high-level function :func:`smooth_img` with
        `fwhm='fast'`.
        """
        neighbor_weight = 0.2
        # 6 neighbors in 3D if not on an edge
        nb_neighbors = 6
        # This scale ensures that a uniform array stays uniform
        # except on the array edges
        scale = 1 + nb_neighbors * neighbor_weight

        # Need to copy because the smoothing is done in multiple statements
        # and there does not seem to be an easy way to do it in place
        smoothed_arr = arr.copy()
        weighted_arr = neighbor_weight * arr

        smoothed_arr[:-1] += weighted_arr[1:]
        smoothed_arr[1:] += weighted_arr[:-1]
        smoothed_arr[:, :-1] += weighted_arr[:, 1:]
        smoothed_arr[:, 1:] += weighted_arr[:, :-1]
        smoothed_arr[:, :, :-1] += weighted_arr[:, :, 1:]
        smoothed_arr[:, :, 1:] += weighted_arr[:, :, :-1]
        smoothed_arr /= scale

        return smoothed_arr

    def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
        """Smooth images by applying a Gaussian filter.
        Apply a Gaussian filter along the three first dimensions of `arr`.
        Parameters
        ----------
        arr : :class:`numpy.ndarray`
            4D array, with image number as last dimension. 3D arrays are also
            accepted.
        affine : :class:`numpy.ndarray`
            (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
            are also accepted (only these coefficients are used).
            If `fwhm='fast'`, the affine is not used and can be None.
        fwhm : scalar, :class:`numpy.ndarray`/:obj:`tuple`/:obj:`list`, 'fast' or None, optional
            Smoothing strength, as a full-width at half maximum, in millimeters.
            If a nonzero scalar is given, width is identical in all 3 directions.
            A :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` must have 3 elements,
            giving the FWHM along each axis.
            If any of the elements is zero or None, smoothing is not performed
            along that axis.
            If  `fwhm='fast'`, a fast smoothing will be performed with a filter
            [0.2, 1, 0.2] in each direction and a normalisation
            to preserve the local average value.
            If fwhm is None, no filtering is performed (useful when just removal
            of non-finite values is needed).
        ensure_finite : :obj:`bool`, optional
            If True, replace every non-finite values (like NaNs) by zero before
            filtering. Default=True.
        copy : :obj:`bool`, optional
            If True, input array is not modified. True by default: the filtering
            is not performed in-place. Default=True.
        Returns
        -------
        :class:`numpy.ndarray`
            Filtered `arr`.
        Notes
        -----
        This function is most efficient with arr in C order.
        """
        # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
        # See issue #1537
        if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
            warnings.warn("The parameter 'fwhm' for smoothing is specified "
                        "as {0}. Setting it to None "
                        "(no smoothing will be performed)"
                        .format(fwhm))
            fwhm = None
        if arr.dtype.kind == 'i':
            if arr.dtype == np.int64:
                arr = arr.astype(np.float64)
            else:
                arr = arr.astype(np.float32)  # We don't need crazy precision.
        if copy:
            arr = arr.copy()
        if ensure_finite:
            # SPM tends to put NaNs in the data outside the brain
            arr[np.logical_not(np.isfinite(arr))] = 0
        if isinstance(fwhm, str) and (fwhm == 'fast'):
            arr = _fast_smooth_array(arr)
        elif fwhm is not None:
            fwhm = np.asarray([fwhm]).ravel()
            fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])
            affine = affine[:3, :3]  # Keep only the scale part.
            fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            #n.b. FSL specifies blur in sigma, SPM in FWHM
            # FWHM = sigma*sqrt(8*ln(2)) = sigma*2.3548.
            #convert fwhm to sd in voxels see https://github.com/0todd0000/spm1d
            fwhmvox = fwhm / vox_size
            sd = fwhmvox / math.sqrt(8 * math.log(2))
            for n, s in enumerate(sd):
                if s > 0.0:
                    gaussian_filter1d(arr, s, output=arr, axis=n)
        return arr

    def binary_zero_crossing(fdata):
        """binarize (negative voxels are zero)
        Parameters
        ----------
        fdata : numpy.memmap from Niimg-like object
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """
        edge = np.where(fdata > 0.0, 1, 0)
        edge = distance_transform_edt(edge)
        edge[edge > 1] = 0
        edge[edge > 0] = 1
        edge = edge.astype('uint8')
        return edge
    
    def difference_of_gaussian(fdata, affine, fwhmNarrow, verbose=0):
        """Apply Difference of Gaussian (DoG) filter.
        https://en.wikipedia.org/wiki/Difference_of_Gaussians
        https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm
        D. Marr and E. C. Hildreth. Theory of edge detection. Proceedings of the Royal Society, London B, 207:187-217, 1980
        Parameters
        ----------
        fdata : numpy.memmap from Niimg-like object
        affine : :class:`numpy.ndarray`
            (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
            are also accepted (only these coefficients are used).
        fwhmNarrow : int
            Narrow kernel width, in millimeters. Is an arbitrary ratio of wide to narrow kernel.
                human cortex about 2.5mm thick
                Large values yield smoother results
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """

        #Hardcode 1.6 as ratio of wide versus narrow FWHM
        # Marr and Hildreth (1980) suggest narrow to wide ratio of 1.6
        # Wilson and Giese (1977) suggest narrow to wide ratio of 1.5
        fwhmWide = fwhmNarrow * 1.6
        #optimization: we will use the narrow Gaussian as the input to the wide filter
        fwhmWide = math.sqrt((fwhmWide*fwhmWide) - (fwhmNarrow*fwhmNarrow))
    
        imgNarrow = _smooth_array(fdata, affine, fwhmNarrow)
        imgWide = _smooth_array(imgNarrow, affine, fwhmWide)
        img = imgNarrow - imgWide
        img = binary_zero_crossing(img)
        return img 
    
    def dog_img(img,brainmask, fwhm, verbose=0):
        """Find edges of a NIfTI image using the Difference of Gaussian (DoG).
        Parameters
        ----------
        img : Niimg-like object
            Image(s) to run DoG on (see :ref:`extracting_data`
            for a detailed description of the valid input types).
        fwhm : int
            Edge detection strength, as a full-width at half maximum, in millimeters.
        verbose : :obj:`int`, optional
            Controls the amount of verbosity: higher numbers give more messages
            (0 means no messages). Default=0.
        Returns
        -------
        :class:`nibabel.nifti1.Nifti1Image`
        """
        
    
        dog_fdata = dehaze(img,brainmask, 3, verbose)
        
        
        dog = difference_of_gaussian(dog_fdata, img.affine, fwhm, verbose)
        
        
        out_img = nib.Nifti1Image(dog, img.affine, img.header)
        #update header
        out_img.header.set_data_dtype(np.uint8)  
        out_img.header['intent_code'] = 0
        out_img.header['scl_slope'] = 1.0
        out_img.header['scl_inter'] = 0.0
        out_img.header['cal_max'] = 0.0
        out_img.header['cal_min'] = 0.0
    
    
        return out_img
        
    #load data 
    dataNii = nib.load(path_ref_image)    
    ref_image = dataNii.get_fdata(dtype=np.float32)
    image = nib.load(path_image)  
    brainmask =  nib.load(path_brainmask)
    
    # get edges in a NIfTI image 
    dog_imported_img = dog_img(image,brainmask, fwhm=3, verbose=1)
    image =  dog_imported_img.get_fdata(dtype=np.float32)  
  
    #padd the data for nice subplots
    pad_ref_image = np.zeros((ref_image.shape[1], ref_image.shape[1], ref_image.shape[1]))
    pad_ref_image[int(abs(ref_image.shape[1]-ref_image.shape[0])/2):ref_image.shape[0]+int(abs(ref_image.shape[1]-ref_image.shape[0])/2), :, 
                  int(abs(ref_image.shape[1]-ref_image.shape[2])/2):ref_image.shape[2]+int(abs(ref_image.shape[1]-ref_image.shape[2])/2)] = ref_image
    ref_image = pad_ref_image[..., np.newaxis]
 
    pad_image = np.zeros((image.shape[1], image.shape[1], image.shape[1]))
    pad_image[int(abs(image.shape[1]-image.shape[0])/2):image.shape[0]+int(abs(image.shape[1]-image.shape[0])/2), :, 
              int(abs(image.shape[1]-image.shape[2])/2):image.shape[2]+int(abs(image.shape[1]-image.shape[2])/2)] = image   
    image = pad_image[..., np.newaxis]
 
    # creating cmap to overlay reference image and given image
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
    sli_image = []
    sli_ref_image = [] 
    X_Y_Z = []
     
    for i in range(3):
        for j in range(14, image.shape[i] - 20, 4):
        
            if i==0:         
                sli_image.append(image[int(j), :, :])   
                sli_ref_image.append(ref_image[int(j), :, :])    
                X_Y_Z.append(int(j))  
            if i==1:  
                sli_image.append(image[:, int(j), :]) 
                sli_ref_image.append(ref_image[:, int(j), :])
            if i==2:  
                sli_image.append(image[:, :, int(j)])  
                sli_ref_image.append(ref_image[:, :, int(j)])

    figure, axis = plt.subplots(nb_of_slices, 9, figsize=(160, 80))
    figure.patch.set_facecolor('k')
    count = 90
    for j in range(9):

            for i in range(nb_of_slices):   
                axis[i,j].imshow(
                    ndimage.rotate(sli_ref_image[count],90)[..., 0],
                    cmap='gray')           
                axis[i,j].imshow(
                    ndimage.rotate(sli_image[count],90)[..., 0],
                    cmap=my_cmap)

                label = axis[i, j].set_xlabel('k = ' + str(X_Y_Z[count - 90]))
                label.set_fontsize(30)  # Définir la taille de la police du label
                label.set_color('white')
                axis[i, j].get_xaxis().set_ticks([])
                axis[i, j].get_yaxis().set_ticks([])
                
                count += 1

    figure.tight_layout() 
    
    QC_coreg = 'QC_coreg_edges.png'        
    plt.savefig('QC_coreg_edges.png')       
    return (op.abspath(QC_coreg))
