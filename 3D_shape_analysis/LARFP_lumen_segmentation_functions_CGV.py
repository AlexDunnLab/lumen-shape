#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:11:52 2018
Functions for LARFP_lumen_segmentation_CGV.py

@author: clauvasq
"""
# import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
io.use_plugin('tifffile')
from skimage.filters import threshold_otsu
from skimage import filters
from skimage import morphology
from scipy import ndimage
import cv2

pw_desktop = '/Users/clauvasq/Desktop/'
def med_filter(stack, med_sel=7):
    """
    Applies median filter to image stack by z-slice with disk selem of 7, by
    defaultself.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]
    med_sel : int, default = 7
        size of selem to use (disk selem!)

    Returns
    -------
    med_stack : ndarray
        median filtered stack
    """
    med_stack = stack.copy()
    for i in range(len(stack)):
        z_slice = stack[i, :, :]
        med_slice = filters.median(z_slice, selem=morphology.disk(med_sel))
        med_stack[i, :, :] = med_slice
    return med_stack

def otsu_morph_seg(stack, hole_size=2048, opt=1):
    """
    Function segments z-stack on a per-slice basis.

    This is to try to remove errors of lumen segmentation caused by actin loops in
    in the lumen

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X], otsu-thresholded

    hole_size : int, default = 2028
        for filling in holes in otsu segmentation

    opt : 1, 2, 3
        option 1: otsu segment each slice, then remove small holes
        option 2: morph. opening (dilation then erosion) w/ selem=disk(9), then
        remove small holes
        option 3: for squiggly lumens, closing, remove small holes, erosion, and
        then opening

    Returns
    -------
    bin_stack : ndarray
        ndarry of cyst, with segmented lumen, hopefully
    """
    bin_stack = stack.copy()
    for i in range(len(stack)):
        z_slice = stack[i, :, :]
        if np.count_nonzero(z_slice) > 0:
            if opt == 1:
                otsu_slice = threshold_otsu(z_slice)
                bin_slice = z_slice > otsu_slice
                bin1 = np.array(morphology.remove_small_holes(bin_slice, hole_size),
                                dtype=np.uint8)
            elif opt == 2:
                bin_slice = morphology.binary_closing(z_slice, selem=morphology.disk(9))
                bin1 = np.array(morphology.remove_small_holes(bin_slice, hole_size),
                                dtype=np.uint8)
            else:
                z1 = morphology.binary_closing(z_slice)
                z2 = morphology.remove_small_holes(z1, hole_size)
                z3 = morphology.binary_erosion(z2, selem=morphology.disk(5))
                bin1 = morphology.binary_opening(z3, selem=morphology.disk(2))
            bin_stack[i, :, :] = bin1
        else:
            bin_stack[i, :, :] = np.zeros(np.shape(stack[i, :, :]))
    return bin_stack

def two_cell_seg(stack, hole_size=128, disk_size=3, obj_size=100):
    """
    Function segments z-stack w/ 2 cells and bright actin enrichment/lumen
    Otsu thresholds max projection of stack, then opening on a per slice basis
    and remove objects.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]

    hole_size : int, default=128

    disk_size : int, default = 3
        size of radius of selem for morphological opening

    obj_size : int, default = 100
        size of minimum sized object, anything smaller will be removed.

    Returns
    -------
    bin_stack : ndarray
        ndarry of cyst, with segmented lumen, hopefully
    """
    bin_stack = stack.copy()
    otsu = threshold_otsu(np.max(stack, 0))
    otsu_stack = np.array(stack > otsu, dtype=np.uint8)
    for i in range(len(stack)):
        im_otsu = otsu_stack[i, :, :]
        morph0 = morphology.remove_small_holes(im_otsu, hole_size)
        morph1 = morphology.closing(morph0, selem=morphology.disk(disk_size))
        morph2 = morphology.remove_small_objects(morph1, min_size=obj_size)
        bin_stack[i, :, :] = morph2
    bin_stack = np.array(bin_stack, dtype=np.uint8)
    return bin_stack

def dim_signal_seg(stack, med_sel=5, otsu_factor=1.5, hole_size=1024, obj_size=500):
    """
    Function segments z-stack w/ of cells with dim signal.
    Applies median filter, then does otsu threshold, and threshold 2*otsu value
    on a per slice basis. Then does morphological operations to fill in lumen
    and remove other objects.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]

    med_sel : int, default=5
        size of selem to use (disk selem!)

    otsu_factor : float, default=1.5
        multiplier for otsu value to threshold by

    hole_size : int, default=1024
        size of holes to remove for morphological processes

    obj_size : int, default = 500
        size of minimum sized object, anything smaller will be removed.

    Returns
    -------
    bin_stack : ndarray
        ndarry of cyst, with segmented lumen, hopefully
    """
    bin_stack = stack.copy()
    otsu = threshold_otsu(np.max(stack, 0))
    otsu_stack = np.array(stack > otsu_factor*otsu, dtype=np.uint8)
    for i in range(len(stack)):
        z_slice = otsu_stack[i, :, :]
        # med_slice = filters.median(z_slice, selem=morphology.disk(med_sel))
        # otsu = threshold_otsu(med_slice)
        # otsu_slice = med_slice > otsu_factor*otsu
        morph1 = morphology.remove_small_holes(z_slice, hole_size)
        morph2 = morphology.remove_small_objects(morph1, min_size=obj_size)
        bin_stack[i, :, :] = morph2
    bin_stack = np.array(bin_stack, dtype=np.uint8)
    return bin_stack

def eight_bit_seg(stack, hole_size=2048):
    """
    Function segments lumen of **8-bit** z-stack on a per-slice basis.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X], otsu-thresholded

    hole_size : int, default = 2048
        for filling in holes in otsu segmentation

    Returns
    -------
    bin_stack : ndarray
        ndarry of cyst, with segmented lumen, hopefully
    """
    bin_stack = stack.copy()
    for i in range(len(stack)):
        z_slice = stack[i, :, :]
        z1 = morphology.binary_dilation(z_slice)
        z2 = morphology.remove_small_holes(z1, hole_size)
        z3 = morphology.binary_erosion(z2, selem=morphology.disk(4))
        bin_stack[i, :, :] = z3
    return bin_stack

def eight_bit_cyst_seg(stack, disk_size=7):
    bin_stack = stack.copy()
    for i in range(len(stack)):
        z_slice = stack[i, :, :]
        z1 = z_slice > z_slice.mean()
        z2 = morphology.binary_closing(z1, selem=morphology.disk(3))
        z3 = morphology.remove_small_holes(z2, min_size=8192)
        z4 = morphology.binary_erosion(z3, selem=morphology.disk(disk_size))
        z5 = morphology.remove_small_objects(z4, 2048)
        bin_stack[i, :, :] = z5
    return bin_stack
def lumen_post(stack, disk_size=5):
    """
    binary erosion on image stack of indicated disk size
    use after contour finding on lumen segmentation, occasionally
    """
    disk_sel = morphology.disk(disk_size)
    post_stack = np.copy(stack)
    for i in range(len(stack)):
        post_stack[i, :, :] = morphology.binary_erosion(stack[i, :, :],
                                                        selem=disk_sel)
    return post_stack

def cyst_edge(stack, low_pct=0.01, hi_pct=0.99, plot=False):
    """
    Determines edges of cyst in z-slices
    Does this by projecting stack in Y (or X). Then, takes mean along X (or Y),
    giving line projection of intensity in Z from both X and Y directions. Then,
    gets cumuluative sum, and uses low_pct and hi_pct as lower and upper bounds,
    respectively, for z-slices. Uses minimum from Y and X for lower, and maximum
    of Y and X for upper.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]
    low_pct : 0-1
        lower bound of area under intensity curve
    hi_pct : 0-1
        upper bound of area under intensity curve
    plot : default = False
        if True, then plots out projections, mean line projection, and cumsum

    Returns
    -------
    z_lower, z_upper: int, int
        bounds, inclusive of z-slices that include cyst
    """

    # project image along Y and X, respectively
    im_projY = stack.sum(1)
    im_projX = stack.sum(2)

    # take mean along X and Y, respecitively
    lineProjY = np.mean(im_projY, 1)
    lineProjX = np.mean(im_projX, 1)

    # determine edges of peak = find area under curve, and find where each
    # reach certain pct of total areas
    lineProjY_csum = np.cumsum(lineProjY)
    lineProjX_csum = np.cumsum(lineProjX)
    Y_csum = lineProjY_csum[-1]
    X_csum = lineProjX_csum[-1]
    z_fromY = [np.where(lineProjY_csum > low_pct*Y_csum)[0][0],
               np.where(lineProjY_csum > hi_pct*Y_csum)[0][0]]
    z_fromX = [np.where(lineProjX_csum > low_pct*X_csum)[0][0],
               np.where(lineProjX_csum > hi_pct*X_csum)[0][0]]

    # find min of z from Y and  X, and find max  z from Y and X
    z_lower = min(z_fromY[0], z_fromX[0])
    z_upper = min(z_fromY[1], z_fromX[1])

    # plotting
    if plot == True:
        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0, 0].imshow(im_projY)
        ax[1, 0].imshow(im_projX)
        ax[0, 1].plot(lineProjY)
        ax[1, 1].plot(lineProjX)
        ax[0, 2].plot(lineProjY_csum)
        ax[1, 2].plot(lineProjX_csum)
    # take mean along X, to determine z-coordinates
    # make
    return z_lower, z_upper

def bgsub_zyx_morph(stack, sel_e=7, hole_size=2048, obj_size=512, sel_e2=5, opt=2):
    """
    Segmentation of whole cyst via background subtraction in z direction,
    y direction, and x direction.
    (1) median filters
    (2) background subtractions
    (3) morphological operations to clean up
    (4) medain filter again to smooth segmentation

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]

    sel_e : int, default = 7
        size of selem in disk for first morphological erosion

    hole_size : int, default = 2048
        size of holes to remove

    obj_size : int, default = 512
        size of objects to remove

    sel_e2 : int, defualt = 5
        size of selem in disk for second morphological erosion

    opt : 1 or 2, defualt = 2
        different order of morphological operations, option 2 seems to work
        better...

    Returns
    -------
    med_stack : ndarray
        ndarry of cyst, with segmented cyst

    """
    # median filter
    med_stack = med_filter(stack, med_sel=3)
    Z, Y, X = stack.shape
    z_fgm = np.copy(stack)
    y_fgm = np.copy(stack)
    x_fgm = np.copy(stack)
    # initialize bacground subtraction
    # go through each z_slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for z in range(Z):
        frame = med_stack[z, :, :]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        z_fgm[z, :, :] = fgmask_2
    # go through each y-slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for y in range(Y):
        frame = med_stack[:, y, :]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        y_fgm[:, y, :] = fgmask_2
    # go through each x-slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for x in range(X):
        frame = med_stack[:, :, x]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        x_fgm[:, :, x] = fgmask_2
    # sum up all pixels for differnet background subtractions
    all_fgm = z_fgm + y_fgm + x_fgm
    # otsu threshold
    ot_fgm = all_fgm.copy()
    ot_th = threshold_otsu(all_fgm)
    ot_fgm = np.array(all_fgm > ot_th, dtype=np.uint8)
    # morphological operations
    morph_fgm = ot_fgm.copy()
    if opt == 1:
        for z in range(Z):
            z_slice = ot_fgm[z, :, :]
            z_erode  = morphology.binary_erosion(z_slice, selem=morphology.disk(sel_e))
            z_fill = morphology.remove_small_holes(z_erode, hole_size)
            z_rmobj = morphology.remove_small_objects(z_fill, min_size=obj_size)
            z_erode2 = morphology.binary_erosion(z_rmobj, selem=morphology.disk(sel_e2))
            morph_fgm[z, :, :] = z_erode2
    if opt == 2:
        for z in range(Z):
            z_slice = ot_fgm[z, :, :]
            z_fill = morphology.remove_small_holes(z_slice, hole_size)
            z_erode  = morphology.binary_erosion(z_fill, selem=morphology.disk(sel_e))
            z_rmobj = morphology.remove_small_objects(z_erode, min_size=obj_size)
            morph_fgm[z, :, :] = z_rmobj
    morph_fgm = np.array(morph_fgm, dtype=np.uint8)
    med_stack = med_filter(morph_fgm, med_sel=7)
    return med_stack
# Andrew - this is the function used for hESC colony segmenation :D
def bgsub_zyx_otsu(stack):
    """
    Segmentation of whole colony via background subtraction in z direction,
    y direction, and x direction.
    (1) median filters
    (2) background subtractions
    (3) otsu filter

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]

    sel_e : int, default = 7
        size of selem in disk for first morphological erosion
    Returns
    -------
    ot_fgm : ndarray
        ndarry of preliminary segmenation of colony
    """
    # median filter
    med_stack = med_filter(stack, med_sel=3)
    Z, Y, X = stack.shape
    z_fgm = np.copy(stack)
    y_fgm = np.copy(stack)
    x_fgm = np.copy(stack)
    # initialize bacground subtraction
    # go through each z_slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for z in range(Z):
        frame = med_stack[z, :, :]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        z_fgm[z, :, :] = fgmask_2
    # go through each y-slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for y in range(Y):
        frame = med_stack[:, y, :]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        y_fgm[:, y, :] = fgmask_2
    # go through each x-slice, bkg subtract
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for x in range(X):
        frame = med_stack[:, :, x]
        fgmask = fgbg.apply(frame)
        fgmask_2 = np.array(fgmask > 0, dtype=np.uint8)
        x_fgm[:, :, x] = fgmask_2
    # sum up all pixels for differnet background subtractions
    all_fgm = z_fgm + y_fgm + x_fgm
    # otsu threshold
    ot_fgm = all_fgm.copy()
    ot_th = threshold_otsu(all_fgm)
    ot_fgm = np.array(all_fgm > ot_th, dtype=np.uint8)

    return ot_fgm
def cyst_edge(stack, low_pct=0.01, hi_pct=0.99, plot=False):
    """
    Determines edges of cyst in z-slices
    Does this by projecting stack in Y (or X). Then, takes mean along X (or Y),
    giving line projection of intensity in Z from both X and Y directions. Then,
    gets cumuluative sum, and uses low_pct and hi_pct as lower and upper bounds,
    respectively, for z-slices. Uses minimum from Y and X for lower, and maximum
    of Y and X for upper.

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst with dimensions [Z, Y, X]
    low_pct : 0-1
        lower bound of area under intensity curve
    hi_pct : 0-1
        upper bound of area under intensity curve
    plot : default = False
        if True, then plots out projections, mean line projection, and cumsum

    Returns
    -------
    z_lower, z_upper: int, int
        bounds, inclusive of z-slices that include cyst
    """

    # project image along Y and X, respectively
    im_projY = stack.sum(1)
    im_projX = stack.sum(2)

    # take mean along X and Y, respecitively
    lineProjY = np.mean(im_projY, 1)
    lineProjX = np.mean(im_projX, 1)

    # determine edges of peak = find area under curve, and find where each
    # reach certain pct of total areas
    lineProjY_csum = np.cumsum(lineProjY)
    lineProjX_csum = np.cumsum(lineProjX)
    Y_csum = lineProjY_csum[-1]
    X_csum = lineProjX_csum[-1]
    z_fromY = [np.where(lineProjY_csum > low_pct*Y_csum)[0][0],
               np.where(lineProjY_csum > hi_pct*Y_csum)[0][0]]
    z_fromX = [np.where(lineProjX_csum > low_pct*X_csum)[0][0],
               np.where(lineProjX_csum > hi_pct*X_csum)[0][0]]

    # find min of z from Y and  X, and find max  z from Y and X
    z_lower = min(z_fromY[0], z_fromX[0])
    z_upper = min(z_fromY[1], z_fromX[1])

    # plotting
    if plot == True:
        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0, 0].imshow(im_projY)
        ax[1, 0].imshow(im_projX)
        ax[0, 1].plot(lineProjY)
        ax[1, 1].plot(lineProjX)
        ax[0, 2].plot(lineProjY_csum)
        ax[1, 2].plot(lineProjX_csum)
    # take mean along X, to determine z-coordinates
    # make
    return z_lower, z_upper

def cyst_post(stack, disk_size=5, disk_size_y=1, seq_erosions=True):
    """
    post contour finding of cyst segmentation. Goes through stack in z, erodes,
    then smoothes. Then goes through stack y, erodes, then smooths. Then, based
    on depth of z, applies larger erosions (larger z-slices have larger selems)

    Parameters
    ----------
    stack : ndarray
        ndarray of cyst segmenation with dimensions [Z, Y, X]

    disk_size : int, default = 5
        default size of selem for z-based erosion & smoothing

    disk_size_y : int, default = 1
        default size of selem for y-based erosion & smoothing

    Returns
    -------
    stack3 : ndarray
        ndarray of cyst segmenation
    """
    # searches through stack in z and applies erosion, and then smooths w/
    # median filter
    stack1 = np.copy(stack)
    for z in range(len(stack)):
        z_slice = stack[z, :, :]
        z_erode = morphology.binary_erosion(z_slice, selem=morphology.disk(disk_size))
        z_smooth = filters.median(z_erode, selem=morphology.disk(disk_size))
        stack1[z, :, :] = z_smooth
    stack2 = np.copy(stack1)
    # searches through stack in y and applies erosion, and then smooths w/
    # median filter
    for y in range(np.shape(stack)[1]):
        y_slice= stack1[:, y, :]
        y_erode = morphology.binary_erosion(y_slice)
        y_smooth = filters.median(y_erode, selem=morphology.disk(disk_size_y))
        stack2[:, y, :] = y_smooth
    # finds bounds of cyst and then for deepest (large z) slices, erodes more
    stack3 = stack2.copy()
    if seq_erosions == True:
        Z = stack2.shape[0]
        LB, UB = cyst_edge(stack2)
        upper_range = int((UB-LB)/2)
        mid_cyst = LB + upper_range
        sel_range = [16, 12, 8, 4]
        for i in range(mid_cyst, Z):
            z_slice = stack2[i, :, :]
            if i >=UB+2:
                z_erode = np.zeros(z_slice.shape)
            elif i < UB+2 and i >= UB - int(upper_range/4):
                disk_e = morphology.disk(sel_range[3])
                z_erode = morphology.binary_erosion(z_slice, selem=disk_e)
            elif i < UB - int(upper_range/4) and i >= UB - int(upper_range/2):
                disk_e = morphology.disk(sel_range[2])
                z_erode = morphology.binary_erosion(z_slice, selem=disk_e)
            else:
                z_erode = z_slice
            stack3[i, :, :] = z_erode
        stack3 = np.array(stack3 > 0, dtype=np.uint8)
    else:
        stack3 = np.array(stack3 > 0, dtype=np.uint8)
    return stack3

def lumen_contours(stack, object_size=512, longest=True):
    """
    Function takes a binary image and finds contours of lumen in each z-slice

    Parameters
    ----------
    stack : binary ndarray
        ndarray of cyst with segmented lumen

    object_size : int, default = 512
        minimum size contour
    longest : boolean, default = True
        determines if choose longest_contour after contour finding, or if choose
        second longest (would choose second longest if cyst contour shows up in
        lumen segmentation)

    Returns
    -------
    out_stack: ndarray
        ndarry of with contours on each z-slice

    """
    out_stack = 0*stack
    for i in range(len(stack)):
        out_slice = 0*stack[i,:,:]
        # cv2 3.3 requires that there be three ouputs (the first one is an im)
        # cv2 4+ requires only two outputs...
        contours, hierarchy = cv2.findContours(stack[i,:,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            longest_contour = np.argmax([c.shape[0] for c in contours])
            second_longest = np.argmax([contours[j].shape[0] if j!=longest_contour else 0 for j in range(len(contours))])
            third_longest = np.argmax([contours[j].shape[0] if j not in [longest_contour,second_longest] else 0 for j in range(len(contours))])
            if longest == True:
                contourp = contours[longest_contour][:,0,:].T
            else:
                contourp = contours[second_longest][:,0,:].T # trying to see if longest_contourworks
                # contourp = contours[third_longest][:,0,:].T # trying to see if longest_contourworks
            out_slice[contourp[1,:],contourp[0,:]] = 255.
            out_slice = ndimage.binary_fill_holes(out_slice)
            out_slice = morphology.remove_small_objects(out_slice, min_size=object_size)
    #         if np.sum(out_slice)<0.5*np.prod(out_slice.shape):
            out_stack[i,:,:] = out_slice
    return out_stack


def lumen_contours_multiple(stack, object_size=512):
    """
    Function takes a binary image and finds contours of lumen in each z-slice

    Parameters
    ----------
    stack : binary ndarray
        ndarray of cyst with segmented lumen

    object_size : int, default = 512
        minimum size contour
    # longest : boolean, default = True
    #     determines if choose longest_contour after contour finding, or if choose
    #     second longest (would choose second longest if cyst contour shows up in
    #     lumen segmentation)

    Returns
    -------
    out_stack: ndarray
        ndarry of with contours on each z-slice

    """
    out_stack = 0*stack
    out_stack2 = 0*stack
    out_stack3 = 0*stack
    for i in range(len(stack)):
        out_slice = 0*stack[i,:,:]
        out_slice2 = 0*stack[i,:,:]
        out_slice3 = 0*stack[i,:,:]
        # cv2 3.3 requires that there be three ouputs (the first one is an im)
        contours, hierarchy = cv2.findContours(stack[i,:,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            longest_contour = np.argmax([c.shape[0] for c in contours])
            second_longest = np.argmax([contours[j].shape[0] if j!=longest_contour else 0 for j in range(len(contours))])
            third_longest = np.argmax([contours[j].shape[0] if j not in [longest_contour,second_longest] else 0 for j in range(len(contours))])
            contourp1 = contours[longest_contour][:,0, :].T
            contourp2 = contours[second_longest][:, 0, :].T
            contourp3 = contours[third_longest][:, 0, :].T
            # if longest == True:
            #     contourp = contours[longest_contour][:,0,:].T
            # else:
            #     contourp = contours[second_longest][:,0,:].T # trying to see if longest_contourworks
            #     # contourp = contours[third_longest][:,0,:].T # trying to see if longest_contourworks
            # out_slice[contourp[1,:],contourp[0,:]] = 1.
            out_slice[contourp1[1, :], contourp1[0, :]] = 255.
            out_slice2[contourp2[1, :], contourp2[0, :]] = 255.
            out_slice3[contourp3[1, :], contourp3[0, :]] = 255.
            out_slice = ndimage.binary_fill_holes(out_slice)
            out_slice = morphology.remove_small_objects(out_slice, min_size=object_size)
            out_slice2 = ndimage.binary_fill_holes(out_slice2)
            out_slice2 = morphology.remove_small_objects(out_slice2, min_size=object_size)
            out_slice3 = ndimage.binary_fill_holes(out_slice3)
            out_slice3 = morphology.remove_small_objects(out_slice3, min_size=object_size)
    #         if np.sum(out_slice)<0.5*np.prod(out_slice.shape):
            out_stack[i,:,:] = out_slice
            out_stack2[i, :, :] = out_slice2
            out_stack3[i, :, :] = out_slice3
    return out_stack, out_stack2, out_stack3

def plot_stack_save(im, colormap='gray', pw=pw_desktop,
                    name='python_fig', subtitle=False):
    """
    Plots all the slices in a z-stack in a subplot & saves it.


    Parameters
    ----------
        im : ndarray
            image stack with dimensions [Z, Y, X]

        colormap : default = 'gray'

        pw : default = '/Users/Claudia/Desktop/'
            where to save png

        name : default = 'python_fig'
            what to save png as

        subtitle : default = False
            if True, puts labels on each z-slice

    Returns
    -------
        fig, ax : objects
    """
    Z = im.shape[0]
    nrows = np.int(np.ceil(np.sqrt(Z)))
    ncols = np.int(Z // nrows + 1)
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    for z in range(Z):
        i = z // ncols
        j = z % ncols
        axes[i, j].imshow(im[z, ...], interpolation='nearest', cmap=colormap)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if subtitle == True:
            axes[i, j].set_title('z = '+str(z), fontsize=8)
    # Remove empty plots
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig(pw + name + '.png', bbox_inches='tight')
    return (fig, axes)

def plot_stack_overlay(im_seg, im, pw=pw_desktop, name='python_fig',
                       subtitle=True):
    """
    Plots all the slices in a z-stack and overlays segmentation on in a subplot
    & saves it.


    Parameters
    ----------
        im_seg : ndarray
            image stack of segmentation with dimensions [Z, Y, X]

        im : ndarray
            image stack with dimensions [Z, Y, X]

        colormap : default = 'gray'

        pw : default = '/Users/clauvasq/Desktop/'
            where to save png

        name : default = 'python_fig'
            what to save png as

        subtitle : default = True
            if True, puts labels on each z-slice

    Returns
    -------
        fig, ax : objects
    """
    Z = im.shape[0]
    nrows = np.int(np.ceil(np.sqrt(Z)))
    ncols = np.int(Z // nrows + 1)
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    for z in range(Z):
        i = z // ncols
        j = z % ncols
        axes[i, j].imshow(im[z, ...], interpolation='nearest', cmap='gray')
        axes[i, j].imshow(im_seg[z, ...], interpolation='nearest', alpha=0.4)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if subtitle == True:
            axes[i, j].set_title('z = '+str(z), fontsize=8)
    # Remove empty plots
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig(pw + name + '.png', bbox_inches='tight')
    return fig, ax
# need to figure out how to get animation to work in spyder
def play_stack(stack, cbar=False, pw = pw_desktop, filename='test'):
    fig = plt.figure()

    def animfunc(i):
        im = plt.imshow(stack[i,:,:], animated=True, cmap='Greys_r')
        if cbar and i==0:
            plt.colorbar()
        plt.title(i)

    animator = animation.FuncAnimation(fig, animfunc, frames=len(stack),
                                       interval=150, blit=False,
                                       repeat_delay=5000)
    plt.draw()
    plt.show()
    animator.save(pw + filename + '.mp4')
    return
