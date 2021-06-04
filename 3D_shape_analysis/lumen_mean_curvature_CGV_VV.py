#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:11:52 2018
Functions for LARFP_lumen_analysis_CGV.py

@author: clauvasq modified from vipul's orignal code
vipul updated shape_parameters function

"""
# import packages
import numpy as np
import skimage.io as io
io.use_plugin('tifffile')
#import skimage.filters
from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

from fourier_contour_fit import *
import cv2
#import mayavi.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D # for use in python 3
from matplotlib import cm
from matplotlib import animation
import Polygon

# functions!
def make_z_contours(z_stack):
    """
    Searches through  z-dim of image stack & find contours on XY-slices.

    Parameters
    ----------
    z_stack : ndarray
        ndarray of segmented lumen with dimensions [Z, Y, X]

    Returns
    -------
    z_contour_stack : list
        Each list item is an ndarray with the contour on that z-slice. If no
        segmentation/contour exists, then the item is empty
    """
    z_contour_stack = []
    for i in range(len(z_stack)):
        imgray = np.uint8(z_stack[i,:,:])
        # imgray[170:300,230]=0
        # imgray[225,200:250]=0
        contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            z_contours_slice = []
            for this_contour in contours:
                z_contours_slice +=[np.array([this_contour[j][0] for j in range(len(this_contour))]).T]
            z_contour_stack += [z_contours_slice]
        else:
            z_contour_stack += [np.array([])]
    return z_contour_stack

def make_x_contours(z_stack):
    """
    Searches through  x-dim of image stack & find contours on YZ-slices.

    Parameters
    ----------
    z_stack : ndarray
        ndarray of segmented lumen with dimensions [Z, Y, X]

    Returns
    -------
    x_contour_stack : list
        Each list item is an ndarray with the contour on that x-slice. If no
        segmentation/contour exists, then the item is empty
    """
    x_contour_stack = []
    # search through z_stack in X to get YZ planes
    # x_stack = np.transpose(z_stack, axes=[2,1,0])
    xID = np.squeeze(np.where(z_stack.sum(0).sum(0) > 0))  # IDs of YZ slices w/ intensity
    for i in range(z_stack.shape[2]):
        # if np.isin(i, xID):
        if i in xID:
            imgray = np.asanyarray(np.uint8(z_stack[:, :, i]), order='C')
            contours, hier = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                x_contours_slice = []
                for this_contour in contours:
                    x_contours_slice +=[np.array([this_contour[j][0] for j in range(len(this_contour))]).T]
                x_contour_stack += [x_contours_slice]
            else:
                x_contour_stack += [np.array([])]
        else:
            x_contour_stack += [np.array([])]
    return x_contour_stack

def fit_z_parameters(z_contour_stack):
    """
    Finds fourier fit for each contour in z-slice.

    Note, contour requires at least 31 points to be fit. It iteratively fits 16
    times

    Parameters
    ----------
    z_contour_stack : list
        Each list item is an ndarray with the contour on that z-slice. If no
        segmentation/contour exists, then the item is empty (from function
        make_z_contours)

    Returns
    -------
    z_fit_params : list
        List w/ each element corresponding to a z-slice. Empty if there are no
        contours (or fits). Otherwise has list with fits
    """
    z_fit_params = []
    for i in range(len(z_contour_stack)):
        if len(z_contour_stack[i])>0:
            z_fits_slice = [optimal_fit(z_contour_stack[i][j]) if len(z_contour_stack[i][j][0])>31 else [] for j in range(len(z_contour_stack[i]))]
            print ('z-slice completed: ' +str(i))
            z_fit_params += [z_fits_slice]
        else:
            z_fit_params  += [[]]
    return z_fit_params

def fit_x_parameters(x_contour_stack):
    """
    Finds fourier fit for each contour in x-slice.

    Note, contour requires at least 31 points to be fit. It iteratively fits 16
    times

    Parameters
    ----------
    x_contour_stack : list
        Each list item is an ndarray with the contour on that x-slice. If no
        segmentation/contour exists, then the item is empty (from function
        make_x_contours)

    Returns
    -------
    x_fit_params : list
        List w/ each element corresponding to a x-slice. Empty if there are no
        contours (or fits). Otherwise has list with fits
    """
    x_fit_params = []
    for i in range(len(x_contour_stack)):
    #     print len(x_contour_stack[i])
        if len(x_contour_stack[i])>0:
            x_fits_slice = [optimal_fit(x_contour_stack[i][j]) if len(x_contour_stack[i][j][0])>31 else [] for j in range(len(x_contour_stack[i]))]
            print ('x-slice completed: ' +str(i))
            x_fit_params += [x_fits_slice]
        else:
            x_fit_params += [[]]
    return x_fit_params

def PolygonArea(corners):
    """
    Area of polygon given a list of vertices
    This result is due to a nifty theorem often called the Shoelace Theorem!
    """
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def shape_parameters(zfits, xy_res, z_res):
    """
    Finds surface area and volume of lumen/cyst given z-fits of fourier fits

    Model a lumen as a stack of pancakes, where the height of each pancake
    is the z-resolution, and the shadow of the pancake is given by the
    contour fit of that z-slice.

    Areas and perimeters are computed using the Polygon package.

    Parameters
    ----------
    zfits: fourier fit parameters in z

    xy_res: resolution of of pixels in x and y dimennsion

    z_res: resolution of z-slices

    Returns
    -------
    SA : surface area of input shape in um^2

    Vol : volume of input shape in um^3
    """

    # surface_area = 0
    # volume = 0
    SA = 0
    Vol = 0
    bottom=-1
    top=-1

    #go slice by slice and store polygon fits, areas, and perimeters
    slice_polygon_approximations = []
    slice_areas  = []
    slice_perims = []

    #store areas and polygon approximations
    for i in range(len(zfits)):
        if len(zfits[i])>0:
            slice_perimeter = 0
            slice_Polygon = Polygon.Polygon()
            for j in range(len(zfits[i])):
                params = zfits[i][j]
                if len(params)>0:
                    popt_x = params[0]
                    popt_y = params[1]
                    n_coeffs_x = (len(popt_x)-1)//2
                    n_coeffs_y = (len(popt_y)-1)//2
                    num_vals = params[2]

                    #derivatives
                    n = np.array(range(1,n_coeffs_x+1))
                    popt_xp = np.array([0]+list(n*popt_x[n_coeffs_x+1:])+list(-n*popt_x[1:n_coeffs_x+1]))
                    n = np.array(range(1,n_coeffs_y+1))
                    popt_yp = np.array([0]+list(n*popt_y[n_coeffs_y+1:])+list(-n*popt_y[1:n_coeffs_y+1]))

                    fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.linspace(0,num_vals,num_vals),*popt_x)
                    fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.linspace(0,num_vals,num_vals),*popt_y)
                    polygon_corners = np.array([fit_x,fit_y]).T

                    xp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.linspace(0,num_vals,num_vals),*popt_xp)
                    yp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.linspace(0,num_vals,num_vals),*popt_yp)

                    #slice perimeter and area
                    ds = 2*np.pi/len(xp)*np.sqrt(xp*xp+yp*yp)
                    perimeter = np.sum(ds)

                    slice_Polygon.addContour(polygon_corners)
                    slice_perimeter += perimeter
                else:
                    pass
            slice_areas += [slice_Polygon.area()]
            slice_perims += [slice_perimeter]
            slice_polygon_approximations += [slice_Polygon]
        else:
            slice_areas += [0]
            slice_perims += [0]
            slice_polygon_approximations += [None]

    # print (len(slice_polygon_approximations))
    #put everything together to get the volume and surface area
    for i in range(len(zfits)-1):
        # print (i)
        if len(zfits[i])>0:
            area = slice_areas[i] * xy_res * xy_res
            perim = slice_perims[i] * xy_res
            this_polygon = slice_polygon_approximations[i]
            next_polygon = slice_polygon_approximations[i+1]
            if (slice_areas[i]>0) and (slice_areas[i-1]==0):
                #area of the bottom of the bottom pancake
                bottom_area = area
                #lateral area of the bottom pancake
                lateral_area = perim * z_res
                #area between the bottom pancake and the next pancake
                in_between_area = (this_polygon ^ next_polygon).area()*xy_res*xy_res
                # in_between_area = (slice_perims[i]+slice_perims[i+1])/2. * z_res
                SA += bottom_area + lateral_area + in_between_area

                #volume of the bottom pancake.
                Vol += area * z_res
                # print (i)
            elif (slice_areas[i]>0) and (slice_areas[i+1]==0):
                top_area = area
                lateral_area = perim * z_res
                SA += top_area + lateral_area

                Vol += area * z_res
            elif (slice_areas[i]>0) and (slice_areas[i+1]>0):
                #this pancake is between two pancakes.
                # print(i)

                #lateral area of this pancake
                lateral_area = perim * z_res
                #area between this pancake and the next pancake
                in_between_area = (this_polygon ^ next_polygon).area()*xy_res*xy_res
                # in_between_area = (slice_perims[i]+slice_perims[i+1])/2. * z_res
                # print (in_between_area)
                # print in_between_area, lateral_area
                SA += lateral_area + in_between_area

                Vol += area * z_res
    # print (slice_areas)
    return SA, Vol

# not annotated yet
def shape_parameters_xz(zfits, z_pixel_size):
    # z_pixel_size: ratio of z resolution to x resolution
    surface_area = 0
    volume = 0
    bottom=-1
    top=-1
    areas = []
    perims = []
    for i in range(len(zfits)):
        if len(zfits[i])>0:
            params = zfits[i]
            popt_x = params[0]
            popt_y = params[1]
            n_coeffs_x = (len(popt_x)-1)/2
            n_coeffs_y = (len(popt_y)-1)/2
            num_vals = params[2]

            #derivatives
            n = np.array(range(1,n_coeffs_x+1))
            popt_xp = np.array([0]+list(n*popt_x[n_coeffs_x+1:])+list(-n*popt_x[1:n_coeffs_x+1]))
            n = np.array(range(1,n_coeffs_y+1))
            popt_yp = np.array([0]+list(n*popt_y[n_coeffs_y+1:])+list(-n*popt_y[1:n_coeffs_y+1]))

            fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
            fit_y = z_pixel_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)
            xp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xp)
            yp = z_pixel_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yp)

            #slice perimeter and area
            ds = 2*np.pi/len(xp)*np.sqrt(xp*xp+yp*yp)
            perimeter = np.sum(ds)
            area = PolygonArea(np.array([fit_x,fit_y]).T)

            areas += [area]
            perims += [perimeter]
        else:
            areas += [None]
            perims += [None]
    # print areas
    # plt.plot([perims[i]/2./np.pi/(np.sqrt(areas[i]/np.pi)) for i in range(len(perims)) if perims[i]!=None])
    # plt.show()
    for i in range(len(zfits)):
        if len(zfits[i])>0 and top==-1:
            # print i
            if (bottom!=-1 or len(zfits[i+1])>0):
                # print 'good'
                area = areas[i]
                perim = perims[i]
                if bottom==-1:
                    bottom=i
                    surface_area += area
                    surface_area += (perim + perims[i+1])/2.
                    volume += area
                    # print i
                elif top==-1 and len(zfits[i+1])==0:
                    top = i
                    surface_area += area
                    # print i
                else:
                    surface_area += (perim + perims[i+1])/2.
                    volume += area
    return surface_area,volume

def curvature_z_slice(params, orig_contour):
    """
    Calculates curvature, normal, and tangent vectors of given curve on a given
    z-slice.

    Parameters
    ----------
    params : list
        fourier fit paramters of single contour on a z-slice

    orig_contour : list
        contour of single z-slice

    Returns
    -------
    curvature_at_points : list
        local expression of curvature

    normal_at_points : list
        normal of curvuature

    tangent_at_points : list
        tangent of curvature
    """
    # takes in fourier fit parameters and gets x,y coord
    popt_x = params[0]
    popt_y = params[1]
    n_coeffs_x = int((len(popt_x)-1)/2)
    n_coeffs_y = int((len(popt_y)-1)/2)
    num_vals = 500
    fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
    fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)
    fitpos = list(zip(fit_x,fit_y))  # for Python3
    # fitpos = np.array(zip(fit_x,fit_y))  # for Python2

    #necessary derivatives
    n = np.array(range(1,n_coeffs_x+1))
    popt_xp = np.array([0]+list(n*popt_x[n_coeffs_x+1:])+list(-n*popt_x[1:n_coeffs_x+1]))
    popt_xpp = np.array([0]+list(n*popt_xp[n_coeffs_x+1:])+list(-n*popt_xp[1:n_coeffs_x+1]))
    popt_xppp = np.array([0]+list(n*popt_xpp[n_coeffs_x+1:])+list(-n*popt_xpp[1:n_coeffs_x+1]))

    n = np.array(range(1,n_coeffs_y+1))
    popt_yp = np.array([0]+list(n*popt_y[n_coeffs_y+1:])+list(-n*popt_y[1:n_coeffs_y+1]))
    popt_ypp = np.array([0]+list(n*popt_yp[n_coeffs_y+1:])+list(-n*popt_yp[1:n_coeffs_y+1]))
    popt_yppp = np.array([0]+list(n*popt_ypp[n_coeffs_y+1:])+list(-n*popt_ypp[1:n_coeffs_y+1]))

    xp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xp)
    xpp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xpp)
    xppp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xppp)

    yp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yp)
    ypp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_ypp)
    yppp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yppp)

    #curvature
    curvature = (-xp*ypp+yp*xpp)/np.power(xp*xp+yp*yp,3./2)
    #also compute normal and tangent vecs to this curve
    vel_vec = zip(xp,yp)  # for Python3
    # vel_vec = np.array(zip(xp,yp))  # for Python2
    tan_vec = [i/np.linalg.norm(i) for i in list(vel_vec)]

    acc_vec = list(zip(xpp,ypp))  # for Python3
    # acc_vec = list(np.array(zip(xpp,ypp)))  # for Python2
    curv_vec = [acc_vec[i]-tan_vec[i]*np.dot(tan_vec[i],acc_vec[i]) for i in range(len((acc_vec)))]
    norm_vec = [curv_vec[i]/np.linalg.norm(curv_vec[i]) for i in range(len(curv_vec))]

    curvature_at_points = []
    normal_at_points = []
    tangent_at_points = []
    for i in range(len(orig_contour.T)):
        pt_id = np.argmin([np.linalg.norm(fitpos[j]-orig_contour.T[i,:]) for j in range(len(fitpos))])
        curvature_at_points += [curvature[pt_id]]
        normal_at_points += [norm_vec[pt_id]]
        tangent_at_points += [tan_vec[pt_id]]
    return curvature_at_points, normal_at_points, tangent_at_points

def curvature_x_slice(params, orig_contour_1, z_size):
    """
    Calculates curvature, normal, and tangent vectors of given curve on a given
    x-slice.

    Parameters
    ----------
    params : list
        fourier fit paramters of single contour on a x-slice

    orig_contour : list
        contour of single x-slice

    z_size : float
        ratio of z-slice resolution to xy-resolution
    Returns
    -------
    curvature_at_points : list
        local expression of curvature

    normal_at_points : list
        normal of curvuature

    tangent_at_points : list
        tangent of curvature
    """
    popt_x = params[0]
    popt_y = params[1]
    n_coeffs_x = int((len(popt_x)-1)/2)
    n_coeffs_y = int((len(popt_y)-1)/2)
    num_vals = 500

    fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
    fit_y = z_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)
    orig_contour = orig_contour_1.copy()
    orig_contour[1,:] = z_size*orig_contour[1,:]
    fitpos = list(zip(fit_x,fit_y))  # for Python3
    # fitpos = np.array(zip(fit_x,fit_y))  # for Python2

    #necessary derivatives
    n = np.array(range(1,n_coeffs_x+1))
    popt_xp = np.array([0]+list(n*popt_x[n_coeffs_x+1:])+list(-n*popt_x[1:n_coeffs_x+1]))
    popt_xpp = np.array([0]+list(n*popt_xp[n_coeffs_x+1:])+list(-n*popt_xp[1:n_coeffs_x+1]))
    popt_xppp = np.array([0]+list(n*popt_xpp[n_coeffs_x+1:])+list(-n*popt_xpp[1:n_coeffs_x+1]))

    n = np.array(range(1,n_coeffs_y+1))
    popt_yp = np.array([0]+list(n*popt_y[n_coeffs_y+1:])+list(-n*popt_y[1:n_coeffs_y+1]))
    popt_ypp = np.array([0]+list(n*popt_yp[n_coeffs_y+1:])+list(-n*popt_yp[1:n_coeffs_y+1]))
    popt_yppp = np.array([0]+list(n*popt_ypp[n_coeffs_y+1:])+list(-n*popt_ypp[1:n_coeffs_y+1]))

    xp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xp)
    xpp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xpp)
    xppp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xppp)

    yp = z_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yp)
    ypp = z_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_ypp)
    yppp = z_size*make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yppp)

    #curvature
    curvature = (-xp*ypp+yp*xpp)/np.power(xp*xp+yp*yp,3./2)
    #also compute normal and tangent vecs to this curve
    vel_vec = zip(xp,yp)  # for Python3
    # vel_vec = np.array(zip(xp,yp))  # for Python2
    tan_vec = [i/np.linalg.norm(i) for i in list(vel_vec)]

    acc_vec = list(zip(xpp,ypp)) # for Python3
    # acc_vec = list(np.array(zip(xpp,ypp)))  # for Python2
    curv_vec = [acc_vec[i]-tan_vec[i]*np.dot(tan_vec[i],acc_vec[i]) for i in range(len((acc_vec)))]
    # curvature = [np.linalg.norm(curv_vec[i]) for i in range(len(curv_vec))]
    norm_vec = [curv_vec[i]/np.linalg.norm(curv_vec[i]) for i in range(len(curv_vec))]


    curvature_at_points = []
    normal_at_points = []
    tangent_at_points = []
    for i in range(len(orig_contour.T)):
        pt_id = np.argmin([np.linalg.norm(fitpos[j]-orig_contour.T[i,:]) for j in range(len(fitpos))])
        curvature_at_points += [curvature[pt_id]]
        normal_at_points += [norm_vec[pt_id]]
        tangent_at_points += [tan_vec[pt_id]]
    return curvature_at_points, normal_at_points, tangent_at_points

def z_curvature_stacks(z_fit_params, z_contour_stack):
    """
    Calculates curvature, normal, and tangent vectors of contours on a given
    image stack on z-slice

    Iteratively steps through z_fit_params and z_contour_stack and calls
    curvature_z_slice

    Parameters
    ----------
    z_fit_params : list
        fourier fit paramters for contours on all z-slices

    z_contour_stack : list
        contours of all z-slices

    Returns
    -------
    z_curvature_stack : list
        local expression of curvatures

    z_norm_stack : list
        normals of curvuature

    z_tan_stack : list
        tangents of curvature
    """
    z_curvature_stack = []
    z_norm_stack = []
    z_tan_stack = []
    for i in range(len(z_fit_params)):
        if len(z_fit_params[i])>0:
            z_curvature_slice = []
            z_norm_stack_slice = []
            z_tan_stack_slice = []
            for k in range(len(z_fit_params[i])):
                if len(z_fit_params[i][k])>0:
                    curvature,a,b = curvature_z_slice(z_fit_params[i][k],z_contour_stack[i][k])
                    z_curvature_slice += [curvature]

                    z_norm_stack_elem = []
                    z_tan_stack_elem = []
                    for j in range(len(a)):
                        z_norm_stack_elem += [np.array(list(a[j])+[0])]
                        z_tan_stack_elem += [np.array(list(b[j])+[0])]
                    z_norm_stack_slice += [z_norm_stack_elem]
                    z_tan_stack_slice += [z_tan_stack_elem]
                else:
                    z_curvature_slice += [[]]
                    z_norm_stack_slice += [[]]
                    z_tan_stack_slice += [[]]
            z_curvature_stack += [z_curvature_slice]
            z_norm_stack += [z_norm_stack_slice]
            z_tan_stack += [z_tan_stack_slice]
            # print (i)
        else:
            z_curvature_stack += [[]]
            z_norm_stack += [[]]
            z_tan_stack += [[]]
    return z_curvature_stack, z_norm_stack, z_tan_stack

def x_curvature_stacks(x_fit_params, x_contour_stack, z_height):
    """
    Calculates curvature, normal, and tangent vectors of contours on a given
    image stack on x-slice

    Iteratively steps through x_fit_params and x_contour_stack and calls
    curvature_x_slice

    Parameters
    ----------
    x_fit_params : list
        fourier fit paramters for contours on all z-slices

    x_contour_stack : list
        contours of all z-slices

    z_height : float
        z-resoltuion over xy-resolution

    Returns
    -------
    x_curvature_stack : list
        local expression of curvatures

    x_norm_stack : list
        normals of curvuature

    x_tan_stack : list
        tangents of curvature
    """
    x_curvature_stack = []
    x_norm_stack = []
    x_tan_stack = []
    for i in range(len(x_fit_params)):
        if len(x_fit_params[i])>0:
            x_curvature_slice = []
            x_norm_stack_slice = []
            x_tan_stack_slice = []
            for k in range(len(x_fit_params[i])):
                if len(x_fit_params[i][k])>0:
                    curvature, a,b = curvature_x_slice(x_fit_params[i][k],x_contour_stack[i][k],z_height)
                    x_curvature_slice += [curvature]

                    x_norm_stack_elem = []
                    x_tan_stack_elem = []
                    for j in range(len(a)):
            #             print np.array([0]+list(a[j]))
                        x_norm_stack_elem += [np.array([0]+list(a[j]))]
                        x_tan_stack_elem += [np.array([0]+list(b[j]))]
                    x_norm_stack_slice += [x_norm_stack_elem]
                    x_tan_stack_slice += [x_tan_stack_elem]
                else:
                    x_curvature_slice += [[]]
                    x_norm_stack_slice += [[]]
                    x_tan_stack_slice += [[]]

            x_curvature_stack += [x_curvature_slice]
            x_norm_stack += [x_norm_stack_slice]
            x_tan_stack += [x_tan_stack_slice]
            # print (i)
        else:
            x_curvature_stack += [[]]
            x_norm_stack += [[]]
            x_tan_stack += [[]]
    return x_curvature_stack, x_norm_stack, x_tan_stack

# not annotated yet
def array_from_contour_stack(value_stack,z_contour_stack,arr):
    output_arr = (np.nan*arr).tolist()
    #for every z slice:
    for i in range(len(z_contour_stack)):
        # if there is a contour at that slice:
        if len(z_contour_stack[i])>0:
            #if there is a fit value for this contour
            if len(value_stack[i])>0:
                #for every contour in this slice
                for k in range(len(z_contour_stack[i])):
                    #if there are values for this contour
                    if len(value_stack[i][k])>0:
                        for j in range(len(z_contour_stack[i][k][0])):
                            output_arr[i][z_contour_stack[i][k][0][j]][z_contour_stack[i][k][1][j]] = value_stack[i][k][j]
    return output_arr

# not annotated yet
def array_from_yz_contour_stack(value_stack,x_contour_stack,arr):
    output_arr = (np.nan*arr).tolist()
    # for every x slice:
    for i in range(len(x_contour_stack)):
        # if there is a contour at that x slice
        if len(x_contour_stack[i])>0:
            #if there is a fit value for this contour
            if len(value_stack[i])>0:
                for k in range(len(x_contour_stack[i])):
                    # print i
                    # print len(x_contour_stack[i][k][0]), len(value_stack[i][k])
                    if len(value_stack[i][k])>0:
                        for j in range(len(x_contour_stack[i][k][0])):
                            # print value_stack[i][j]
                            output_arr[x_contour_stack[i][k][1][j]][i][x_contour_stack[i][k][0][j]] = value_stack[i][k][j]
    return output_arr


def crop_array(arr):
    """
    Finds where array has values that are not nan, keeps those and gets rid of
    all boundary nans in curvature array.

    Parameters
    ----------
    arr : [Z, Y, X]

    Returns
    -------
    arr  : [cropZ, cropY, cropX] (w/out nans)
    """
    #get rid of all the boundary nans.
    i1 = 0
    i2 = arr.shape[0]
    for i in range(arr.shape[0]):
        if not np.all(np.isnan(arr[:i,:,:])):
            i1 = i
            break
    for i in range(arr.shape[0],0,-1):
        if not np.all(np.isnan(arr[i:,:,:])):
            i2 = i
            break
    j1 = 0
    j2 = arr.shape[1]
    for i in range(arr.shape[1]):
        if not np.all(np.isnan(arr[:,:i,:])):
            j1 = i
            break
    for i in range(arr.shape[1],0,-1):
        if not np.all(np.isnan(arr[:,i:,:])):
            j2 = i
            break
    k1 = 0
    k2 = arr.shape[2]
    for i in range(arr.shape[2]):
        if not np.all(np.isnan(arr[:,:,:i])):
            k1 = i
            break
    for i in range(arr.shape[2],0,-1):
        if not np.all(np.isnan(arr[:,:,i:])):
            k2 = i
            break
    return arr[i1:i2,j1:j2,k1:k2]


def mean_curvature_array(z_curv_arr, z_norm_arr, z_tan_arr, x_curv_arr, x_norm_arr, x_tan_arr):
    """
    Calculates mean curvature at each point.

    Parameters
    ----------
    z_curv_arr : array of curvatures from each pixel from z contour fit

    z_norm_arr : array of normal vectors from each pixel from z contour fit

    z_tan_arr : array of tangent vectors from each pixel from z contour fit

    x_curv_arr : array of curvatures from each pixel from x contour fit

    x_norm_arr : array of normal vectors from each pixel from x contour fit

    x_tan_arr : array of tanget vectors from each pixel from x contour fit

    Returns
    -------
    mean_curv_arr : array of mean curvatures calculated by mean of normal
                    curvature in x (x_norm_curv) and z (z_norm_curv)

    approx_mean_curv_arr : array of mean curvatures calculated by mean of
                           curvature in x and z (x_curv & z_curv)
    """
    mean_curv_arr = np.nan*z_curv_arr
    approx_mean_curv_arr = mean_curv_arr.copy()
    for i in range(len(z_norm_arr)):
        for j in range(len(z_norm_arr[0])):
            for k in range(len(z_norm_arr[0][0])):
                if not np.isnan(z_curv_arr[i][j][k]):
                    x_tan = x_tan_arr[i][j][k]
                    z_tan = z_tan_arr[i][j][k]

                    x_norm = x_norm_arr[i][j][k]
                    z_norm = z_norm_arr[i][j][k]

                    x_curv = x_curv_arr[i][j][k]
                    z_curv = z_curv_arr[i][j][k]
                    if not np.any([np.any(np.isnan(l)) for l in [x_tan,z_tan,x_norm,z_norm,x_curv,z_curv]]):
                        surf_norm = np.cross(x_tan,z_tan)
                        if np.dot(surf_norm,z_norm)<0:
                            surf_norm=-surf_norm
                        z_norm_curv = np.dot(z_curv*z_norm,surf_norm)
                        x_norm_curv = np.dot(x_curv*x_norm,surf_norm)
                        H = (x_norm_curv + z_norm_curv)/2.
                        mean_curv_arr[i][j][k] = H
                        approx_mean_curv_arr[i][j][k] = (x_curv + z_curv)/2.
    return mean_curv_arr, approx_mean_curv_arr

# not fully annotated
def save_array(arr, name, image=False):
    """
    saves array (mean curvature array) that's cropped, i.e removes nans in
    indicated location as pickle file

    Parameters
    ----------
    arr : array of Z, Y, Z curvatures??
    name : filepath, str
    image : boolean, Default = False

    Returns
    -------
    none, just saves pickle file in correct indicated location
    """
    if image:
        output_arr = np.zeros(tuple(list(arr.shape)),dtype=np.uint16)
        output_arr[np.isnan(arr)] = 0
        siz = arr.shape
        scale_factor = 255./np.nanmax(arr)-np.nanmax(-arr)
        offset = np.nanmax(-arr)
        for i in range(siz[0]):
            # print (i)
            for j in range(siz[1]):
                for k in range(siz[2]):
                    if not np.isnan(arr[i,j,k]):
                        output_arr[i,j,k] = (arr[i,j,k]+offset)*scale_factor
    #                     output_arr[i,j,k,3] = 1.
        io.imsave(name+'_scale='+str(scale_factor)+'_offset='+str(offset)+'.tiff', output_arr)
    fil = open(name + '_mean_curvature.pickle','wb')  # Python3
    # fil = open(name+'.pickle','w+')  # Python2
    pickle.dump(arr, fil)
    fil.close()

def save_full_array(arr, name, image=False):
    """
    saves FULL array (mean curvature array)

    Parameters
    ----------
    arr : array of Z, Y, X curvatures??
    name : filepath, str
    image : boolean, Default = False

    Returns
    -------
    none, just saves pickle file in correct indicated location
    """
    if image:
        output_arr = np.zeros(tuple(list(arr.shape)),dtype=np.uint16)
        output_arr[np.isnan(arr)] = 0
        siz = arr.shape
        scale_factor = 255./np.nanmax(arr)-np.nanmax(-arr)
        offset = np.nanmax(-arr)
        for i in range(siz[0]):
            print (i)
            for j in range(siz[1]):
                for k in range(siz[2]):
                    if not np.isnan(arr[i,j,k]):
                        output_arr[i,j,k] = (arr[i,j,k]+offset)*scale_factor
    #                     output_arr[i,j,k,3] = 1.
        io.imsave(name+'_scale='+str(scale_factor)+'_offset='+str(offset)+'.tiff', output_arr)
    fil = open(name + '_mean_curvature_UNCROPPED.pickle','wb')  # Python3
    # fil = open(name+'.pickle','w+')  # Python2
    pickle.dump(arr, fil)
    fil.close()

def remove_corners(z_stack,threshold_curvature, name):
    '''
    uses threshold curvatures of z-stack contours to exclude corners (~cell edges)
    from data.

    Parameters
    ----------
    z_stack : ndarray
        segmented lumen image stack with dimensions [Z, Y, X]
    threshold_curvature : float
        local curvature beyond which a point is considered to be part of a "corner"
        and is thus excluded from consideration of membrane shape

    Returns
    -------
    corner_mask : numpy array
        an array which masks all points called as "corners"
    '''

    #load z contour parameters
    f = open(name + '_fit_params.pickle','rb') #for python 3
    # f = open(name + '_fit_params_pickled2.pickle','rb')#for python 2
    z_fit_params = pickle.load(f)['z']
    f.close()

    #load z contour stack.
    f_c = open(name + '_contour_stack.pickle', 'rb') #for python 3
    # f_c = open(name + '_contour_stack_pickled2.pickle','rb')#for python 2
    z_contour_stack = pickle.load(f_c)['z']
    f_c.close()

    z_curvature_stack, z_norm_stack, z_tan_stack = z_curvature_stacks(z_fit_params, z_contour_stack)

    corner_mask = 0*z_stack
    #for every z slice:
    for i in range(len(z_stack)):
        print ('slice',i)
        #if there are contours in this slice:
        if len(z_contour_stack[i])>0:
            #for every contour in that slice:
            for k in range(len(z_contour_stack[i])):
                if len(z_contour_stack[i][k])>0:
                    if len(z_contour_stack[i][k][0])>0:
                        for j in range(len(z_curvature_stack[i][k])):
                            if z_curvature_stack[i][k][j]>threshold_curvature:
                                corner_mask[i][z_contour_stack[i][k][1][j]][z_contour_stack[i][k][0][j]] = 1


    return corner_mask

def make_curvature_array(z_stack, z_height, name):
    """
    Traces contours of segmented lumen, fourier fits, computes curvatures,
    normal vectors, and tangent vectors

    Parameters
    ----------
    z_stack : ndarray
        segmented lumen image stack with dimensions [Z, Y, X]

    z_height : float
        z-resolution over xy-resolution

    name : str
        pathway + filename to save outputs

    Returns
    -------
    H_arr : array
        array of mean curvatures
        also saved as pickel file : name + _mean_curvature.pickle

    name + _contour_stack.pickle
        saves contours for each z-slice and for each x-slice

    name + _fit_params.pickle
        saves fourier fit parameters for each z-slice and for each x-slice

    """
    # Trace the contours of the segmented lumen
    z_contour_stack = make_z_contours(z_stack)
    x_contour_stack = make_x_contours(z_stack)
    print ('Made Contours. Fitting smooth curves...')
    f_c = open(name + '_contour_stack.pickle', 'wb')
    pickle.dump({'z':z_contour_stack, 'x':x_contour_stack}, f_c)
    f_c.close()

    # Fit a smooth closed curve to the xy-sections and the yz-sections
    z_fit_params = fit_z_parameters(z_contour_stack)
    x_fit_params = fit_x_parameters(x_contour_stack)
    f = open(name + '_fit_params.pickle','wb') # use for Python3
    # f = open(name+'_fit_params.pickle','w+') # use for Python2
    pickle.dump({'z':z_fit_params, 'x':x_fit_params}, f)
    f.close()
    print ('Fit smooth curves. Computing curvatures...')

    # From these curves, find curvatures, normal vectors, and tangent vectors
    z_curvature_stack, z_norm_stack, z_tan_stack = z_curvature_stacks(z_fit_params, z_contour_stack)
    x_curvature_stack, x_norm_stack, x_tan_stack = x_curvature_stacks(x_fit_params, x_contour_stack, z_height)
    print ('Computed curvatures. Assembling arrays...')

    # Associate a curvature, tangent, and normal vector to each pixel in the boundary of the lumen
    z_curvature_arr = np.array(array_from_contour_stack(z_curvature_stack, z_contour_stack, z_stack))
    x_curvature_arr = np.array(array_from_yz_contour_stack(x_curvature_stack, x_contour_stack, z_stack))

    x_norm_arr = array_from_yz_contour_stack(x_norm_stack, x_contour_stack, z_stack)
    x_tan_arr = array_from_yz_contour_stack(x_tan_stack, x_contour_stack, z_stack)

    z_norm_arr = array_from_contour_stack(z_norm_stack, z_contour_stack,z_stack)
    z_tan_arr = array_from_contour_stack(z_tan_stack, z_contour_stack,z_stack)
    print ('Assembled arrays. Computing mean curvatures...')

    # Use the cross section curvature, normal, and tangent to estimate the mean curvature for each point in the array
    H_arr, not_H_arr = mean_curvature_array(z_curvature_arr, z_norm_arr, z_tan_arr, x_curvature_arr, x_norm_arr, x_tan_arr)


    print ('Computed mean curvatures. Saving array...')
    H_arr_crop = crop_array(H_arr)
    save_array(H_arr_crop, name, image=True)
    save_full_array(H_arr, name)
    # save_array(H_arr, )

    return H_arr

def make_curvature_array_wFits(z_stack, z_height, name, fit_params, all_contours):
    """
    Given contours and fits, computes curvatures, normal vectors, and tangent
    vectors.

    Parameters
    ----------
    z_stack : ndarray
        segmented lumen image stack with dimensions [Z, Y, X]

    z_height : float
        z-resolution over xy-resolution

    name : str
        pathway + filename to save outputs

    all_fit_params : dictionary
        fourier fit paramters in z and x, from saved pickle file

    all_contours : dictionary
        contours in z and x, from saved pickle file

    Returns
    -------
    H_arr : array
        array of mean curvatures
        also saved as pickle file : name + _mean_curvature.pickle
    """
    z_fit_params = fit_params['z']
    x_fit_params = fit_params['x']
    z_contour_stack = all_contours['z']
    x_contour_stack = all_contours['x']

    # From these curves, find curvatures, normal vectors, and tangent vectors
    z_curvature_stack, z_norm_stack, z_tan_stack = z_curvature_stacks(z_fit_params, z_contour_stack)
    x_curvature_stack, x_norm_stack, x_tan_stack = x_curvature_stacks(x_fit_params, x_contour_stack, z_height)
    print ('Computed curvatures. Assembling arrays...')

    # Associate a curvature, tangent, and normal vector to each pixel in the boundary of the lumen
    z_curvature_arr = np.array(array_from_contour_stack(z_curvature_stack, z_contour_stack, z_stack))
    x_curvature_arr = np.array(array_from_yz_contour_stack(x_curvature_stack, x_contour_stack, z_stack))

    x_norm_arr = array_from_yz_contour_stack(x_norm_stack, x_contour_stack, z_stack)
    x_tan_arr = array_from_yz_contour_stack(x_tan_stack, x_contour_stack, z_stack)

    z_norm_arr = array_from_contour_stack(z_norm_stack, z_contour_stack,z_stack)
    z_tan_arr = array_from_contour_stack(z_tan_stack, z_contour_stack,z_stack)
    print ('Assembled arrays. Computing mean curvatures...')

    # Use the cross section curvature, normal, and tangent to estimate the mean curvature for each point in the array
    H_arr, not_H_arr = mean_curvature_array(z_curvature_arr, z_norm_arr, z_tan_arr, x_curvature_arr, x_norm_arr, x_tan_arr)
    print ('Computed mean curvatures. Saving array...')
    H_arr_crop = crop_array(H_arr)
    save_array(H_arr_crop, name, image=True)
    save_full_array(H_arr, name)
    return H_arr

def plot_smoothed_contours_3D(xz_fits, fname, z=True, pw='Users/Claudia/Desktop/'):
    """
    Plots smoothed contours from fourier fit in 3D & saves png where indicated.
    For Python3!

    Uses Axes3D from mpl_toolkits.mplot3d. Colormap = viridis

    Parameters
    ----------
    xz_fits : list
        fourier fits parameters in z OR x

    fname : str
        image stack name

    z : Boolean
        Default - 'True', means using z_parameters (XY contours)
        'False' means using x_parameters (YZ contours)
    pw : str
        Default - 'Users/Claudia/Desktop/'
        Set to wherever want to save png of plot

    Returns
    -------
    fig, ax - 3D plot of smoothed contours
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_colors = sum(1 for e in xz_fits if e)  # counts non-empty params
    # ax.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_colors)])
    ax.set_prop_cycle('color',plt.cm.viridis(np.linspace(0,1,n_colors)))
    for i in range(len(xz_fits)):
        if len(xz_fits[i]) > 0:
            for j in range(len(xz_fits[i])):
                params = xz_fits[i][j]
                if len(params)>0:
                    popt_x = params[0]
                    popt_y = params[1]
                    n_coeffs_x = (len(popt_x)-1)//2
                    n_coeffs_y = (len(popt_y)-1)//2
                    num_vals = params[2]

                    if z == True:
                        fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x, num_vals)(np.array(range(num_vals)), *popt_x)
                        fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y, num_vals)(np.array(range(num_vals)), *popt_y)
                        fit_z = 0*fit_x+i
                        pname = '_XY'
                    else:
                        fit_y = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
                        fit_z = (make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y))
                        fit_x = 0*fit_y+i
                        pname = '_YZ'
                    ax.plot(fit_x, fit_y, fit_z)
    ax.set_title(fname + ': smoothed contours', fontsize=10, fontweight='bold')
    fig.savefig(pw + fname + pname + '_smoothed_contours.png')
    return fig, ax

# def plot_smoothed_contours(parameters, z=True, x=False):
    # """
    # plot_smoothed_contours: plots smoothed contours in 3D plot using mayavi
    # """
#     fig = mlab.figure()
#     if z:
#         for i in range(len(parameters['z'])):
#             if len(parameters['z'][i])>0:
#                 for j in range(len(parameters['z'][i])):
#                     params = parameters['z'][i][j]
#                     if len(params)>0:
#                         popt_x = params[0]
#                         popt_y = params[1]
#                         n_coeffs_x = (len(popt_x)-1)/2
#                         n_coeffs_y = (len(popt_y)-1)/2
#                         num_vals = params[2]
#
#                         fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
#                         fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)
#                         fit_z = 0*fit_x+5*i
#                         mlab.plot3d(fit_x,fit_y,fit_z, figure=fig, tube_radius=0.4)
#     if x:
#         for i in range(int(len(parameters['x']))):
#             if len(parameters['x'][i])>0:
#                 for j in range(len(parameters['x'][i])):
#                     params = parameters['x'][i][j]
#                     if len(params)>0:
#                         popt_x = params[0]
#                         popt_y = params[1]
#                         n_coeffs_x = (len(popt_x)-1)/2
#                         n_coeffs_y = (len(popt_y)-1)/2
#                         num_vals = params[2]
#
#                         fit_y = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
#                         fit_z = 5*(make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y))
#                         fit_x = 0*fit_y+i
#                         mlab.plot3d(fit_x,fit_y,fit_z, figure=fig, tube_radius=0.4)
#     mlab.show()
#

def plot_mean_curv_3D(arr, xy_res, z_res, v, fname, pw='Users/clauvasq/Desktop/', save_png=True):
    """
    Plots curvature values of lumen & saves png where indicaterd. For Python3!

    Uses Axes3D from mpl_toolkits.mplot3d. Colormap = RdBu
    Red = negative curvature
    Blue = positive curvature

    Parameters
    ----------
    arr : list
        curvature values

    xy_res : float
        xy-resolution um per pixels

    z_res : float
        z-resolution um per slice

    v : False or float
        False - don't have volume yet if flot, then it's used to normalize the
        curvatures with lumen volume

    fname : str
        image stack name

    pw : str
        Default - 'Users/Claudia/Desktop/'
        Set to wherever want to save png of plot

    save : boolean, True
        if True, save to indicarted file

    Returns
    -------
    fig, ax - 3D plot of smoothed contours
    """
    z, y, x = np.where(~(np.isnan(arr)))
    xyz = arr[z, y, x]

    # take cuberoot of lumen to normalize curvatures
    if v == False:
        zyx_norm = [mc/xy_res for mc in xyz]
    else:
        v = np.cbrt(v)
        zyx_norm = [mc*v/xy_res for mc in xyz]

    max_c = 2*np.nanmedian(np.abs(zyx_norm))
    min_c = -max_c
    fig = plt.figure()
    ax = Axes3D(fig)
    scat = ax.scatter(x*xy_res, y*xy_res, z*z_res, c = zyx_norm, s=20, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    fig.colorbar(scat, shrink=0.5, aspect=10)
    ax.set_title(fname + ': mean curvature', fontsize=10, fontweight='bold')
    ax.set_xlabel('x-dimension (um)', fontweight='bold')
    ax.set_ylabel('y-dimension (um)', fontweight='bold')
    ax.set_zlabel('z-dimension (um)', fontweight='bold')
    if save_png == True:
        fig.savefig(pw + fname + '_mean_curvature.png')
    else:
        fig.savefig(pw + fname + '_mean_curvature.eps')
    return fig, ax

def plot_mean_curv_3D_mp4(arr, xy_res, z_res, fname, pw='Users/clauvasq/Desktop/'):
    """
    Plots curvature values of lumen & saves png where indicaterd, and saves mp4
    of image rotating For Python3!

    Uses Axes3D from mpl_toolkits.mplot3d. Colormap = RdBu
    Red = negative curvature
    Blue = positive curvature

    Parameters
    ----------
    arr : list
        curvature values

    fname : str
        image stack name

    xy_res : float
        xy-resolution um per pixels

    z_res : float
        z-resolution um per slice

    pw : str
        Default - 'Users/Claudia/Desktop/'
        Set to wherever want to save png of plot

    save : boolean, True
        if True, save to indicarted file

    Returns
    -------
    fig, ax - 3D plot of smoothed contours
    """
    # unpack curvature data
    z, y, x = np.where(~(np.isnan(arr)))
    xyz = arr[z, y, x]
    max_c = 2*np.nanmedian(np.abs(arr))
    min_c = -max_c

    # create figure and 3DAxes
    fig = plt.figure()
    ax = Axes3D(fig)

    # create base frame for animation
    def init():
        scat = ax.scatter(x*xy_res, y*xy_res, z*z_res, c = xyz, s=20, cmap=cm.RdBu, vmin=min_c,
                          vmax=max_c)
        ax.set_title(fname + ': mean curvature', fontsize=10, fontweight='bold')
        fig.colorbar(scat, shrink=0.5, aspect=10)
        return fig,

    # called sequentially
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # call animator
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360,
                                   interval=20, blit=True)
    # save animation as mp4
    anim.save(pw + fname + '_mean_curvature.mp4', fps=30,
              extra_args=['-vcodec', 'libx264'])
    return fig, ax

# def plot_3D_array(arr):
#   """
#   plot_3D_array: plot mean curvatures for each point in each curve and on each
#   slice. Uses mayavi
#   """
#     pts0,pts1,pts2 = np.where(~(np.isnan(arr)))
#     max_c = 2*np.nanmedian(np.abs(arr))
#     min_c = -max_c
#     mlab.figure()
#     mlab.points3d(pts1,pts2,5*pts0,arr[pts0,pts1,pts2],vmin=min_c, vmax=max_c,colormap='RdBu', scale_mode='none',scale_factor=3.)
#     mlab.colorbar()
#     mlab.show()
def plot_mean_curv_3D_both(arr_lumen, arr_cyst, xy_res, z_res, fname='test', pw='Users/clauvasq/Desktop/', save_png=True):
    """
    Plots curvature values of cyst, lumen, and lumen in cyst. Saves png in
    specified pathway.

    Uses Axes3D from mpl_toolkits.mplot3d. Colormap = RdBu
    Red = negative curvature
    Blue = positive curvature

    Parameters
    ----------
    arr_lumen : list
        curvature values of lumen

    arr_cyst : list
        curvature values of cyst

    xy_res : float
        xy-resolution um per pixels

    z_res : float
        z-resolution um per slice

    fname : str
        image stack name

    pw : str
        Default - 'Users/clauvasq/Desktop/'
        Set to wherever want to save png of plot

    save : boolean, True
        if True, save to indicated file

    Returns
    -------
    fig, ax - 3D plot of smoothed contours
    """
    # unpack arrays
    z_cyst, y_cyst, x_cyst = np.where(~(np.isnan(arr_cyst)))
    xyz_cyst = arr_cyst[z_cyst, y_cyst, x_cyst]


    min_z, max_z = min(z_cyst), max(z_cyst)
    min_y, max_y = min(y_cyst), max(y_cyst)
    min_x, max_x = min(x_cyst), max(x_cyst)
    # sets ticks so that cyst and lumen are on plotted with the same scale
    xticks_major = np.arange(min_x*xy_res, max_x*xy_res, step=5)
    yticks_major = np.arange(min_y*xy_res, max_y*xy_res, step=5)
    zticks_major = np.arange(min_z*z_res, max_z*z_res, step=5)
    z_ticks = np.arange(max_z*z_res, step = 5)

    z_lumen, y_lumen, x_lumen = np.where(~(np.isnan(arr_lumen)))
    xyz_lumen = arr_lumen[z_lumen, y_lumen, x_lumen]

    max_c_cl = [2*np.nanmedian(np.abs(arr_cyst)), 2*np.nanmedian(np.abs(arr_lumen))]
    max_c = max(max_c_cl)
    min_c = -max_c

    # plotting!
    fig = plt.figure(figsize=plt.figaspect(1/3))
    # plot cyst
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scat1 = ax.scatter(x_cyst*xy_res, y_cyst*xy_res, z_cyst*z_res, c=xyz_cyst,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])
    ax.set_title('mean curvature of cyst')
    # plot lumen
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    scat2 = ax.scatter(x_lumen*xy_res, y_lumen*xy_res, z_lumen*z_res, c=xyz_lumen,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    ax.set_xlim(min_x*xy_res, max_x*xy_res)
    ax.set_ylim(min_y*xy_res, max_y*xy_res)
    ax.set_zlim(min_z*z_res, max_z*z_res)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])
    ax.set_title('mean curvature of lumen')
    # plot cyst enclosing lumen, eventually rotate
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    scat1 = ax.scatter(x_cyst*xy_res, y_cyst*xy_res, z_cyst*z_res, c=xyz_cyst,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    scat2 = ax.scatter(x_lumen*xy_res, y_lumen*xy_res, z_lumen*z_res, c=xyz_lumen,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    ax.set_xlim(min_x*xy_res, max_x*xy_res)
    ax.set_ylim(min_y*xy_res, max_y*xy_res)
    ax.set_zlim(min_z*z_res, max_z*z_res)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])
    ax.view_init(elev=100)
    fig.colorbar(scat2, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()
    if save_png == True:
        fig.savefig(pw + fname + '_mean_curvature cyst_lumen_both.png')
    return fig, ax



def plot_overlay_3D(arr_0, arr_1, vol, xy_res, z_res, colorbars_same=False, colorbars_on=False, fname='test', pw='Users/clauvasq/Desktop/', save_eps=True):
    """
    Plots curvature values of lumen before and after DMSO/drug treatment,
    individualluy, and then layered. Saves eps in specified pathway.

    Uses Axes3D from mpl_toolkits.mplot3d. Colormap = RdBu
    Red = negative curvature
    Blue = positive curvature

    Parameters
    ----------
    arr_0 : list
        curvature values of lumen, before treatment

    arr_1 : list
        curvature values of lumen, after treatment

    vol : list
        volume of lumen before, and after!
    xy_res : float
        xy-resolution um per pixels

    z_res : float
        z-resolution um per slice

    colorbars_on : boolean, False
        if False, no colorbars of curvatur
        if True, plots colorbars but aspect ratio of plot gets weird
    colorbars_same : float OR boolean (false)
        if False - nothing happens, python chooses how to set min/max
        if float - then that's the max mean curvature and it will use that to
                set the curvatures
    fname : str
        image stack name

    pw : str
        Default - 'Users/clauvasq/Desktop/'
        Set to wherever want to save png of plot

    save : boolean, True
        if True, save to indicated file

    Returns
    -------
    fig, ax - 3D plot of smoothed contours
    """

    # unpack arrays and volumen normalize curvatures
    z_0, y_0, x_0 = np.where(~(np.isnan(arr_0)))
    xyz_0 = arr_0[z_0, y_0, x_0]
    v0 = np.cbrt(vol[0])
    xyz_0_norm = [mc*v0/xy_res for mc in xyz_0]

    z_1, y_1, x_1 = np.where(~(np.isnan(arr_1)))
    xyz_1 = arr_1[z_1, y_1, x_1]
    v1 = np.cbrt(vol[1])
    xyz_1_norm = [mc*v1/xy_res for mc in xyz_1]

    # set ticks so that both lumens are plotted on the same scale
    min_z, max_z = min(min(z_0), min(z_1)), max(max(z_0), max(z_1))
    min_y, max_y = min(min(y_0), min(y_1)), max(max(z_0), max(z_1))
    min_x, max_x = min(min(x_0), min(x_1)), max(max(x_0), max(x_1))

    xticks_major = np.arange(min_x*xy_res, max_x*xy_res, step=5)
    yticks_major = np.arange(min_y*xy_res, max_y*xy_res, step=5)
    zticks_major = np.arange(min_z*z_res, max_z*z_res, step=5)
    z_ticks = np.arange(max_z*z_res, step = 5)

    # set curvatures min/max so that both lumens on same scale
    if colorbars_same == False:
        max_c_0_1 = [2*np.nanmedian(np.abs(xyz_0_norm)), 2*np.nanmedian(np.abs(xyz_1_norm))]
        max_c = max(max_c_0_1)
        min_c = -max_c
    else:
        max_c = colorbars_same
        min_c = -max_c

    #plotting
    fig = plt.figure(figsize=plt.figaspect(1/3))

    # plotting before lumen
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scat1 = ax.scatter(x_0*xy_res, y_0*xy_res, z_0*z_res, c=xyz_0_norm,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])
    ax.set_title('mean curvature of lumen, t0')
    if colorbars_on == True:
        fig.colorbar(scat1, shrink=0.5, aspect=10)

    # plotting after lumen
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    scat2 = ax.scatter(x_1*xy_res, y_1*xy_res, z_1*z_res, c=xyz_1_norm,
                       s=10, cmap=cm.RdBu, vmin=min_c, vmax=max_c)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])
    ax.set_title('mean curvature of lumen, t1')
    if colorbars_on == True:
        fig.colorbar(scat2, shrink=0.5, aspect=10)

    # plotting before/after lumen on top of each other
    # concatenate x, y, z for before and after conditions
    x_all = np.concatenate((x_0, x_1))
    y_all = np.concatenate((y_0, y_1))
    z_all = np.concatenate((z_0, z_1))
    # make list to keep ids of before and after conditions
    temp_color0 = [0 for i in range(len(x_0))]
    temp_color1 = [1 for i in range(len(x_1))]
    xyz_color = np.concatenate((temp_color0, temp_color1))

    # sort list by x-indices
    x_ind = np.argsort(x_all)
    x_sort = x_all[x_ind]
    y_sort = y_all[x_ind]
    z_sort = z_all[x_ind]
    xyz_sort = xyz_color[x_ind]

    # sort by y-indices
    y_ind = np.argsort(y_sort)
    x_sort = x_sort[y_ind]
    y_sort = y_sort[y_ind]
    z_sort = z_sort[y_ind]
    xyz_sort = xyz_sort[y_ind]

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    scat1 = ax.scatter(x_0*xy_res, y_0*xy_res, z_0*z_res, c='#ee83f8',
                       s=10, vmin=min_c, vmax=max_c, alpha=0.3)
    scat2 = ax.scatter(x_1*xy_res, y_1*xy_res, z_1*z_res, c='#7cf35c',
                       s=10, vmin=min_c, vmax=max_c, alpha=0.3)
    # scat3 = ax.scatter(x_all*xy_res, y_all*xy_res, z_all*z_res, c=xyz_color,
    #                     s=20, cmap=cm.tab10, vmin=min_c, vmax=max_c, alpha=0.3)
    ax.set_xticklabels([str(x-min_x*xy_res) for x in xticks_major])
    ax.set_yticklabels([str(y-min_y*xy_res) for y in yticks_major])
    ax.set_zticklabels([str(z-min_z*z_res) for z in zticks_major])

    if colorbars_on == True:
        fig.colorbar(scat1, shrink=0.5, aspect=10)
        fig.colorbar(scat2, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

    if save_eps == True:
        if colorbars_same == False:
            if colorbars_on == True:
                fig.savefig(pw + fname + '_mean_curvature_lumen_t0t1 colorbars.eps')
            else:
                fig.savefig(pw + fname + '_mean_curvature_lumen_t0t1 brown_teal.eps')
        else:
            if colorbars_on == True:
                fig.savefig(pw + fname + '_mean_curvature_lumen_t0t1 cb_same colorbars.eps')
            else:
                fig.savefig(pw + fname + '_mean_curvature_lumen_t0t1 cb_same brown_teal.eps')
    return fig, ax, max_c

# need to fix annotation
def cart2sphere(im, origin, xyz_scaling):
    """
    cart2sphere: converts cartesian to spherical coordinates around specified
    origin using - https://en.wikipedia.org/wiki/Spherical_coordinate_system

    INPUT: image stack to convert
        origin to use as center (tuple in z, y x)
        xyz scaling - ratio of z to xy resolution (um_per_z/um_per_px)
    OUTPUT: matrices of r, theta, and phi
        note - this is in physics notation! theta = inclination & phi = azimuth

    """
    z, y, x = np.nonzero(im)
    z0, y0, x0 = origin
    r = np.sqrt((x-x0)**2 + (y-y0)**2 + ((z-z0)*xyz_scaling)**2)
    theta = np.arccos(xyz_scaling*(z-z0)/r)
    phi = np.arctan2((y-y0), (x-x0))
    return r, theta, phi

# need to fix annotation

def sphere2cart(r, theta, phi, xyz_scaling):
    """
    sphere2cart: converts spherical coordinates to cartesian

    INPUT: r, theta, phi tuples to convert

    OUTPUT: img stack in z, y, x
    """
    z = r*np.cos(theta)/xyz_scaling
    y = r*np.sin(theta)*np.sin(phi)
    x = r*np.sin(theta)*np.cos(phi)
    return z, y, x
