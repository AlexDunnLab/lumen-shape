import numpy as np
import skimage.io as io
io.use_plugin('tifffile')
from skimage.filters import threshold_otsu, threshold_local, rank
#import skimage.filters
from skimage.measure import regionprops, find_contours
#from skimage.feature import peak_local_max
from scipy import ndimage, interpolate
from skimage.morphology import reconstruction, label, disk, binary_opening, binary_dilation, skeletonize, thin, medial_axis, convex_hull_image
#import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
#import scipy.cluster.hierarchy as hier
from skimage import draw
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import curve_fit
from scipy.signal import medfilt

def make_fourier_fit(na, nb, numel):
    def fourier(x, *a):
        ret = 0.*x+a[0]
        for deg in range(1, na+1):
            ret += a[deg] * np.cos((deg)* x*2*np.pi/numel)
        for deg in range(na+1, na+nb+1):
            ret += a[deg] * np.sin((deg-na)* x*2*np.pi/numel)
        return ret
    return fourier
def fit_fourier_param(relative,n_coeffs_x, n_coeffs_y,show_plots=False):

    #nth order fourier series fit
    coeff0x = [0]+[0]*2*n_coeffs_x
    coeff0y = [0]+[0]*2*n_coeffs_y
    num_vals = len(relative[0])
    popt_x,pcov = curve_fit(make_fourier_fit(n_coeffs_x,n_coeffs_x,num_vals),np.array(range(num_vals)),relative[0,:],p0=coeff0x)
    popt_y,pcov = curve_fit(make_fourier_fit(n_coeffs_y,n_coeffs_y,num_vals),np.array(range(num_vals)),relative[1,:],p0=coeff0y)
    fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
    fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)

    if show_plots:
        # further plots

        plt.figure()
        plt.hold(b=True)
        plt.scatter(range(num_vals), relative[0,:])
        p2, = plt.plot(range(num_vals), fit_x,'.-')#,linewidth=5)
        plt.xlabel('t')
        plt.ylabel('x (px)')
        plt.title('Lumen x-coordinate, with Fourier series smoothing')
        plt.show()

        plt.figure()
        plt.scatter(range(num_vals), relative[1,:])
        p2, = plt.plot(range(num_vals), fit_y,'.-')#,linewidth=5)
        plt.xlabel('t')
        plt.ylabel('y (px)')
        plt.title('Lumen y-coordinate, with Fourier series smoothing')
        plt.show()

        #plot curvature
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

        curvature = (-xp*ypp+yp*xpp)/np.power(xp*xp+yp*yp,3./2)

        #curvature changes (dips)

        #compute mean curvature
        #arclength
        ds = 2*np.pi/len(xp)*np.sqrt(xp*xp+yp*yp)

        #integrate curvature
        mean_curvature = np.dot(ds,curvature)/(np.sum(ds))

        deriv_of_curvature = 2*np.pi/len(xp) * ((yppp*xp-xppp*yp)/np.power(xp*xp+yp*yp,3./2) + (-3./2)*(ypp*xp-xpp*yp)*np.power(xp*xp+yp*yp,-5./2)*(2*xp*xpp+2*yp*ypp))

        tortuosity = np.sum(np.power(deriv_of_curvature,2))
        plt.figure()
        plt.plot(np.cumsum(ds),curvature)
        plt.xlabel('s (px)')
        plt.ylabel('${\\kappa}$ (1/px)')
        plt.title('Calculated curvature (mean %.5g), with smoothing'%(mean_curvature))
        plt.show()

        #plot fit contour and smoothed contour together
        plt.figure()
        plt.plot(fit_x,fit_y,label='Smoothed with %d coefficients in x and %d in y'%(n_coeffs_x,n_coeffs_y), linewidth=5)
        plt.plot(relative[0,:],relative[1,:],'.-',label='Original lumen',linewidth=1)
        plt.title('Lumen with Fourier series smoothing: %d %d'%(n_coeffs_x,n_coeffs_y))
        plt.gca().set_aspect('equal')
        # plt.legend(loc=0)
        plt.show()
        area = PolygonArea(relative.T)
        print ('area:',area)
        print ('perimeter:',np.sum(ds))
        print ('tortuosity:',tortuosity)
        print ('circularity:', (np.sum(ds)/(2*np.pi))/(np.sqrt(area/np.pi)))
        # diffs = np.diff(relative,axis=1)
        # perim = sum(map(np.linalg.norm,diffs.T))
        # print 'crude perimeter:',perim
        # print 'curvature of circle with that area:',1/np.sqrt(area/np.pi)
    return popt_x,popt_y,[fit_x,fit_y]
# fit_fourier_param(relative,2,2,show_plots=True);
# bayesian information criterion:
# assume Gaussian noise with variance given by mean squared deviation from fit.
def bic(nx,ny,relative,fit):
    sd = np.sqrt(sum(map(np.linalg.norm,list(fit-relative))))
#     sd=10
#     print sd
    logpdf = lambda x: np.log(norm.pdf(x,scale=sd))
#     print sum(map(logpdf,fit-contour_polar[:,0]))
    return np.log(len(fit))*(2*nx+2+2*ny)-2*sum(map(logpdf,map(np.linalg.norm,list(fit-relative))))
def optimal_fit(relative):
    fitparams_x,fitparams_y,fit = fit_fourier_param(relative,0,0)
    minbic = bic(1,1,relative,fit)
    nparamx = 1
    nparamy = 1
    bestparam = [fitparams_x,fitparams_y]
    bics = []
    for ix in range(16):
        bicx = []
        for iy in range(16):
            fitparams_x,fitparams_y,fit = fit_fourier_param(relative,ix,iy)
            bictest = bic(ix,iy,relative,fit)
            if bictest<minbic:
                minbic = bictest
                nparamx = ix
                nparamy = iy
                bestparam = [fitparams_x,fitparams_y, len(relative[0])]
            bicx+= [bictest]
        bics+=[bicx[:]]
    return bestparam
