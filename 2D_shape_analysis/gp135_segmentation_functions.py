import numpy as np
from scipy.optimize import curve_fit
from skimage.measure import regionprops, find_contours
from skimage.morphology import reconstruction, label
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

def throw_away_small_blobs(im):
    labeled_stack = label(im)
    im2 = im.copy()
    props = regionprops(labeled_stack)
    # Go through each object and throw away any object smaller than a size threshold
    for p in props:
    #    print(p.area)
        if p.area < 20:
            im2[labeled_stack == p.label] = 0
    return im2

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def make_fourier_fit(na, nb, numel):
    def fourier(x, *a):
        ret = 0.*x+a[0]
        for deg in range(1, na+1):
            ret += a[deg] * np.cos((deg)* x*2*np.pi/numel)
        for deg in range(na+1, na+nb+1):
            ret += a[deg] * np.sin((deg-na)* x*2*np.pi/numel)
        return ret
    return fourier

def fit_fourier_param(relative, n_coeffs_x, n_coeffs_y, fn_root, show_plots=False):
    #nth order fourier series fit
    coeff0x = [0]+[0]*2*n_coeffs_x
    coeff0y = [0]+[0]*2*n_coeffs_y
    num_vals = len(relative[0])
    popt_x,pcov = curve_fit(make_fourier_fit(n_coeffs_x,n_coeffs_x,num_vals),np.array(range(num_vals)),relative[0,:],p0=coeff0x)
    popt_y,pcov = curve_fit(make_fourier_fit(n_coeffs_y,n_coeffs_y,num_vals),np.array(range(num_vals)),relative[1,:],p0=coeff0y)
    fit_x = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_x)
    fit_y = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_y)

    #necessary derivatives
    n = np.array(range(1,n_coeffs_x+1))
    popt_xp = np.array([0]+list(n*popt_x[n_coeffs_x+1:])+list(-n*popt_x[1:n_coeffs_x+1]))
    popt_xpp = np.array([0]+list(n*popt_xp[n_coeffs_x+1:])+list(-n*popt_xp[1:n_coeffs_x+1]))

    n = np.array(range(1,n_coeffs_y+1))
    popt_yp = np.array([0]+list(n*popt_y[n_coeffs_y+1:])+list(-n*popt_y[1:n_coeffs_y+1]))
    popt_ypp = np.array([0]+list(n*popt_yp[n_coeffs_y+1:])+list(-n*popt_yp[1:n_coeffs_y+1]))

    xp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xp)
    xpp = make_fourier_fit(n_coeffs_x, n_coeffs_x,num_vals)(np.array(range(num_vals)),*popt_xpp)
    yp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_yp)
    ypp = make_fourier_fit(n_coeffs_y, n_coeffs_y,num_vals)(np.array(range(num_vals)),*popt_ypp)

    curvature = (-xp*ypp+yp*xpp)/np.power(xp*xp+yp*yp,3./2)

    #compute mean curvature
    #arclength
    ds = np.sqrt(xp*xp+yp*yp)

    #integrate curvature
    mean_curvature = np.dot(ds,curvature)/(np.sum(ds))

    if show_plots:
        fig, ax = plt.subplots(nrows=1, ncols=4)
        # further plots
        ax[0].scatter(range(num_vals), relative[0,:])
        ax[0].plot(range(num_vals), fit_x,'.-')#,linewidth=5)
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('x (px)')
        ax[0].set_title('Lumen x-coordinate, with Fourier series smoothing')

        ax[1].scatter(range(num_vals), relative[1,:])
        ax[1].plot(range(num_vals), fit_y,'.-')#,linewidth=5)
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('y (px)')
        ax[1].set_title('Lumen y-coordinate, with Fourier series smoothing')

        #plot curvature
        plt.figure()
        ax[2].plot(np.cumsum(ds)*(2*np.pi/len(xp)),curvature)
        ax[2].set_xlabel('s (px)')
        ax[2].set_ylabel('${\\kappa}$ (1/px)')
        ax[2].set_title('Calculated curvature (mean %.5g), with smoothing'%(mean_curvature))

        #plot fit contour and smoothed contour together
        ax[3].plot(fit_x,fit_y,label='Smoothed with %d coefficients in x and %d in y'%(n_coeffs_x,n_coeffs_y), linewidth=5)
        ax[3].plot(relative[0,:],relative[1,:],'.-',label='Original lumen',linewidth=1)
        ax[3].set_title('Lumen with Fourier series smoothing: %d %d'%(n_coeffs_x,n_coeffs_y))

    plt.savefig(fn_root + '_curvature_plots.png')
    area = PolygonArea(relative.T)
    print ('area:',area)
    smooth_perim = np.sum(ds)*(2*np.pi/len(xp))
    print ('perimeter:',smooth_perim)
    diffs = np.diff(relative,axis=1)
    perim = sum(map(np.linalg.norm,diffs.T))
    print ('crude perimeter:',perim)
    IPQ = 4*np.pi*area/(smooth_perim**2)
    print ('Isoperimetric Quotient:', IPQ)

    # calculate convex hull and solidity
    hull = ConvexHull(relative.T)
    points = relative.T
    xc = points[hull.vertices,0]
    yc = points[hull.vertices, 1]
    points_c =[]
    for i in range(len(xc)):
        points_c += [[xc[i], yc[i]]]
    area_convex = PolygonArea(np.array(points_c))
    print('convex area:', area_convex)
    solidity = area/area_convex
    print('solidity:', solidity)
    diffs_c = np.diff(points_c,axis=1)
    perim_c = sum(map(np.linalg.norm,diffs_c.T))
    print('convex perimeter:', perim_c)

    return area, smooth_perim, IPQ, area_convex, solidity, perim_c, fit_x, fit_y
