import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize as opt
import pickle

# Geometry
def circular_arc_length(v1,v2,r):
    '''
    length of a circular arc anchored at v1 and v2 and with radius of curvature r
    '''
    l = np.linalg.norm(v1-v2)
    return 2*np.abs(r)*np.arcsin(l/2/np.abs(r))
def circular_segment_area(v1,v2,r):
    '''
    area of a circular segment cut from a circle of radius r by a secant at v1-v2
    '''
    l = np.linalg.norm(v1-v2)
    return r**2*np.arcsin(l/2/np.abs(r))-l/2.*np.sqrt(r**2-(l/2)**2)
def arc_center_angles(v1,v2,r):
    '''
    return the center and angles corresponding to the arc through points v1 and v2 with radius r.
    v1 and v2 must be listed ccw across the circle
    '''
    l = np.linalg.norm(v1-v2)
    midpoint = 0.5*(v1+v2)
    slope = -1./((v2[1]-v1[1])/(v2[0]-v1[0]))
    distance2 = r**2-(l/2.)**2
    xdistances = [np.sqrt(distance2/(slope**2+1.)), -np.sqrt(distance2/(slope**2+1.))]
    center_candidates = [midpoint+[xdist, slope*xdist] for xdist in xdistances]
    #we are defining that a positive curvature means that (v2-v1) x (center-v1 is negative)
    center0 = center_candidates[0]
    if r*np.cross(v2-v1, center0-v1)<0:
        center = center0
    else:
        center = center_candidates[1]
    if r>0:
    #we need to traverse v2 -> v1
        r0 = v2-center
        r1 = v1-center
        theta0 = np.arctan2(r0[1],r0[0])
        theta1 = np.arctan2(r1[1],r1[0])
    else:
        r0 = v1-center
        r1 = v2-center
        theta0 = np.arctan2(-r0[1],-r0[0])
        theta1 = np.arctan2(-r1[1],-r1[0])
    
    return center, theta0*180/np.pi, theta1*180/np.pi
def intersecting_arc_area(v1a,vi,ra,v2b,rb,which='both',thetadiffs=False):
    '''
    given two arcs a and b, listed ccw around the lumen, 
    check if they intersect inside the lumen. 
    Compute the area of the lune of intersection
    '''
    if ra<0 or rb<0:
        return 0
    centera,theta1a,theta2a = arc_center_angles(v1a,vi,ra)
    centerb,theta1b,theta2b = arc_center_angles(vi,v2b,rb)
    d_ab = centerb-centera
    d_ai = vi - centera
    d_bi = vi - centerb
    intersection_a = np.cross(d_ai, d_ab)
    #if the arcs intersect
    if intersection_a>0:
        theta_diff_a = 2* np.arcsin(intersection_a/(np.linalg.norm(d_ab)*np.linalg.norm(d_ai)))
        intersection_b = np.cross(-d_ab, d_bi)
        theta_diff_b = 2* np.arcsin(intersection_b/(np.linalg.norm(d_ab)*np.linalg.norm(d_bi)))
        segment_a = ra**2*(theta_diff_a/2) - ra**2*np.sin(theta_diff_a/2)*np.cos(theta_diff_a/2)
        segment_b = rb**2*(theta_diff_b/2) - rb**2*np.sin(theta_diff_b/2)*np.cos(theta_diff_b/2)
        if which=='both':
            return segment_a+segment_b
        elif which=='a':
            if thetadiffs:
                return theta_diff_a
            return segment_a
        elif which=='b':
            if thetadiffs:
                return theta_diff_b
            return segment_b
    else:
        return 0
def intersecting_arc_len(v1a,vi,ra,v2b,rb,which='both',thetadiffs=False):
    '''
    given two arcs a and b, listed ccw around the lumen, 
    check if they intersect inside the lumen. 
    Compute the area of the lune of intersection
    '''
    if ra<0 or rb<0:
        return 0,0
    centera,theta1a,theta2a = arc_center_angles(v1a,vi,ra)
    centerb,theta1b,theta2b = arc_center_angles(vi,v2b,rb)
    d_ab = centerb-centera
    d_ai = vi - centera
    d_bi = vi - centerb
    intersection_a = np.cross(d_ai, d_ab)
    #if the arcs intersect
    if intersection_a>0:
        theta_diff_a = 2* np.arcsin(intersection_a/(np.linalg.norm(d_ab)*np.linalg.norm(d_ai)))
        intersection_b = np.cross(-d_ab, d_bi)
        theta_diff_b = 2* np.arcsin(intersection_b/(np.linalg.norm(d_ab)*np.linalg.norm(d_bi)))
        segment_a = theta_diff_a*ra
        segment_b = theta_diff_b*rb
        if which=='both':
            return segment_a+segment_b
        elif which=='a':
            if thetadiffs:
                return theta_diff_a
            return segment_a, 2*ra*np.sin(theta_diff_a/2.)
        elif which=='b':
            if thetadiffs:
                return theta_diff_b
            return segment_b, 2*rb*np.sin(theta_diff_b/2.)
    else:
        return 0,0
def polygon_area(list_of_points):
    '''
    Shoelace theorem for area of a polygon
    '''
    corners = np.array(list_of_points)
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = area / 2.0
    return area
def polar_to_cartesian(r,theta):
    '''
    convert polar to cartesian
    '''
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    xy = np.array([x,y])
    return xy.T
def cartesian_to_polar(xy):
    '''
    convert polar to cartesian
    '''
    xy = np.array(xy)
    r = np.linalg.norm(xy,axis=1)
    theta = np.arctan2(xy[:,1],xy[:,0])
    
    return r,theta


def simulated_annealing(func, x0, T=1.,stepsize=0.01, niter=10, maxiter=10000,plot=False):
    '''
    Minimizing a function by simulated annealing--linear annealing schedule
    '''

    best_val = func(x0)
    print(best_val)
    x_best = x0
    Ts = np.linspace(T,0,maxiter)
    vals = []
    iters = 0
    for i in range(maxiter):
        T = Ts[i]
        x_try = x_best+np.random.randn(*x_best.shape)*(stepsize)
        try_val = func(x_try)
        if try_val<best_val:
            x_best = x_try
            vals+=[try_val]
            best_val=try_val
            iters+=1
            if iters==niter:
                break
        else:
            check = np.random.rand()
            if check < np.exp(-(try_val-best_val)/T):
                x_best = x_try
                best_val = try_val
                vals+=[try_val]
    if plot:
        plt.plot(vals)
    return {'x':x_best}

def grow_cyst(params,stretched=False,l_l=False):
    '''
    Simulate a growing cyst, from 3 cells up to 10 cells.
    Uses stochastic gradient descent to minimize the energy
    before dividing one of the cells in two and continuing.
    '''
    A0=np.nan
    if stretched:
        dirname = './growing_stretchedinducedmitosis_lvms_k=%.1f_p=%.2f_la=%.2f/'%(params['k'],params['P_lumen'],params['l_a'])
        if l_l:
            dirname = './lateral_expansion_growing_stretchedinducedmitosis_lvms_k=%.1f_p=%.2f_la=%.2f/'%(params['k'],params['P_lumen'],params['l_a'])
    else:
        dirname = './growing_lvms_k=%.1f_p=%.2f_la=%.2f/'%(params['k'],params['P_lumen'],params['l_a'])
    if not(os.path.isdir(dirname)):
        os.mkdir(dirname)
    while np.isnan(A0):
        lvm = LumenVertexModel(3,**params)
        lvm.p = 0.
        A0 = lvm.A0
        if np.isinf(lvm.pressure_tension_energy()):
            A0 = np.nan
    for n_cells in [3,4,5,6,7,8,9,10]:
        lvm.minimize_energy(lvm.pressure_tension_energy,niter=800,maxiter=20000,stepsize=0.005,T=1e-32,plot=False)
        lvm.p = params['P_lumen']
        serialno = str(np.random.rand()).split('.')[-1]
        if not l_l:
            lvm.save(dirname+'%s.pick'%(serialno))
        else:
            lvm.draw_model()
        if stretched:
            lvm.divide_cell(random=False,stretched=True)
        else:
            lvm.divide_cell(random=True)

class LumenVertexModel:
    '''
    class for a 2d cross-sectional kind of vertex model for a lumen
    '''
    def __init__(self,n_cells,jitter=0., P_cell=1., P_lumen=0.5, cell_thickness=1.,k=1., l_a=0.8, l_l=1.,p0=4.):
        # initialized with every vertex equally spaced along the unit circle. Can jitter this later.
        self.apical_vertices = [n_cells/5.*(jitter*(np.random.rand())*np.array([np.cos(2*np.pi*i/n_cells),np.sin(2*np.pi*i/n_cells)])) for i in range(n_cells)]
        
        centroid = np.mean(self.apical_vertices, axis=0)
        self.basal_vertices = [centroid + (cell_thickness+1)*((v-centroid)/np.linalg.norm(v-centroid)) for v in self.apical_vertices]       
        self.apical_radii = [2*np.linalg.norm(self.apical_vertices[i-1]) for i in range(n_cells)]
        self.basal_radii = [1+cell_thickness for i in range(n_cells)]
        
        self.n_cells = n_cells

        self.P_cell = P_cell
        self.P_lumen = P_lumen
        
        self.l_a = l_a
        self.l_b = 0.
        self.l_l = l_l
        self.p0=p0
        
        
        self.k = k
        self.A0 = np.mean([self.cell_area(i) for i in range(n_cells)])
        
    def divide_cell(self, random=True, stretched=False):
        '''
        Divide cells
        '''
        if random:
            which_cell = np.random.choice(range(1,self.n_cells))
            print(which_cell)
        elif stretched:
            cell_area_terms = np.array([ self.cell_area(i) for i in range(self.n_cells)])
            if any(cell_area_terms < 0) or any(np.isnan(cell_area_terms)):
                return np.inf
            self.A0 = np.mean(cell_area_terms)
            
            lumen_pressure_term = -self.P_lumen * self.lumen_area()/self.A0
            if self.lumen_area()<0:
                return np.inf
            apical_length_terms = np.array([self.k * (self.apical_len(i)/np.sqrt(self.A0) - self.l_a)**2 for i in range(self.n_cells)])
            basal_length_terms = np.array([self.k * (self.basal_len(i)/np.sqrt(self.A0) - self.l_b)**2 for i in range(self.n_cells)])
            
            lateral_length_terms = np.array([self.k * (self.lateral_len(i)/np.sqrt(self.A0) - self.l_l)**2 for i in range(self.n_cells)])
            lateral_length_terms = lateral_length_terms + np.roll(lateral_length_terms,-1)
            stretching_terms = apical_length_terms + basal_length_terms + lateral_length_terms
            which_cell = np.argmax(stretching_terms)

        new_apical_v = 0.5*(self.apical_vertices[which_cell-1]+self.apical_vertices[which_cell])
        new_basal_v = 0.5*(self.basal_vertices[which_cell-1]+self.basal_vertices[which_cell])

        self.apical_vertices = np.concatenate((self.apical_vertices[:which_cell],[new_apical_v],self.apical_vertices[which_cell:]))
        self.basal_vertices = np.concatenate((self.basal_vertices[:which_cell],[new_basal_v],self.basal_vertices[which_cell:]))
        self.apical_radii = np.concatenate((self.apical_radii[:which_cell],[self.apical_radii[which_cell]],self.apical_radii[which_cell:]))
        self.basal_radii = np.concatenate((self.basal_radii[:which_cell],[self.basal_radii[which_cell]],self.basal_radii[which_cell:]))

        self.n_cells+=1

    def plot_cell_area(self,idx):
        '''
        area of cell idx
        '''
        cell_apical_vertices = [self.apical_vertices[idx-1],self.apical_vertices[idx]]
        cell_basal_vertices = [self.basal_vertices[idx-1],self.basal_vertices[idx]]
        vertices = np.array(cell_apical_vertices[::-1]+cell_basal_vertices)
        plt.plot(vertices[:,0],vertices[:,1])
    
    def plot_cells(self):
        for i in range(self.n_cells):
            self.plot_cell_area(i)

    def cell_overlap_area(self,idx,direction,thetadiffs=False):
        '''
        account for overlapping areas. 
        direction=-1 means overlap with the cell before
        '''
        
        if direction==-1:
            v1a = self.apical_vertices[idx-2]
            vi = self.apical_vertices[idx-1]
            v2b = self.apical_vertices[idx]
            ra = self.apical_radii[idx-1]
            rb = self.apical_radii[idx]
            return intersecting_arc_area(v1a,vi,ra,v2b,rb,which='b',thetadiffs=thetadiffs)
        elif direction==1:
            v1a = self.apical_vertices[idx-1]
            vi = self.apical_vertices[idx]
            v2b = self.apical_vertices[(idx+1)%self.n_cells]
            ra = self.apical_radii[idx]
            rb = self.apical_radii[(idx+1)%self.n_cells]
            return intersecting_arc_area(v1a,vi,ra,v2b,rb,which='a',thetadiffs=thetadiffs)
    def cell_overlap_len(self,idx,direction,thetadiffs=False):
        '''
        account for overlapping areas. 
        direction=-1 means overlap with the cell before
        '''
        
        if direction==-1:
            v1a = self.apical_vertices[idx-2]
            vi = self.apical_vertices[idx-1]
            v2b = self.apical_vertices[idx]
            ra = self.apical_radii[idx-1]
            rb = self.apical_radii[idx]
            return intersecting_arc_len(v1a,vi,ra,v2b,rb,which='b',thetadiffs=thetadiffs)
        elif direction==1:
            v1a = self.apical_vertices[idx-1]
            vi = self.apical_vertices[idx]
            v2b = self.apical_vertices[(idx+1)%self.n_cells]
            ra = self.apical_radii[idx]
            rb = self.apical_radii[(idx+1)%self.n_cells]
            return intersecting_arc_len(v1a,vi,ra,v2b,rb,which='a',thetadiffs=thetadiffs)


    def cell_area(self,idx):
        '''
        area of cell idx
        '''
        cell_apical_vertices = [self.apical_vertices[idx-1],self.apical_vertices[idx]]
        cell_basal_vertices = [self.basal_vertices[idx-1],self.basal_vertices[idx]]
        
        vertices_in_one_direction = cell_apical_vertices[::-1]+cell_basal_vertices
        cell_polygon_area = polygon_area(vertices_in_one_direction)
        
        apical_sector_area = circular_segment_area(*cell_apical_vertices, np.abs(self.apical_radii[idx]))*np.sign(self.apical_radii[idx])
        basal_sector_area = circular_segment_area(*cell_basal_vertices, np.abs(self.basal_radii[idx]))*np.sign(self.basal_radii[idx])
        
        #areas of overlap with the neighboring cells.
        left_overlap = self.cell_overlap_area(idx,-1)
        right_overlap = self.cell_overlap_area(idx,1) 

        return cell_polygon_area + apical_sector_area + basal_sector_area - left_overlap - right_overlap
    
    def apical_len(self,idx):
        '''
        length of apical side of cell idx
        '''
        v1,v2 = [self.apical_vertices[idx-1],self.apical_vertices[idx]]
        r = np.abs(self.apical_radii[idx])

        #lengths of overlap with the neighboring cells.
        left_minus,left_plus = self.cell_overlap_len(idx,-1)
        right_minus, right_plus = self.cell_overlap_len(idx,1) 
        return circular_arc_length(v1,v2,r) - left_minus - right_minus + left_plus + right_plus
    def basal_len(self,idx):
        '''
        length of basal side of cell idx
        '''
        v1,v2 = [self.basal_vertices[idx-1],self.basal_vertices[idx]]
        r = self.basal_radii[idx]
        return circular_arc_length(v1,v2,r)
    def lateral_len(self,idx):
        '''
        length of lateral side of cell idx
        '''
        v1,v2 = self.apical_vertices[idx],self.basal_vertices[idx]
        return np.linalg.norm(v1-v2)
    
    def lumen_area(self,verbose=False):
        '''
        area of lumen
        '''
        poly_area = np.abs(polygon_area(self.apical_vertices))
        lumen_area = poly_area
        for idx in range(self.n_cells):
            cell_apical_vertices = [self.apical_vertices[idx-1],self.apical_vertices[idx]]
            lumen_area -= circular_segment_area(*cell_apical_vertices, np.abs(self.apical_radii[idx]))*np.sign(self.apical_radii[idx])
            if self.apical_radii[idx-1]>0 and self.apical_radii[idx]>0:
                intersection = intersecting_arc_area(self.apical_vertices[idx-2],
                    self.apical_vertices[idx-1],
                    self.apical_radii[idx-1],
                    self.apical_vertices[idx],
                    self.apical_radii[idx])
                lumen_area += intersection
                if verbose:
                    print('intersection here! Area:', intersection)
        if lumen_area/poly_area < 0.2:
            return -1

        return lumen_area
    def cyst_area(self):
        '''
        Area of the whole cyst
        '''
        poly_area = np.abs(polygon_area(self.basal_vertices))
        cyst_area = poly_area
        for idx in range(self.n_cells):
            cell_basal_vertices = [self.basal_vertices[idx-1],self.basal_vertices[idx]]
            cyst_area += circular_segment_area(*cell_basal_vertices, np.abs(self.basal_radii[idx]))*np.sign(self.basal_radii[idx])
        return cyst_area

    def lumen_area_correction(self,resolution=0.01):
        corner_theta1s =[]
        corner_theta0s =[]
        for i in range(self.n_cells):
            
            #draw the apical faces:
            left_overlap = self.cell_overlap_area(i,-1)
            right_overlap = self.cell_overlap_area(i,1)
            r = self.apical_radii[i]
            c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
            if left_overlap>0:
                theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                t1 = (t1-theta_diff_a*180/np.pi)%(360)
            if right_overlap>0:
                theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                t0 = (t0+theta_diff_b*180/np.pi)%(360)
            corner_theta1s +=[(t1+90)%180]
            corner_theta0s +=[(t0+90)%180]
        corner_thetas = np.array([(corner_theta1s[i-1]-corner_theta0s[i])%180 for i in range(len(corner_theta1s))])
        correction = resolution**2*self.A0/4*np.sum(1./np.tan(corner_thetas*np.pi/180/2))
        return correction
    def lumen_perim_correction(self,resolution=0.01):
        corner_theta1s =[]
        corner_theta0s =[]
        for i in range(self.n_cells):
            
            #draw the apical faces:
            left_overlap = self.cell_overlap_area(i,-1)
            right_overlap = self.cell_overlap_area(i,1)
            r = self.apical_radii[i]
            c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
            if left_overlap>0:
                theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                t1 = (t1-theta_diff_a*180/np.pi)%(360)
            if right_overlap>0:
                theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                t0 = (t0+theta_diff_b*180/np.pi)%(360)
            corner_theta1s +=[(t1+90)%180]
            corner_theta0s +=[(t0+90)%180]
        corner_thetas = np.array([(corner_theta1s[i-1]-corner_theta0s[i])%180 for i in range(len(corner_theta1s))])
        correction = resolution*np.sqrt(self.A0)*np.sum(1./np.sin(corner_thetas*np.pi/180/2) - 1)
        return correction

    def lumen_perim(self, openn=False):
        '''
        perimeter of lumen
        '''
        if openn:
            perim = 0
            for i in range(self.n_cells):
                
                #draw the apical faces:
                left_overlap = self.cell_overlap_area(i,-1)
                right_overlap = self.cell_overlap_area(i,1)
                r = self.apical_radii[i]
                c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
                if left_overlap>0:
                    theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                    t1 = (t1-theta_diff_a*180/np.pi)%(360)
                if right_overlap>0:
                    theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                    t0 = (t0+theta_diff_b*180/np.pi)%(360)
                if np.abs(t1-t0)%360>180:
                    perim+= r*(360-np.abs(t1-t0)%360)*np.pi/180
                else:
                    perim += r*((np.abs(t1-t0)%360)*np.pi/180)
        else:
            perim= np.sum([self.apical_len(i) for i in range(self.n_cells)])
        return perim
    
    def lumen_IPQ(self):
        '''
        isoperimetric quotient = 4piA/P^2
        '''
        return 4*np.pi*self.lumen_area()/(self.lumen_perim()**2)
    
    def lumen_polygonal_IPQ(self):
        '''
        quotient (A/P^2)/(A/P^2 for the regular polygon)
        i.e., how close is the lumen to filling out the regular polygon?
        '''
        AP2_lumen = self.lumen_area()/(self.lumen_perim()**2)
        n = self.n_cells
        regular_A = n/(4*np.tan(np.pi/n))
        regular_P2 = n**2
        AP2_regular = regular_A/regular_P2
        return AP2_lumen/AP2_regular
    
    def lumen_open_IPQ(self,resolution_correct=False,resolution=0.01):
        '''
        IPQ for only the open part of the lumen
        '''
        if resolution_correct:
            return 4*np.pi*(self.lumen_area()-self.lumen_area_correction(resolution=resolution))/((self.lumen_perim(openn=True)-self.lumen_perim_correction(resolution=resolution))**2)
        else:
            return 4*np.pi*(self.lumen_area())/(self.lumen_perim(openn=True)**2)
    def lumen_open_polygonal_IPQ(self):
        '''
        polygonal IPQ, but only for the open part of the lumen
        '''
        perim = 0
        for i in range(self.n_cells):
            
            #draw the apical faces:
            left_overlap = self.cell_overlap_area(i,-1)
            right_overlap = self.cell_overlap_area(i,1)
            r = self.apical_radii[i]
            c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
            if left_overlap>0:
                theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                t1 = t1-theta_diff_a*180/np.pi
            if right_overlap>0:
                theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                t0 = t0+theta_diff_b*180/np.pi
            perim+= r*((t1-t0)*np.pi/180%(np.pi*2))
        AP2_lumen = self.lumen_area()/(perim**2)
        n = self.n_cells
        regular_A = n/(4*np.tan(np.pi/n))
        regular_P2 = n**2
        AP2_regular = regular_A/regular_P2
        return AP2_lumen/AP2_regular
    
    def cyst_IPQ(self):
        '''
        isoperimetric quotient = 4piA/P^2
        '''
        P = sum([self.basal_len(i) for i in range(self.n_cells)])
        A = self.lumen_area() + sum([self.cell_area(i) for i in range(self.n_cells)])
        return 4*np.pi*A/P/P
        
    def pressure_tension_energy(self):
        '''
        Energy function used in Vasquez, Vachharajani, et al.
        '''
        cell_area_terms = np.array([ self.cell_area(i) for i in range(self.n_cells)])
        if any(cell_area_terms < 0) or any(np.isnan(cell_area_terms)):
            return np.inf
        self.A0 = np.mean(cell_area_terms)
        cell_area_terms = np.sum((cell_area_terms/self.A0-1)**2)
        lumen_pressure_term = -self.P_lumen * self.lumen_area()/self.A0
        if self.lumen_area()<0:
            return np.inf
        apical_length_terms = np.sum([self.k * (self.apical_len(i)/np.sqrt(self.A0) - self.l_a)**2 for i in range(self.n_cells)])
        basal_length_terms = np.sum([self.k * (self.basal_len(i)/np.sqrt(self.A0) - self.l_b)**2 for i in range(self.n_cells)])
        lateral_length_terms = np.sum([self.k * (self.lateral_len(i)/np.sqrt(self.A0) - self.l_l)**2 for i in range(self.n_cells)])
        return cell_area_terms + lumen_pressure_term + apical_length_terms + basal_length_terms + lateral_length_terms


    def pressure_tension_energy_basolateral(self):
        '''
        Alternative energy function, where basolateral domain is regulated all as one.
        '''
        cell_area_terms = np.array([ self.cell_area(i) for i in range(self.n_cells)])
        if any(cell_area_terms < 0) or any(np.isnan(cell_area_terms)):
            return np.inf
        self.A0 = np.mean(cell_area_terms)
        cell_area_terms = np.sum((cell_area_terms/self.A0-1)**2)
        lumen_pressure_term = -self.P_lumen * self.lumen_area()/self.A0
        if self.lumen_area()<0:
            return np.inf
        apical_length_terms = np.sum([self.k * (self.apical_len(i)/np.sqrt(self.A0) - self.l_a)**2 for i in range(self.n_cells)])
        basolateral_length_terms = np.sum([self.k * ((self.basal_len(i)+self.lateral_len(i)+self.lateral_len(i-1))/np.sqrt(self.A0) - self.l_b)**2 for i in range(self.n_cells)])
        return cell_area_terms + lumen_pressure_term + apical_length_terms + basolateral_length_terms

    def manning_VM_energy(self):
        '''
        Energy function as used in Bi et al. Nat Phys 2015.
        '''
        cell_area_terms = np.array([ self.cell_area(i) for i in range(self.n_cells)])
        if any(cell_area_terms < 0):
            return np.inf
        cell_area_terms = np.sum((cell_area_terms/self.A0-1)**2)
        if self.lumen_area()<0:
            return np.inf
        cell_perim_terms = np.sum([self.k * ((self.lateral_len(i)+self.lateral_len(i-1)+self.basal_len(i)+self.apical_len(i))/np.sqrt(self.A0) - self.p0)**2 for i in range(self.n_cells)])
        
        return cell_area_terms + cell_perim_terms
 
    def minimize_energy(self,energyfun, **kwarg):
        n_cells = self.n_cells
        #an objective function that specifies each vertex with its POLAR position
        #x is formatted as follows:
        # - first all the apical thetas
        # - then all the apical rs
        # - then all the apical radii
        # - then all the basal thetas
        # - then all the basal rs
        # - then all the basal radii
        
        def decode_and_set_input(x):
            apical_thetas = np.mod(np.array(x[:n_cells]),2*np.pi)
            apical_rs = np.array(x[n_cells:2*n_cells])
            apical_radii = 1./np.array(x[2*n_cells:3*n_cells])
            
            basal_thetas = np.mod(np.array(x[3*n_cells:4*n_cells]),2*np.pi)
            basal_rs = np.array(x[4*n_cells:5*n_cells])
            basal_radii = 1./np.array(x[5*n_cells:6*n_cells])
            
            apical_indices = np.argsort(apical_thetas)
            basal_indices = np.argsort(basal_thetas)
            
            apical_thetas = apical_thetas[apical_indices]
            apical_rs = apical_rs[apical_indices]
            self.apical_vertices = polar_to_cartesian(apical_rs, apical_thetas)
            
            basal_thetas = basal_thetas[basal_indices]
            basal_rs = basal_rs[basal_indices]
            self.basal_vertices = polar_to_cartesian(basal_rs, basal_thetas)
            
            self.apical_radii = apical_radii[apical_indices]
            self.basal_radii = basal_radii[basal_indices]
            
        def objective_function(x):
            decode_and_set_input(x)
            return energyfun()
        
        apical_rs, apical_thetas = cartesian_to_polar(self.apical_vertices)
        basal_rs, basal_thetas = cartesian_to_polar(self.basal_vertices)
        
        x0 = np.concatenate((apical_thetas,apical_rs, 1./np.array(self.apical_radii), basal_thetas, basal_rs, 1./np.array(self.basal_radii)))
        
        res = simulated_annealing(objective_function, x0,**kwarg)
        xf = res['x']
        decode_and_set_input(xf)
        
    def draw_model(self, ax=None, highlightlumen=False):
        
        if ax is None:
            plt.figure()
            ax = plt.gca()
        min_x, max_x = 0,0
        min_y, max_y = 0,0
        for i in range(self.n_cells):
            
            #draw the apical faces:
            left_overlap = self.cell_overlap_area(i,-1)
            right_overlap = self.cell_overlap_area(i,1)
            r = self.apical_radii[i]
            c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
            if left_overlap>0:
                theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                t1 = (t1-theta_diff_a*180/np.pi)%360

                x,y = c[0] + r*np.cos(np.pi/180*t1), c[1]+r*np.sin(np.pi/180*t1)
                x0,y0 = tuple(self.apical_vertices[i-1])
                ax.plot([x0,x],[y0,y],c='green')
            if right_overlap>0:
                theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                t0 = (t0+theta_diff_b*180/np.pi)%360
                x,y = c[0] + r*np.cos(t0*np.pi/180), c[1] + r*np.sin(t0*np.pi/180)
                x0,y0 = tuple(self.apical_vertices[i])
                ax.plot([x0,x],[y0,y],c='green')
            if highlightlumen:
                arc = patches.Arc(c,2*r,2*r,theta1=t0,theta2=t1,ec='blue')
            else:
                arc = patches.Arc(c,2*r,2*r,theta1=t0,theta2=t1,ec='green')
            
            ax.add_patch(arc)
            
            #draw the basal faces:
            r = -self.basal_radii[i]
            c,t0,t1 = arc_center_angles(self.basal_vertices[i-1],self.basal_vertices[i],r)
            arc = patches.Arc(c,2*r,2*r,theta1=t0,theta2=t1,ec='red')
            ax.add_patch(arc)
            
            #draw the lateral faces(flat):
            av, bv = self.apical_vertices[i],self.basal_vertices[i]
            lateral_face = np.array([av,bv])
            ax.plot(lateral_face[:,0],lateral_face[:,1],'red')
            
            if bv[0]<min_x:
                min_x = bv[0]
            elif bv[0]>max_x:
                max_x = bv[0]
            if bv[1]<min_y:
                min_y = bv[1]
            elif bv[1]>max_y:
                max_y = bv[1]
        ax.set_aspect('equal')
        plt.xlim([min_x-0.5, max_x+0.5])
        plt.ylim([min_y-0.5, max_y+0.5])
    def draw_lumen(self,thickness):
        
        plt.ioff()
        plt.figure(figsize=(np.sqrt(self.n_cells),np.sqrt(self.n_cells)))
        ax = plt.gca()
        min_x, max_x = 0,0
        min_y, max_y = 0,0
        for i in range(self.n_cells):
            
            if self.apical_vertices[i,0]<min_x:
                min_x = self.apical_vertices[i,0]
            elif self.apical_vertices[i,0]>max_x:
                max_x = self.apical_vertices[i,0]

            if self.apical_vertices[i,1]<min_y:
                min_y = self.apical_vertices[i,1]
            elif self.apical_vertices[i,1]>max_y:
                max_y = self.apical_vertices[i,1]
            #draw the apical faces:
            left_overlap = self.cell_overlap_area(i,-1)
            right_overlap = self.cell_overlap_area(i,1)
            r = self.apical_radii[i]
            c,t0,t1 = arc_center_angles(self.apical_vertices[i-1],self.apical_vertices[i],r)
            if left_overlap>0:
                theta_diff_a = self.cell_overlap_area(i, -1, thetadiffs=True)
                t1 = (t1-theta_diff_a*180/np.pi)%360

                x,y = c[0] + r*np.cos(np.pi/180*t1), c[1]+r*np.sin(np.pi/180*t1)
                x0,y0 = tuple(self.apical_vertices[i-1])
            if right_overlap>0:
                theta_diff_b = self.cell_overlap_area(i,1,thetadiffs=True)
                t0 = (t0+theta_diff_b*180/np.pi)%360
                x,y = c[0] + r*np.cos(t0*np.pi/180), c[1] + r*np.sin(t0*np.pi/180)
                x0,y0 = tuple(self.apical_vertices[i])
            arc = patches.Arc(c,2*r,2*r,theta1=t0,theta2=t1,ec='green',lw=thickness)
            
            ax.add_patch(arc)
        ax.set_aspect('equal')
        plt.xlim([min_x-0.5, max_x+0.5])
        plt.ylim([min_y-0.5, max_y+0.5])
    def save(self, filename):
        '''
        write the whole LumenVertexModel object to a serialized (pickled) format
        '''
        f = open(filename,'wb')
        pickle.dump(self,f)
        f.close()

    @classmethod
    def from_file(self,filename):
        '''
        read a LumenVertexModel object from a serialized file
        '''
        f = open(filename,'rb')
        x = pickle.load(f)
        f.close()
        return x

