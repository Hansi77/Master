from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from itertools import chain
import os
import scipy.spatial as spsa
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as solver
import third_attempt as arch

if __name__ == "__main__":
    N = 5
    vel = 2 #inlet max velocity
    #Type 3: [length scale pre corner 0.4(1+mu),inlet width 0.2(1+mu),outlet width 0.2(1+mu),length scale post corner 0.4(1+mu)]
    #Type 5: [length pre-hole 0.45(1+mu),length post-hole 0.45(1+mu), width 0.2(1+mu)]
    #Type 6: [width outlet 0.2(1+mu),width inlet 0.4(1+mu)]
    #Type 7: [width inlet 0.2(1+mu),width outlet 0.4(1+mu)]
    types = [5,6,3,7,3,5]
    mus = [[0,0,1],[.5,0],[-.25,.5,0,-.25],[0,0],[-.25,1,.5,.25],[1,1,0.5]] #mu-list without the velocity parameter

    if len(types) != len(mus):
        print("Different sizes of inndata lists! exiting program")
        sys.exit(1)

    ux = []
    uy = []
    p = []

    start_time = time.time()
    i = 0
    rotate = 0
    xi = 0
    yi = 0
    rot_mat = np.asarray([[0,-1],[1,0]])

    for typ,mu in zip(types,mus):
        rotate = rotate%4
        mu.append(vel)
        ux_i,uy_i,p_i,points_i,lin_points_i, out =  arch.get_archetype(N,typ,mu)
        #not_neu = np.ones(len(points_i), np.bool)
        #not_neu[out] = 0
        if typ == 3:
            vel = uy_i[out].max()
        else:
            vel = ux_i[out].max()

        if rotate == 0:
            xi_new = points_i[:,0].max()
            yi_new = points_i[:,1].max()
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            xi += xi_new
            if typ == 3:
                yi += yi_new
        elif rotate == 1:
            temp = ux_i
            ux_i = -uy_i
            uy_i = temp
            xi_new = points_i[:,1].max()
            for i in range(len(points_i)):
                points_i[i] = rot_mat@points_i[i]
            for i in range(len(lin_points_i)):
                lin_points_i[i] = rot_mat@lin_points_i[i]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            yi = points_i[:,1].max()
            if typ == 3:
                xi -= xi_new
        elif rotate == 2:
            ux_i = -ux_i
            uy_i = -uy_i
            yi_new = points_i[:,1].max()
            xi_new = points_i[:,0].max()
            for i in range(len(points_i)):
                points_i[i] = rot_mat@rot_mat@points_i[i]
            for i in range(len(lin_points_i)):
                lin_points_i[i] = rot_mat@rot_mat@lin_points_i[i]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            xi -= xi_new
            if typ == 3:
                yi -= yi_new
        elif rotate == 3:
            temp = ux_i
            ux_i = uy_i
            uy_i = -temp
            yi_new = points_i[:,0].max()
            xi_new = points_i[:,1].max()
            for i in range(len(points_i)):
                points_i[i] = rot_mat@rot_mat@rot_mat@points_i[i]
            for i in range(len(lin_points_i)):
                lin_points_i[i] = rot_mat@rot_mat@rot_mat@lin_points_i[i]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            yi -= yi_new
            if typ == 3:
                xi += xi_new

        if i == 0:
            points = points_i
            lin_points = lin_points_i
        else:
            points = np.concatenate((points,points_i))
            lin_points = np.concatenate((lin_points,lin_points_i))
        if typ == 3:
            rotate += 1

        
        ux = np.concatenate((ux,ux_i))
        uy = np.concatenate((uy,uy_i))
        if typ == 7:
            p += p_i.max()
        else:
            p += p_i[0]
        p = np.concatenate((p,p_i))
        i += 1

    print("TIME:", time.time()-start_time)

    '''
    epsilon = 0.001
    y_out = points[points[:,0] > points[:,0].max() -epsilon,1]
    u_out = ux[points[:,0] > points[:,0].max() -epsilon]

    y_in = points[points[:,0] < points[:,0].min() +epsilon,1]
    u_in = ux[points[:,0] < points[:,0].min() +epsilon]

    Q_in = 0
    for i in range(len(y_in)-1):
        Q_in += .5*(u_in[i] + u_in[i+1])*(y_in[i+1]-y_in[i])
    Q_out = 0
    for i in range(len(y_out)-1):
        Q_out += .5*(u_out[i] + u_out[i+1])*(y_out[i+1]-y_out[i])
    '''

    #print("Q_in:", np.round(Q_in,4))
    #print("Q_out:", np.round(Q_out,4))

    p_tri = arch.plotHelp(lin_points,N-1,1.5,coord_mask = False)
    v_tri = arch.plotHelp(points,N,1.5,coord_mask = False)

    #figur2, x-hastighet
    arch.contourPlotter(ux,v_tri,title="x-velocity, $u_x$",fname="combo_figur1",HD = True)
    #figur3, y-hastighet
    arch.contourPlotter(uy,v_tri,title="y-velocity, $u_y$",fname="combo_figur2",HD = True)
    #figur4, hastighetsmagnitude
    arch.contourPlotter(np.sqrt(ux**2 + uy**2),v_tri,title="Velocity-magnitude, $|u|$",fname="combo_figur3",HD = True)
    #figur6, trykk
    arch.contourPlotter(p,p_tri,title="Pressure, p",fname="combo_figur4",HD = True)

    plt.close('all')