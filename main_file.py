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

N = 5

vel = 2
types = [5,6,5,6]
mus = [[.5,.5,1],[0.5,0],[0,0,.5],[0,-.25]]
xi = 0
yi = 0

ux = []
uy = []
p = []

start_time = time.time()
i = 0
for typ,mu in zip(types,mus):
    mu.append(vel)
    ux_i,uy_i,p_i,points_i,lin_points_i, neu = arch.get_archetype(N,typ,mu,xi,yi)
    if i == 0:
        points = points_i
        lin_points = lin_points_i
    else:
        points = np.concatenate((points,points_i))
        lin_points = np.concatenate((lin_points,lin_points_i))
    xi = points_i[:,0].max()
    yi = points_i[:,1].min()
    vel = ux_i[neu].max()
    ux = np.concatenate((ux,ux_i))
    uy = np.concatenate((uy,uy_i))
    p += p_i[0]
    p = np.concatenate((p,p_i))
    i += 1

print("TIME:", time.time()-start_time)

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