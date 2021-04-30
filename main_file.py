from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from itertools import chain

import scipy.spatial as spsa
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as solver
import second_attempt as arch

N = 4
typ = 0
amp = 10
mu = [0,1,2,1]

y_n = arch.get_velocity_type(N,typ,mu[0])
inlet_velocity = amp*(y_n+mu[0]*.1)*(mu[0]*.1+.2-y_n)

for i in range(len(mu)-1):
    ux_i,uy_i,p_i,pts_i,els_i,lin_set_i,neu_i = arch.initialize(inlet_velocity,N=N,typ=typ,mu3 = mu[i],mu4 = mu[i+1])
    inlet_velocity = ux_i[neu_i]
    if i != 0:
        pts_i[:,0] += p_pts[:,0].max()
        p += p_i.max()
        ux = np.concatenate((ux,ux_i))
        uy = np.concatenate((uy,uy_i))
        ux_r = np.concatenate((ux_r,ux_i[lin_set_i]))
        uy_r = np.concatenate((uy_r,uy_i[lin_set_i]))

        p = np.concatenate((p,p_i))
        p_pts = np.concatenate((p_pts,pts_i[lin_set_i]))
        v_pts = np.concatenate((v_pts,pts_i))
    else:
        ux = ux_i
        uy = uy_i
        ux_r = ux_i[lin_set_i]
        uy_r = uy_i[lin_set_i]
        p = p_i
        p_pts = pts_i[lin_set_i]
        v_pts = pts_i

p_tri = arch.plotHelp(p_pts,N-1,max(mu))
v_tri = arch.plotHelp(v_pts,N,max(mu))

#figur 1
arch.contourPlotter(p,p_tri,title = "Velocity and pressure",save = False,cbar = True)
arch.quiverPlotter(ux_r,uy_r,p_pts,fname="figur1_type"+str(typ), newfig = False)
#figur2, x-hastighet
arch.contourPlotter(ux,v_tri,title="x-velocity, $u_x$",fname="figur2_type"+str(typ))
#figur3, y-hastighet
arch.contourPlotter(uy,v_tri,title="y-velocity, $u_y$",fname="figur3_type"+str(typ))
#figur4, hastighetsmagnitude
arch.contourPlotter(np.sqrt(ux**2 + uy**2),v_tri,title="Velocity-magnitude, $|u|$",fname="figur4_type"+str(typ))
#figur5, hastighetsfelt
arch.quiverPlotter(ux,uy,v_pts,fname="figur5_type"+str(typ),title= "Velocity-field")
#figur6, trykk
arch.contourPlotter(p,p_tri,title="Pressure, p",fname="figur6_type"+str(typ))

plt.close('all')