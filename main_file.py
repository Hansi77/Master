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
mu2 = 10
mu3 = 1
mu4 = 0

y_n = arch.get_velocity_type(N,typ,mu3)

inlet_velocity = mu2*(y_n+mu3*.1)*((mu3*.2)+.2-(y_n+mu3*.1))

ux,uy,p,tri1,tri2,points,elements,lin_set,neu = arch.initialize(inlet_velocity,N=N,typ=typ,mu3 = mu3,mu4 = mu4)

ux2,uy2,p2,tri3,tri4,points2,elements2,lin_set2,neu2 = arch.initialize(ux[neu],N=N,typ=typ,mu3 = mu4,mu4 = mu3,p_init=p.min())
points2[:,0] += 1
tri3,tri4 = arch.plotHelp(points2,lin_set2,N,mu3,mu4)


fname = "testfigur"
title = "testfigur"
p3 = np.concatenate((p,p2))
points3 = np.concatenate((points[lin_set],points2[lin_set2]))

tri = mtri.Triangulation(points3[:,0],points3[:,1])
arch.apply_mask(tri,points3,alpha= ((1+max(mu3,mu4))*0.3)/(2**(N-1)))

plt.figure()
plt.title(title)
ax1 = plt.tricontourf(tri,p3,levels = 50,cmap = 'rainbow')
plt.colorbar(ax1)
plt.axis('scaled')
arch.quiverPlotter(ux[lin_set],uy[lin_set],points[lin_set], newfig = False,save = False)
arch.quiverPlotter(ux2[lin_set2],uy2[lin_set2],points2[lin_set2],fname="figur1_type"+str(typ), newfig = False)

#plt.figure()
#arch.plotElements(points,elements)
#plt.title('Domain w/bilinear and biquadratic elements')
#plt.axis('scaled')
#plt.savefig("figur0_type"+str(typ), dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
#figur1, hastighetsfelt og trykk
#arch.contourPlotter(np.concatenate((p1,p2)),title = "Velocity and pressure",save = False)
#arch.quiverPlotter(ux[lin_set],uy[lin_set],points[lin_set], newfig = False,save = False)
#arch.quiverPlotter(ux2[lin_set2],uy2[lin_set2],points2[lin_set2],fname="figur1_type"+str(typ), newfig = False)
#figur2, x-hastighet
#arch.contourPlotter(ux,tri1,title="x-velocity, $u_x$",fname="figur2_type"+str(typ))
#figur3, y-hastighet
#arch.contourPlotter(uy,tri1,title="y-velocity, $u_y$",fname="figur3_type"+str(typ))
#figur4, hastighetsmagnitude
#arch.contourPlotter(np.sqrt(ux**2 + uy**2),tri1,title="Velocity-magnitude, $|u|$",fname="figur4_type"+str(typ))
#figur5, hastighetsfelt
#arch.quiverPlotter(ux,uy,points,fname="figur5_type"+str(typ),title= "Velocity-field")
#figur6, trykk
#arch.contourPlotter(p,tri2,title="Pressure, p",fname="figur6_type"+str(typ))
