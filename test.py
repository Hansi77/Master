from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time

import scipy.spatial as spsa
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as solver

#plottefunksjon for punkter
def plotPoints(p):
    for point in p:
        plt.plot(point[0],point[1],'ro',markersize=3)

#plottefunksjon for elementer
def plotElements(p,tri):
    for element in tri:
        i, j, k = element[0], element[1], element[2]
        #Connecting each corner:
        connectPoints(p[i],p[j])
        connectPoints(p[j],p[k])
        connectPoints(p[k],p[i])
    plt.plot()

#hjelpefunksjon
def connectPoints(a,b,color = 'darkviolet'):
    a1, a2 = a[0], a[1]
    b1, b2 =b[0], b[1]
    plt.plot([a1,b1],[a2,b2], color, marker='',linewidth =.5)

#plottefunksjon for kanter
def plotEdges(inputEdge,p):
    for edge in inputEdge:
        connectPoints(p[edge[0]],p[edge[1]],color = 'r') 

def halfCircleMeshMaker(x0,x1,y0,y1 = np.inf, Nr = 1):
    r = (x1 - x0)/2
    rpart = r/Nr
    rlist = np.asarray([rpart*n for n in range(1,Nr+1)])
    Nlist = [n+1 for n in range(Nr+1)]
    px = [x0+r]
    py = [y0]
    for rad,num in zip(rlist,Nlist):
        #if rotation == "down":
        #    thetalist = np.linspace(np.pi,2*np.pi,4*num+1)
        #else:
        thetalist = np.linspace(0,np.pi,4*num+1)
        for theta in thetalist:
            px.append(px[0]+np.cos(theta)*rad)
            py.append(py[0] +np.sin(theta)*rad)

    px = np.asarray(px)
    py = np.asarray(py)
    p = np.vstack((px,py))
    mesh = spsa.Delaunay(p.T)
    if y1 != np.inf:
        py = (py - y0)*((y1-y0)/r) + y0
    p = np.vstack((px,py))

    return p.T, mesh.simplices

def meshMaker(x0, x1, y0, y1, N, M):
    Lx = np.linspace(x0,x1,N+1)
    Ly = np.linspace(y0,y1,M+1)
    ax,ay = np.meshgrid(Lx,Ly)

    Ax = ax.ravel()
    Ay = ay.ravel()

    bp = np.vstack((Ax, Ay)).T

    mesh = spsa.Delaunay(bp)
    return bp, mesh.simplices

p1, mesh1 = meshMaker(0,2,.3,.5,40,6)
p2, mesh2 = meshMaker(1.3,1.7,.5,.6,8,2)
p3, mesh3 = meshMaker(1.3,1.7,.2,.3,8,2)
p4, mesh4 = halfCircleMeshMaker(.3,.7,.5,y1 = .55,Nr =4)
p5, mesh5 = halfCircleMeshMaker(.3,.7,.3,y1 = .25,Nr =4) 

p = np.concatenate((p1,p2,p3,p4,p5),axis = 0)
mesh = np.concatenate((mesh1,mesh2+len(p1),mesh3+len(p1)+len(p2),mesh4+len(p1)+len(p2)+len(p3),mesh5+len(p1)+len(p2)+len(p3)+len(p4)), axis = 0)

plt.figure(1)
plotElements(p,mesh)
plt.savefig("testmesh", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

p6, mesh6 = halfCircleMeshMaker(.3,.7,.5,y1 = .49,Nr =4)

plt.figure(2)
plotElements(p6,mesh6)
plt.savefig("halvfigur", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

#plt.figure(2)
#plotPoints(p)
#plt.savefig("noder.png")

'''
p1, mesh1 = meshMaker(0,1,0,.5,20,10)
p2, mesh2 = meshMaker(.3,.7,.5,1,8,10)

p = np.concatenate((p1,p2),axis =0)
mesh = np.concatenate((mesh1,mesh2+len(p1)),axis= 0)

plt.figure(1)
plotElements(p1,mesh1)
plt.savefig("testfigur1.png")

plt.figure(2)
plotElements(p2,mesh2)
plt.savefig("testfigur2.png")

plt.figure(3)
plotElements(p,mesh)
plt.savefig("testfigur3.png")
'''

plt.close("all")