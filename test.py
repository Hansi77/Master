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
        connectPoints(p[int(edge[0])],p[int(edge[1])],color = 'r') 

def halfCircleMeshMaker(x0,x1,y0,y1 = np.inf, Nr = 1,edgepoints = True):
    r = (x1 - x0)/2
    rpart = r/Nr
    rlist = np.asarray([rpart*n for n in range(1,Nr+1)])
    Nlist = [n for n in range(1,Nr+1)]
    px = [x0+r]
    py = [y0]
    count = 0
    flatedge = [0]
    curvededge = []
    for rad,num in zip(rlist,Nlist):
        thetalist = np.linspace(0,np.pi,4*num+1)
        for theta in thetalist:
            if num == Nr:
                curvededge.append(count+1)
            if theta == np.pi or theta == 0:
                flatedge.append(count+1)
            px.append(px[0]+np.cos(theta)*rad)
            py.append(py[0] +np.sin(theta)*rad)
            count += 1

    px = np.asarray(px)
    py = np.asarray(py)
    p = np.vstack((px,py))
    mesh = spsa.Delaunay(p.T)
    if y1 != np.inf:
        py = (py - y0)*((y1-y0)/r) + y0
    p = np.vstack((px,py))
    sortflatedge = np.concatenate((flatedge[-1::-2],flatedge[1::2]),axis=0)

    edgepts = np.concatenate((sortflatedge[:-1],curvededge))
    edge = np.zeros((len(edgepts)-1,2))
    for i in range(len(edgepts)-1):
        edge[i][0] = int(edgepts[i])
        edge[i][1] = int(edgepts[i+1])
    if edgepoints:
        return p.T, mesh.simplices, edge, sortflatedge, curvededge
    else:
        return p.T, mesh.simplices,edge

def meshMaker(x0, x1, y0, y1, N, M,edgepoints = False):
    Lx = np.linspace(x0,x1,N+1)
    Ly = np.linspace(y0,y1,M+1)
    ax,ay = np.meshgrid(Lx,Ly)

    Ax = ax.ravel()
    Ay = ay.ravel()

    bp = np.vstack((Ax, Ay)).T
    mesh = spsa.Delaunay(bp)
    south = []
    east = []
    west = []
    north = []
    count = 0
    for point in bp:
        if point[1] == y0:
            south.append(count)
        if point[1] == y1:
            north.append(count)
        if point[0] == x0:
            west.append(count)
        if point[0] == x1:
            east.append(count)
        count += 1
    north = np.flip(north)
    west = np.flip(west)

    edgepts = np.concatenate((south,east[1:],north[1:],west[1:]),axis = 0)
    edge = np.zeros((len(edgepts)-1,2))
    for i in range(len(edgepts)-1):
        edge[i][0] = int(edgepts[i])
        edge[i][1] = int(edgepts[i+1])
    
    if edgepoints:
        return bp, mesh.simplices, south, east, np.flip(north), np.flip(west), edge

    return bp, mesh.simplices, edge

p1, mesh1,south,north,west,east,junk = meshMaker(0,2,.3,.5,40,6,edgepoints = True)
p2, mesh2,junk = meshMaker(1.3,1.7,.5,.6,8,2)
p3, mesh3,edge = meshMaker(1.3,1.7,.2,.3,8,2)
p4, mesh4, flatedge4, curvededge4,edge4 = halfCircleMeshMaker(.3,.7,.5,y1 = .55,Nr =4)
p5, mesh5, flatedge5, curvededge5,edge5 = halfCircleMeshMaker(.3,.7,.3,y1 = .25,Nr =4)

print(south)
print(north)
print(west)
print(east)

p = np.concatenate((p1,p2,p3,p4,p5),axis = 0)
mesh = np.concatenate((mesh1,mesh2+len(p1),mesh3+len(p1)+len(p2),mesh4+len(p1)+len(p2)+len(p3),mesh5+len(p1)+len(p2)+len(p3)+len(p4)), axis = 0)

plt.figure(1)
plotElements(p,mesh)
plt.savefig("testmesh", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

#p6, mesh6, edge = halfCircleMeshMaker(.3,.7,.5,y1 = .6,Nr =4,edgepoints = False)

#plt.figure(2)
#plotElements(p6,mesh6)
#plt.savefig("halvfigur", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(3)
plotEdges(edge,p3)
plt.savefig("edge.png")
plt.close("all")