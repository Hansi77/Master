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

def halfCircleMeshMaker(x0,x1,y0,y1 = np.inf, Nr = 1,edgepoints = False):
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
        return bp, mesh.simplices, edge, south, east, np.flip(north), np.flip(west)

    return bp, mesh.simplices, edge

def nodeReplacer(p1,p2,edge,mesh):
    indexes = np.asarray([i for i in range(len(p2))])
    inverses = indexes[~np.isin(np.arange(indexes.size), edge)]
    replacepoints =p2[edge]
    p2 = p2[inverses]
    tol = 10E-10  

    for count,inv in enumerate(inverses):
        mesh = [[count + len(p1) if x == inv else x for x in sub] for sub in mesh]

    for count,point in enumerate(p1):
        for replace,index in zip(replacepoints,edge):
            if point[0] - tol < replace[0] < point[0] + tol and point[1] - tol < replace[1] < point[1] + tol:
                mesh = [[count if x == index else x for x in sub] for sub in mesh]
    
    mesh = np.asarray(mesh)
    p2 = np.asarray(p2)
    return p2, mesh

def domainCreator(halfcircles=1, rectangles=0,resolution=1,circleheight =1, rectangleheight = 1):
    tol = 10E-10
    length = 1/5 + (3*(halfcircles + rectangles))/5
    N = int(length*10*resolution)
    M = int(4*resolution)
    p,mesh,edge,south,east,north,west = meshMaker(0,length,.5,.7,N,M,edgepoints = True)
    position = 0.2
    circleheight *= 0.2
    rectangleheight *= 0.2

    while halfcircles != 0:
        newp, newmesh, edge, flat, curve = halfCircleMeshMaker(position,position +.4,.7,y1 = .7 + circleheight, Nr = int(2*resolution),edgepoints = True)
        newp, newmesh = nodeReplacer(p,newp,flat,newmesh)
        p = np.concatenate((p,newp),axis = 0)
        mesh = np.concatenate((mesh,newmesh),axis = 0)

        position += .6
        halfcircles -= 1

    while rectangles != 0:
        newp, newmesh, edge, south, east, north, west = meshMaker(position,position + .4,.7,.7 + rectangleheight, 4*resolution,2*resolution,edgepoints=True)
        newp, newmesh = nodeReplacer(p,newp,south,newmesh)
        p = np.concatenate((p,newp),axis = 0)
        mesh = np.concatenate((mesh,newmesh),axis = 0)

        position += .6
        rectangles -= 1

    pfx = p[:,0]
    pfy = p[:,1]

    pflip = np.vstack((pfx,1-pfy)).T
    pflip = pflip[np.asarray(south).max()+1:]
    meshflip = [[x + len(pflip) if x > np.asarray(south).max() else x for x in sub] for sub in mesh]
    p = np.concatenate((p,pflip,),axis = 0)
    mesh = np.concatenate((mesh,meshflip),axis = 0)

    return p, mesh, edge

starttime = time.time()
p, mesh, edge = domainCreator(2, 2,2,.5,.8)
print("Nodes:",len(p))
print("time:",time.time()-starttime)

plt.figure(1)
plotElements(p,mesh)
plt.axis('scaled')
plt.savefig("testmesh", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

#plt.figure(2)
#plotPoints(p)
#plt.savefig("testpoints", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.close("all")
