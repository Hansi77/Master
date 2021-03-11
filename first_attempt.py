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

def plotSqElements(p,elQ):
    for el in elQ:
        connectPoints(p[el[0]],p[el[1]])
        connectPoints(p[el[1]],p[el[2]])
        connectPoints(p[el[2]],p[el[3]])
        connectPoints(p[el[3]],p[el[0]])
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

def quadElements(points,N):
    elQ = []
    non_homo_dir = []
    homo_dir = [i for i in range(5*(2**N)+1)]
    homo_neu = []
    i = 0
    r = 1

    while r != (2**N)+1:
        elQ.append([i,i+1,i+(5*(2**N)+2),i+(5*(2**N)+1)])
        i += 1
        if i==(5*(2**N))*r + r - 1:
            if r != (2**N):
                non_homo_dir.append(i+1)
            if r != 1:
                homo_neu.append(i)
            i += 1
            r += 1
    r = 1
    for j in chain(range(i,i+2*(2**N)+1),range(i+3*(2**N),i +5*(2**N)+2)):
        homo_dir.append(j)
    i += 2*(2**N)
    ti = i
    while r != (2**N) +1:
        if r == 1:
            elQ.append([i,i+1,i+3*(2**N)+2,i+3*(2**N)+1])
        else:
            elQ.append([i,i+1,i+(2**N)+2,i+(2**N)+1])
        i += 1
        if i == ti + (2**N)*r + r - 1:
            if r == 1:
                i += 2*(2**N)+1
                r += 1
                ti += 2*(2**N)
            else:
                homo_dir.append(i)
                homo_dir.append(i+1)
                i += 1
                r += 1
    for j in range(i+1,i+(2**N)+1):
        homo_dir.append(j)
    return np.asarray(elQ),homo_dir,non_homo_dir,homo_neu
    
#definer området:
N = 2
x = np.linspace(0,1,int(5*(2**N)+1))
y = np.linspace(0,.4,int(2*(2**N)+1))
X,Y = np.meshgrid(x,y)

px = X.ravel()
py = Y.ravel()

origpts = np.vstack((px,py)).T
points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005))]

elements,homog_dir,non_homog_dir,neumann = quadElements(points,N)

print(homog_dir)
print(non_homog_dir)
print(neumann)

plt.figure(1)
plotSqElements(points,elements)
for i in range(len(points)):
    plt.annotate(str(i),points[i])
plt.axis('scaled')
plt.savefig("figur1", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

u = np.sin(points[:,0]) + np.cos(points[:,1])

plt.figure(2)

ax1 = plt.tricontourf(points[:,0],points[:,1],elements[:,:3],u,levels = 30,cmap = 'rainbow')
ax = plt.tricontourf(points[:,0],points[:,1],elements[:,[0,2,3]],u,levels = 30,cmap = 'rainbow')
ax = plt.tricontour(points[:,0],points[:,1],elements[:,:3],u,levels = 30,colors = 'black',linewidths=0.3)
ax = plt.tricontour(points[:,0],points[:,1],elements[:,[0,2,3]],u,levels = 30,colors = 'black',linewidths=0.3)
plt.axis('scaled')
plt.colorbar(ax1)
plt.savefig("figur2", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.close('all')