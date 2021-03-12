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

def phiSq():#p,element):
    phi_el = np.zeros((9,9))
    basis_coeffs = np.zeros((9,9))
    #i,j,k,l = element[0],element[1],element[2],element[3]
    #index_vec = [i,j,k,l]
    
    # Grusomt 
    phi_el[0,:] = [1,-1,-1,1,1,1,-1,-1,1]
    phi_el[1,:6] = [1,0,-1,0,0,1]
    phi_el[2,:] = [1,1,-1,-1,1,1,-1,1,1]
    phi_el[3,:5] = [1,-1,0,0,1]
    phi_el[4,0] = 1
    phi_el[5,:5] = [1,1,0,0,1]
    phi_el[6,:] = [1,-1,1,-1,1,1,1,-1,1]
    phi_el[7,:6] = [1,0,1,0,0,1]
    phi_el[8,:] = np.ones(9)

    #basis_functions: konstantene til hver basisfunksjon for elementet: [[C1,C1x,C1y],[C2,C2x,C2y],[C3,C3x,C3y]]               
    for a in range(9):
        b = np.zeros(9)
        b[a] = 1
        c = np.linalg.solve(phi_el,b)    
        basis_coeffs[a] = c
    return basis_coeffs

def phiLin():
    phi_el = np.ones((4,4))
    basis_coeffs = np.zeros((4,4))
    
    phi_el[0,[1,2]] = -1
    phi_el[1,[2,3]] = -1
    phi_el[2,[1,3]] = -1

    #basis_functions: konstantene til hver basisfunksjon for elementet: [[C1,C1x,C1y],[C2,C2x,C2y],[C3,C3x,C3y]]               
    for a in range(4):
        b = np.zeros(4)
        b[a] = 1
        c = np.linalg.solve(phi_el,b)    
        basis_coeffs[a] = c
    return basis_coeffs

def gauss2D(integrand,p,el,degree =0,basis=0):
    p1,p2,p3,p4 = p[el[0]], p[el[1]], p[el[2]], p[el[3]]
    a,b,c,d = p1[0], p2[0], p2[1], p3[1]
    weights = [5/9,8/9,5/9]
    h1 = (b-a)/2
    h2 = (b+a)/2
    h3 = (d-c)/2
    h4 = (d+c)/2
    integral = 0
    eval_pts = [-np.sqrt(3/5),0,np.sqrt(3/5)]
    for w1,ev1 in zip(weights,eval_pts):
        for w2,ev2 in zip(weights,eval_pts):
            integral += w1*w2*h1*h3*integrand(h1*ev1 + h2,h3*ev2+h4)
    return integral

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

csq = phiSq()
clin = phiLin()

refx = np.linspace(-1,1,51)
refX,refY = np.meshgrid(refx,refx)

a = 2
b = 4
c = 1
d = 2
p1 = [a,c]
p2 = [b,c]
p3 = [b,d]
p4 = [a,d]
h1 = (b-a)/2
h2 = (b+a)/2
h3 = (d-c)/2
h4 = (d+c)/2

pts = [p1,p2,p3,p4]
el = [0,1,2,3]

u = lambda x,y: x**3 + y**3

print(gauss2D(u,pts,el))

'''
for i in range(9):
    c = csq[i]
    phi = lambda x,y: c[0] + c[1]*x + c[2]*y + c[3]*x*y + c[4]*(x**2) + c[5]*(y**2)+c[6]*(x**2)*y + c[7]*x*(y**2)+c[8]*(x**2)*(y**2)
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(h1*refX+h2,h3*refY+h4,phi(refX,refY),cmap=plt.cm.Spectral)
    plt.title("reference element quadratic function"+str(i))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("basis"+str(i), dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    
    plt.close('all')


for i in range(4):
    c = clin[i]
    phi = lambda x,y: c[0] + c[1]*x + c[2]*y + c[3]*x*y
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(h1*refX+h2,h3*refY+h4,phi(refX,refY),cmap=plt.cm.Spectral)
    plt.title("reference element quadratic function"+str(i))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("linear_basis"+str(i), dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    
    plt.close('all')

#print(homog_dir)
#print(non_homog_dir)
#print(neumann)

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
'''