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
def plotElements(p,elements):
    for el in elements:
        #Q1 elements
        connectPoints(p[el[0][0]],p[el[0][1]])
        connectPoints(p[el[0][1]],p[el[0][3]])
        connectPoints(p[el[0][3]],p[el[0][2]])
        connectPoints(p[el[0][2]],p[el[0][0]])
        #Q2 elements (only visible part)
        connectPoints(p[el[1][4]],p[el[1][1]],color = 'orange')
        connectPoints(p[el[1][4]],p[el[1][3]],color = 'orange')
        connectPoints(p[el[1][4]],p[el[1][5]],color = 'orange')
        connectPoints(p[el[1][4]],p[el[1][7]],color = 'orange')

#hjelpefunksjon
def connectPoints(a,b,color = 'darkviolet'):
    a1, a2 = a[0], a[1]
    b1, b2 =b[0], b[1]
    plt.plot([a1,b1],[a2,b2], color, marker='',linewidth =.5)

#plottefunksjon for kanter
def plotEdges(inputEdge,p):
    for edge in inputEdge:
        connectPoints(p[int(edge[0])],p[int(edge[1])],color = 'r') 

def boundaries(points,N):
    non_homo_dir = [i*(5*(2**N)+1) for i in range(1,2**N)]
    homo_dir1 = [i for i in range(5*(2**N)+1)]
    homo_dir2 = [i for i in range((2**N)*(5*(2**N)+1),(2**N)*(5*(2**N)+1)+2*(2**N)+1)]
    homo_dir3 = [i for i in range((2**N)*(5*(2**N)+1)+3*(2**N),(2**N)*(5*(2**N)+1)+5*(2**N)+1)]
    homo_dir4 = [(2**N +1)*(5*(2**N)+1)+i*((2**N)+1) for i in range(2**N-1)]
    homo_dir5 = [(2**N +1)*(5*(2**N)+1)+i*((2**N)+1)-1 for i in range(1,2**N)]
    homo_dir6 = [i for i in range((2**N +1)*(5*(2**N)+1)+(2**N-1)*((2**N)+1),len(points))]
    homo_dir = np.concatenate((homo_dir1,homo_dir2,homo_dir3,homo_dir4,homo_dir5,homo_dir6))
    homo_neu = [i*(5*(2**N)+1)-1 for i in range(2,2**N+1)]
    return np.asarray(non_homo_dir), np.asarray(np.sort(homo_dir)), np.asarray(homo_neu)

def hoodTaylor(points,N):
    i = 0
    r = 1
    elements = []
    lin = []
    sq = []
    while r != (2**N)+1:
        lin = np.asarray([i,i+2,i+(10*(2**N)+2),i+(10*(2**N)+4)])
        sq = np.asarray([i,i+1,i+2,i+(5*(2**N)+1),i+(5*(2**N)+2),i+(5*(2**N)+3),i+(10*(2**N)+2),i+(10*(2**N)+3),i+(10*(2**N)+4)])
        el = [lin,sq]
        elements.append(el)
        i += 2
        if i==(5*(2**N))*r + r -1:
            i += 5*(2**N) +2
            r += 2

    r = 1
    i += 2*(2**N)
    ti = i
    while r != (2**N) +1:
        if r == 1:
            lin = np.asarray([i,i+2,i+4*(2**N)+2,i+4*(2**N)+4])
            sq = np.asarray([i, i+1,i+2,i+3*(2**N)+1,i+3*(2**N)+2,i+3*(2**N)+3,i+4*(2**N)+2,i+4*(2**N)+3,i+4*(2**N)+4])
            el = [lin,sq]
            elements.append(el)
        else:
            lin = np.asarray([i,i+2,i+2*(2**N)+2,i+2*(2**N)+4])
            sq = np.asarray([i,i+1,i+2,i+(2**N)+1,i+(2**N)+2,i+(2**N)+3,i+2*(2**N)+2,i+2*(2**N)+3,i+2*(2**N)+4])
            el = [lin,sq]
            elements.append(el)
        i += 2
        if i == ti + (2**N):
            if r == 1:
                i += 3*(2**N)+2
                r += 2
                ti = i
            else:
                i += (2**N)+2
                r += 2
                ti = i
    return elements

def phiSq(points,el):
    phi_el = np.ones((9,9))
    for i,p in enumerate(points[el]):
        phi_el[i] = [1,p[0],p[1],p[0]*p[1],p[0]**2,p[1]**2,(p[0]**2)*p[1],p[0]*(p[1]**2),(p[0]**2)*(p[1]**2)]
    basis_coeffs = np.zeros((9,9))

    for a in range(9):
        b = np.zeros(9)
        b[a] = 1
        c = np.linalg.solve(phi_el,b)
        basis_coeffs[a] = c
    return basis_coeffs

def phiLin(points,el):
    phi_el = np.ones((4,4))
    for i,p in enumerate(points[el]):
        phi_el[i] = [1,p[0],p[1],p[0]*p[1]]
    basis_coeffs = np.zeros((4,4))

    #basis_functions: konstantene til hver basisfunksjon for elementet: [[C1,C1x,C1y],[C2,C2x,C2y],[C3,C3x,C3y]]               
    for a in range(4):
        b = np.zeros(4)
        b[a] = 1
        c = np.linalg.solve(phi_el,b)    
        basis_coeffs[a] = c
    return basis_coeffs

def gauss2D(integrand,p,el,basis,i,j,basis_type = 'lin',degree = 3):
    if basis_type == 'lin': 
        p1,p4 = p[el[0]], p[el[3]]
    else:
        p1,p4 = p[el[0]], p[el[8]]
    a,b,c,d = p1[0], p4[0], p1[1], p4[1]
    h1 = (b-a)/2
    h2 = (b+a)/2
    h3 = (d-c)/2
    h4 = (d+c)/2
    integral = 0
    if degree ==3:
        weights = [5/9,8/9,5/9]
        eval_pts = [-np.sqrt(3/5),0,np.sqrt(3/5)]
    if degree == 4:
        w1 = (18-np.sqrt(30))/36
        w2 = (18+np.sqrt(30))/36
        ev1 = np.sqrt((3/7)-(2/7)*np.sqrt(6/5))
        ev2 = np.sqrt((3/7)+(2/7)*np.sqrt(6/5))
        weights = [w1,w2,w2,w1]
        eval_pts = [-ev2,-ev1,ev1,ev2]
    for w1,ev1 in zip(weights,eval_pts):
        for w2,ev2 in zip(weights,eval_pts):
            integral += w1*w2*h1*h3*integrand(h1*ev1+h2,h3*ev2+h4,basis,i,j)
    return integral

def stiffHelp(points,el,int_func,basis_type = 'lin'):
    if basis_type == 'lin':
        basis = phiLin(points,el)
    else:
        basis = phiSq(points,el)
    dim = len(basis)
    a_el = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            a_el[i,j] += gauss2D(int_func,points,el,basis,i,j,basis_type)
    return a_el

def stiffnessMat(int_func,points,elements,basis_type = 'lin'):
    A = sp.lil_matrix((len(points),len(points)))
    for el in elements:
        if basis_type == 'lin':
            a_el = stiffHelp(points,el[0],int_func,basis_type)
            for i,ai in enumerate(el[0]):
                for j,aj in enumerate(el[0]):
                    A[ai,aj] += a_el[i,j]
        else:
            a_el = stiffHelp(points,el[1],int_func,basis_type)
            for i,ai in enumerate(el[1]):
                for j,aj in enumerate(el[1]):
                    A[ai,aj] += a_el[i,j]
    return A

def loadVec(int_func,points,elements,basis_type = 'lin'):
    f = np.zeros((len(points)))
    i = 0
    for el in elements:
        if basis_type == 'lin':
            basis = phiLin(points,el[0])
            for j,fj in enumerate(el[0]):
                f[fj] += gauss2D(int_func,points,el[0],basis,i,j,basis_type=basis_type)
        else:
            basis = phiSq(points,el[1])
            for j,fj in enumerate(el[1]):
                f[fj] += gauss2D(int_func,points,el[1],basis,i,j,basis_type=basis_type)
    return f

#definer området:
N = 4
x = np.linspace(0,1,int(5*(2**N)+1))
y = np.linspace(0,.4,int(2*(2**N)+1))
X,Y = np.meshgrid(x,y)

px = X.ravel()
py = Y.ravel()

origpts = np.vstack((px,py)).T
points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005))]

elements = hoodTaylor(points,N)
non_homog_dir,homog_dir,neumann = boundaries(points,N)

plt.figure(1)
plotElements(points,elements)
#for i in range(len(points)):
#    plt.annotate(str(i),points[i])
plt.axis('scaled')
plt.savefig("figur1", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

#grad_phi = lambda x, y, c, i, j: 25*(c[i][1] + c[i][3]*y)*(c[j][1]+c[j][3]*y) + (c[i][2]+ c[i][3]*x)*(c[j][2]+ c[j][3]*x)

#f_int = lambda x, y, c, i, j: 3000*(c[j][0] + c[j][1]*x + c[j][2]*y + c[j][3]*x*y)

phi = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y + c[i][3]*x*y + c[i][4]*(x**2) + c[i][5]*(y**2) + c[i][6]*(x**2)*y + c[i][7]*x*(y**2) +c[i][8]*(x**2)*(y**2)
grad_phi_x = lambda x,y,c,i: c[i][1] + c[i][3]*y + 2*c[i][4]*x + 2*c[i][6]*x*y + c[i][7]*(y**2) + 2*c[i][8]*x*(y**2)
grad_phi_y = lambda x,y,c,i: c[i][2] + c[i][3]*x + 2*c[i][5]*y + c[i][6]*(x**2) + 2*c[i][7]*x*y+ 2*c[i][8]*(x**2)*y

bilinear_int = lambda x,y,c,i,j: 25*(grad_phi_x(x,y,c,i)*grad_phi_x(x,y,c,j) + grad_phi_y(x,y,c,i)*grad_phi_y(x,y,c,j))

f_int = lambda x,y,c,i,j: 75000*phi(x,y,c,j)

f_temp = loadVec(f_int,points,elements,basis_type='sq')

loadpts = points[:,0] > .699995
f = np.zeros((len(points)))
f[loadpts] = f_temp[loadpts]


A = stiffnessMat(bilinear_int,points,elements,basis_type='sq')
inner = np.array([i for i in range(len(points))])
inner = inner[~np.isin(np.arange(inner.size), np.concatenate((non_homog_dir,homog_dir)))]
outer = np.concatenate((non_homog_dir,homog_dir))

A_inner = A[inner]
A_inner = A_inner[:,inner]
f_inner = f[inner]
B = A[inner]
B = B[:,outer]


rg_homo = np.zeros_like(homog_dir)
rg_non_homo = np.zeros_like(non_homog_dir)
rgy = points[non_homog_dir][:,1]

rg_non_homo[:] = 2000*rgy*(.2-rgy)
rg = np.concatenate((rg_non_homo,rg_homo))
A_inner = A_inner.tocsr()
B = B.tocsr()

rhs = f_inner - B@rg

u_bar = solver(A_inner,rhs)

u = np.zeros((len(points)))
u[inner] = u_bar
u[outer] = rg

linear_elements = np.asarray([[el for el in inner[0]] for inner in elements])

square_elements = np.asarray([[el for el in inner[1]] for inner in elements])

plt.figure(2)

#ax1 = plt.tricontourf(points[:,0],points[:,1],linear_elements[:,[0,1,3]],u,levels = 20,cmap = 'rainbow')
#ax = plt.tricontourf(points[:,0],points[:,1],linear_elements[:,[0,3,2]],u,levels = 20,cmap = 'rainbow')
#ax = plt.tricontour(points[:,0],points[:,1],linear_elements[:,[0,1,3]],u,levels = 20,colors = 'black',linewidths=0.25)
#ax = plt.tricontour(points[:,0],points[:,1],linear_elements[:,[0,3,2]],u,levels = 20,colors = 'black',linewidths=0.25)

ax1 = plt.tricontourf(points[:,0],points[:,1],u,levels = 20,cmap = 'rainbow')
ax2 = plt.tricontour(points[:,0],points[:,1],u,levels = 20,colors = 'black',linewidths=0.25)
plt.axis('scaled')
plt.colorbar(ax1)
plt.savefig("figur2", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.close('all')

plotting = False
if plotting:

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
