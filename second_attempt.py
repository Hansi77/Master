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

################forskjellige domener og relaterte grenser##################

#tradisjonellt rør
def typeZeroDom(points,N):
    i = 0
    r = 1
    elements = []
    lin = []
    sq = []
    lin_vec = []
    while r != (2**N)+1:
        lin = np.asarray([i,i+2,i+(10*(2**N)+2),i+(10*(2**N)+4)])
        sq = np.asarray([i,i+1,i+2,i+(5*(2**N)+1),i+(5*(2**N)+2),i+(5*(2**N)+3),i+(10*(2**N)+2),i+(10*(2**N)+3),i+(10*(2**N)+4)])
        el = [lin,sq]
        elements.append(el)
        lin_vec = np.concatenate((lin_vec,lin))
        i += 2
        if i==(5*(2**N))*r + r -1:
            i += 5*(2**N) +2
            r += 2
    return elements, np.array(list(set(lin_vec)),dtype=int)

def typeZeroBdry(points,N):
    non_homo_dir = [i*(5*(2**N)+1) for i in range(1,(2**N))]
    homo_dir1 = [i for i in range(5*(2**N)+1)]
    homo_dir2 = [i for i in range((2**N)*(5*(2**N)+1),((2**N)+1)*(5*(2**N)+1) )]
    homo_neu = [i*(5*(2**N)+1)-1 for i in range(2,2**N+1)]
    return np.asarray(non_homo_dir),np.asarray( np.concatenate((homo_dir1,homo_dir2))),np.asarray(homo_neu)

#rør med utstikker
def typeOneBdry(points,N):
    non_homo_dir = [i*(5*(2**N)+1) for i in range(1,(2**N))]
    homo_dir1 = [i for i in range(5*(2**N)+1)]
    homo_dir2 = [i for i in range((2**N)*(5*(2**N)+1),(2**N)*(5*(2**N)+1)+2*(2**N)+1)]
    homo_dir3 = [i for i in range((2**N)*(5*(2**N)+1)+3*(2**N),(2**N)*(5*(2**N)+1)+5*(2**N)+1)]
    homo_dir4 = [(2**N +1)*(5*(2**N)+1)+i*((2**N)+1) for i in range(2**N-1)]
    homo_dir5 = [(2**N +1)*(5*(2**N)+1)+i*((2**N)+1)-1 for i in range(1,2**N)]
    homo_dir6 = [i for i in range((2**N +1)*(5*(2**N)+1)+(2**N-1)*((2**N)+1),len(points))]
    homo_dir = np.concatenate((homo_dir1,homo_dir2,homo_dir3,homo_dir4,homo_dir5,homo_dir6))
    homo_neu = [i*(5*(2**N)+1)-1 for i in range(2,2**N+1)]
    return np.asarray(non_homo_dir), np.asarray(np.sort(homo_dir)), np.asarray(homo_neu)

def typeOneDom(points,N):
    i = 0
    r = 1
    elements = []
    lin = []
    sq = []
    lin_vec = []
    while r != (2**N)+1:
        lin = np.asarray([i,i+2,i+(10*(2**N)+2),i+(10*(2**N)+4)])
        sq = np.asarray([i,i+1,i+2,i+(5*(2**N)+1),i+(5*(2**N)+2),i+(5*(2**N)+3),i+(10*(2**N)+2),i+(10*(2**N)+3),i+(10*(2**N)+4)])
        el = [lin,sq]
        elements.append(el)
        lin_vec = np.concatenate((lin_vec,lin))
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
            lin_vec = np.concatenate((lin_vec,lin))
        else:
            lin = np.asarray([i,i+2,i+2*(2**N)+2,i+2*(2**N)+4])
            sq = np.asarray([i,i+1,i+2,i+(2**N)+1,i+(2**N)+2,i+(2**N)+3,i+2*(2**N)+2,i+2*(2**N)+3,i+2*(2**N)+4])
            el = [lin,sq]
            elements.append(el)
            lin_vec = np.concatenate((lin_vec,lin))
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
    return elements, np.array(list(set(lin_vec)),dtype=int)

#returnerer koeffisientene til de bikvadratiske basisfunksjonene
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

#returnerer koeffisientene til de bilineære basisfunksjonene
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

#numerisk integrasjonsfunksjon
def gauss2D(integrand,p,el,i,j,c1 = 0,c2 = 0,multibasis = False,degree = 3):
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
            if multibasis:
                integral += w1*w2*h1*h3*integrand(h1*ev1+h2,h3*ev2+h4,c1,c2,i,j)
            else:
                integral += w1*w2*h1*h3*integrand(h1*ev1+h2,h3*ev2+h4,c2,i,j)
    return integral

#maskeringsfunksjon for plotting
def apply_mask(triang,p, alpha=0.1):
    # Mask triangles with sidelength bigger some alpha
    triangles = triang.triangles
    # Mask off unwanted triangles.
    xtri = p[triangles,0] - np.roll(p[triangles,0], 1, axis=1)
    ytri = p[triangles,1] - np.roll(p[triangles,1], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2),axis=1)
    # apply masking
    triang.set_mask(maxi > alpha)

#hjelpefunksjon for generering av stivhets- og divergensmatriser
def submat(points,el,int_func,multibasis = False):
    if multibasis:
        c1 = phiLin(points,el[0])
        c2 = phiSq(points,el[1])
        dim1 = len(c1)
        dim2 = len(c2)
    else:
        c2 = phiSq(points,el[1])
        dim1 = len(c2)
        dim2 = dim1
    a_el = np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            if multibasis:
                a_el[i,j] += gauss2D(int_func,points,el[1],i,j,c1 = c1,c2 = c2,multibasis =multibasis)
            else:
                a_el[i,j] += gauss2D(int_func,points,el[1],i,j,c2 = c2)
    return a_el

#genererer stivhetsmatrisa
def createA(int_func,points,elements):
    A = sp.lil_matrix((len(points),len(points)))
    for el in elements:
            a_el = submat(points,el,int_func)
            for i,ai in enumerate(el[1]):
                for j,aj in enumerate(el[1]):
                    A[ai,aj] += a_el[i,j]
    return A

#genererer divergensmatrisa    
def createD(int_func,points,elements):
    B = sp.lil_matrix((len(points),len(points)))
    for el in elements:
        a_el = submat(points,el,int_func,multibasis=True)
        for i,ai in enumerate(el[0]):
            for j,aj in enumerate(el[1]):
                B[ai,aj] += a_el[i,j]
    return B

def createDomain(N,typ = 0):
    x = np.linspace(0,1,int(5*(2**N)+1))
    y = np.linspace(0,1,int(5*(2**N)+1))
    X,Y = np.meshgrid(x,y)

    px = X.ravel()
    py = Y.ravel()
    origpts = np.vstack((px,py)).T
    if typ == 0:
        points = origpts[origpts[:,1] <.20005]
        elements,lin_set = typeZeroDom(points,N)
        non_homog_dir,homog_dir,neumann = typeZeroBdry(points,N)
    elif typ == 1:
        points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005),origpts[:,1]<.40005))]
        elements,lin_set = typeOneDom(points,N)
        non_homog_dir,homog_dir,neumann = typeOneBdry(points,N)
    return points,elements,lin_set,non_homog_dir,homog_dir, neumann

#fjerner ønskede rader og kolonner fra inputmatrisa
def matrixShaver(Mat,rows,cols):
    Out = Mat[rows]
    Out = Out[:,cols]
    return Out

def solHelper(sol,lift,inner,points):
    uxinner = sol[:len(inner)]
    uyinner = sol[len(inner):2*len(inner)]
    p = sol[2*len(inner):]
    ux = np.zeros(len(points))
    uy = np.zeros(len(points))
    ux[inner] = uxinner
    ux[non_homog] = rg
    uy[inner] = uyinner
    return ux,uy,p

def plotHelp(points,lin_set,N):
    tri1 = mtri.Triangulation(points[:,0],points[:,1])
    apply_mask(tri1,points,alpha = 0.3/(2**N))
    lin_pts = points[lin_set]
    tri2 = mtri.Triangulation(lin_pts[:,0],lin_pts[:,1])
    apply_mask(tri2,lin_pts,alpha=0.3/(2**(N-1)))
    return tri1,tri2

#variabler, mu1 er amplitude på hastighetsprofil, mu2 er dynamsik viskositet
mu1 = 10
mu2 = 100

#definerer basisfunksjoner
phi = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y + c[i][3]*x*y + c[i][4]*(x**2) + c[i][5]*(y**2) + c[i][6]*(x**2)*y + c[i][7]*x*(y**2) +c[i][8]*(x**2)*(y**2)
phi_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y + 2*c[i][4]*x + 2*c[i][6]*x*y + c[i][7]*(y**2) + 2*c[i][8]*x*(y**2)
phi_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x + 2*c[i][5]*y + c[i][6]*(x**2) + 2*c[i][7]*x*y+ 2*c[i][8]*(x**2)*y

zeta = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y +c[i][3]*x*y
zeta_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y
zeta_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x

a_bilin = lambda x,y,c,i,j: (1/mu1)*(phi_dx(x,y,c,j)*phi_dx(x,y,c,i) + phi_dy(x,y,c,j)*phi_dy(x,y,c,i))
b_bilin_x = lambda x,y,c1,c2,i,j: -phi_dx(x,y,c2,j)*zeta(x,y,c1,i)
b_bilin_y = lambda x,y,c1,c2,i,j: -phi_dy(x,y,c2,j)*zeta(x,y,c1,i)

#bygger likningssystemet
N = 4
points,elements,lin_set,non_homog,homog,neu = createDomain(N,typ=0)

A = createA(a_bilin,points,elements)
Dx = createD(b_bilin_x,points,elements)
Dy = createD(b_bilin_y,points,elements)

#definerer indre noder
inner = np.array([i for i in range(len(points))])
inner = inner[~np.isin(inner, np.concatenate((non_homog,homog)))]

#fjerner nødvendige rader og kollonner
Ai = matrixShaver(A,inner,inner)
Dxi = matrixShaver(Dx,lin_set,inner)
Dyi = matrixShaver(Dy,lin_set,inner)
Gx = matrixShaver(Dx,lin_set,non_homog)
G = matrixShaver(A,inner,non_homog)

#definerer lifting-funksjonen
y_n = points[non_homog][:,1]
rg = mu2*y_n*(.2-y_n)

#lager høyresiden
fx = -G@rg
fy = np.zeros_like(fx)
fp = -Gx@rg
rhs = np.concatenate((fx,fy,fp))

#bygger blokkmatrisa og løser
Block = sp.bmat([[Ai,None,Dxi.T],[None,Ai,Dyi.T],[Dxi,Dyi,None]]).tocsr()
u_bar = solver(Block,rhs)
ux,uy,p = solHelper(u_bar,rg,inner,points)

print("average out velocity (x)")
print(sum(ux[neu])/len(ux[neu]))
print("average in velocity (x)")
print(sum(ux[non_homog])/len(ux[non_homog]))

#generer triangulering og maskerer for enklere plotting 
tri1,tri2 = plotHelp(points,lin_set,N)

plt.figure(0)
plotElements(points,elements)
#for i in range(len(points)):
#    plt.annotate(str(i),points[i])
plt.axis('scaled')
plt.savefig("figur0", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(1)
ax1 = plt.tricontourf(tri2,p,levels = 20,cmap = 'rainbow')
ax2 = plt.tricontour(tri2,p,levels = 20,colors = 'black',linewidths=0.25)
ax2 = plt.quiver(points[lin_set,0],points[lin_set,1],ux[lin_set],uy[lin_set])
plt.axis('scaled')
plt.colorbar(ax1)
plt.savefig("figur1", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(2)

ax1 = plt.tricontourf(tri1,ux,levels = 20,cmap = 'rainbow')
ax2 = plt.tricontour(tri1,ux,levels = 20,colors = 'black',linewidths=0.25)
plt.axis('scaled')
plt.colorbar(ax1)
plt.savefig("figur2", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(3)

ax1 = plt.tricontourf(tri1,uy,levels = 20,cmap = 'rainbow')
ax2 = plt.tricontour(tri1,uy,levels = 20,colors = 'black',linewidths=0.25)
plt.axis('scaled')
plt.colorbar(ax1)
plt.savefig("figur3", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(4)
plt.quiver(points[:,0],points[:,1],ux,uy)
plt.axis('scaled')
plt.savefig("figur4", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

plt.figure(5)
ax1 = plt.tricontourf(tri2,p,levels = 20,cmap = 'rainbow')
ax2 = plt.tricontour(tri2,p,levels = 20,colors = 'black',linewidths=0.25)
plt.colorbar(ax1)
plt.axis('scaled')
plt.savefig("figur5", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


plt.close('all')