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

def hoodTaylor(points,N):
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

def createLoadStiff(f_int,bilin_int,points,elements,lin_set = 0,basis_type = 'lin'):
    non_homog_dir,homog_dir,neumann = boundaries(points,N)

    loadpts = points[:,0] > .6999995
    f = np.zeros((len(points)))
    inner = np.array([i for i in range(len(points))])
    if basis_type == 'lin':
        lin_inv = np.array([i for i in range(len(points))])
        lin_inv = lin_inv[~np.isin(np.arange(lin_inv.size),lin_set)]
        non_homog_dir = non_homog_dir[np.isin(non_homog_dir,lin_set)]
        homog_dir = homog_dir[np.isin(homog_dir,lin_set)]
        f_temp = loadVec(f_int,points,elements,basis_type='lin')
        

        A = stiffnessMat(bilin_int,points,elements,basis_type='lin')
        delete = np.array(list(set(np.concatenate((lin_inv,non_homog_dir,homog_dir)))))
        inner = inner[~np.isin(np.arange(inner.size), delete)]
        outer = np.concatenate((non_homog_dir,homog_dir))
    else:
        f_temp = loadVec(f_int,points,elements,basis_type='sq')
        A = stiffnessMat(bilin_int,points,elements,basis_type='sq')
        inner = inner[~np.isin(np.arange(inner.size), np.concatenate((non_homog_dir,homog_dir)))]
        outer = np.concatenate((non_homog_dir,homog_dir))
    
    f[loadpts] = f_temp[loadpts]
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
    if basis_type == 'lin':
        u = u[lin_set]
        return u, points[lin_set]
    else:
        return u,points

def apply_mask(triang,p, alpha=0.1):
    # Mask triangles with sidelength bigger some alpha
    triangles = triang.triangles
    # Mask off unwanted triangles.
    xtri = p[triangles,0] - np.roll(p[triangles,0], 1, axis=1)
    ytri = p[triangles,1] - np.roll(p[triangles,1], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2),axis=1)
    # apply masking
    triang.set_mask(maxi > alpha)

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

def createA(int_func,points,elements):
    A = sp.lil_matrix((len(points),len(points)))
    for el in elements:
            a_el = submat(points,el,int_func)
            for i,ai in enumerate(el[1]):
                for j,aj in enumerate(el[1]):
                    A[ai,aj] += a_el[i,j]
    return A
    
def createD(int_func,points,elements):
    B = sp.lil_matrix((len(points),len(points)))
    for el in elements:
        a_el = submat(points,el,int_func,multibasis=True)
        for i,ai in enumerate(el[0]):
            for j,aj in enumerate(el[1]):
                B[ai,aj] += a_el[i,j]
    return B

def createAlpha(int_func,points,elements):
    i = 0
    alpha = np.zeros((len(points)))
    for el in elements:
        basis = phiLin(points,el[0])
        for j,aj in enumerate(el[0]):
            alpha[aj] += gauss2D(int_func,points,el[1],i,j,c2 = basis)
    return alpha

def createTest(points,N):
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

def createDomain(N):
    x = np.linspace(0,1,int(5*(2**N)+1))
    y = np.linspace(0,.4,int(2*(2**N)+1))
    X,Y = np.meshgrid(x,y)

    px = X.ravel()
    py = Y.ravel()
    origpts = np.vstack((px,py)).T
    points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005))]
    return points

#definer området:
N = 4
x = np.linspace(0,1,int(5*(2**N)+1))
y = np.linspace(0,.4,int(2*(2**N)+1))
X,Y = np.meshgrid(x,y)

px = X.ravel()
py = Y.ravel()

phi = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y + c[i][3]*x*y + c[i][4]*(x**2) + c[i][5]*(y**2) + c[i][6]*(x**2)*y + c[i][7]*x*(y**2) +c[i][8]*(x**2)*(y**2)
phi_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y + 2*c[i][4]*x + 2*c[i][6]*x*y + c[i][7]*(y**2) + 2*c[i][8]*x*(y**2)
phi_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x + 2*c[i][5]*y + c[i][6]*(x**2) + 2*c[i][7]*x*y+ 2*c[i][8]*(x**2)*y

zeta = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y +c[i][3]*x*y
zeta_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y
zeta_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x

mu1 = 10
mu2 = 100

a_bilin = lambda x,y,c,i,j: (1/mu1)*(phi_dx(x,y,c,j)*phi_dx(x,y,c,i) + phi_dy(x,y,c,j)*phi_dy(x,y,c,i))
b_bilin_x = lambda x,y,c1,c2,i,j: -phi_dx(x,y,c2,j)*zeta(x,y,c1,i)
b_bilin_y = lambda x,y,c1,c2,i,j: -phi_dy(x,y,c2,j)*zeta(x,y,c1,i)
alpha_int = lambda x,y,c,i,j: zeta(x,y,c,j)

origpts = np.vstack((px,py)).T
points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005))]
elements,lin_set = hoodTaylor(points,N)
non_homog_dir,homog_dir,neumann = boundaries(points,N)

A = createA(a_bilin,points,elements)
Dx = createD(b_bilin_x,points,elements)
Dy = createD(b_bilin_y,points,elements)

alpha = createAlpha(alpha_int,points,elements)
alpha = alpha[lin_set]

inner_sq = np.array([i for i in range(len(points))])
inner_sq = inner_sq[~np.isin(inner_sq, np.concatenate((non_homog_dir,homog_dir)))]
outer_sq = np.concatenate((non_homog_dir,homog_dir))
inner_lin = lin_set[np.isin(lin_set,inner_sq)]
outer_lin = lin_set[np.isin(lin_set,outer_sq)]
non_homog_lin = lin_set[np.isin(lin_set,non_homog_dir)]

#grenser for trykkrommet:

Ai = A[inner_sq]
Ai = Ai[:,inner_sq]
Dx = Dx[lin_set]
Gx = Dx[:,non_homog_dir]
Dx = Dx[:,inner_sq]
Dy = Dy[lin_set]
Gy = Dy[:,non_homog_dir]
Dy = Dy[:,inner_sq]
DxT = Dx.T
DyT = Dy.T

G = A[inner_sq]
G = G[:,non_homog_dir]

sq_x = points[non_homog_dir][:,1]
rgsq = mu2*sq_x*(.2-sq_x)

lift_sq = G@rgsq
lift_lin = Gx@rgsq

fx = np.zeros_like(lift_sq) - lift_sq 
fy = np.zeros_like(lift_sq)
fp = np.zeros_like(lift_lin) - lift_lin

rhs = np.concatenate((fx,fy,fp))
Block = sp.bmat([[Ai,None,DxT],[None,Ai,DyT],[Dx,Dy,None]]).tocsr()
u_bar = solver(Block,rhs)
uxinner = u_bar[:len(inner_sq)]
uyinner = u_bar[len(inner_sq):2*len(inner_sq)]
pinner = u_bar[2*len(inner_sq):]
ux = np.zeros(len(points))
uy = np.zeros(len(points))
p = np.zeros(len(lin_set))
ux[inner_sq] = uxinner
ux[non_homog_dir] = rgsq
uy[inner_sq] = uyinner
p = pinner

plt.figure(10)
exx = np.linspace(0,1,len(ux[non_homog_dir]))
plt.plot(exx,ux[non_homog_dir])
plt.savefig("hastighetsprofil.png")

#p[1:] = pinner

#p[0] = np.dot(alpha[1:],p[1:])
#print(np.dot(alpha,p))
print("average out velocity (x)")
print(sum(ux[non_homog_dir-1])/len(ux[non_homog_dir-1]))
print("average in velocity (x)")
print(sum(ux[non_homog_dir])/len(ux[non_homog_dir]))


tri1 = mtri.Triangulation(points[:,0],points[:,1])
apply_mask(tri1,points,alpha = 0.3/(2**N))
lin_pts = points[lin_set]
tri2 = mtri.Triangulation(lin_pts[:,0],lin_pts[:,1])
apply_mask(tri2,lin_pts,alpha=0.3/(2**(N-1)))

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