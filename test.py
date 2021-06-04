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
N1 = lambda zeta,eta: .25*(1-zeta)*(1-eta)
N2 = lambda zeta,eta: .25*(1+zeta)*(1-eta)
N3 = lambda zeta,eta: .25*(1+zeta)*(1+eta)
N4 = lambda zeta,eta: .25*(1-zeta)*(1+eta)
N0 = lambda zeta,eta: [N1(zeta,eta),N2(zeta,eta),N3(zeta,eta),N4(zeta,eta)]
dx = lambda x: [x[1]-x[0],x[2]-x[3]]
dy = lambda y: [y[3]-y[0],y[2]-y[1]]
deta = lambda eta: [1-eta,1+eta]
det = lambda zeta,eta,x,y: (1/16)*np.dot(dx(x),deta(eta))*np.dot(dy(y),deta(zeta))- (1/16)*np.dot(dx(y),deta(eta))*np.dot(dy(x),deta(zeta))
ev = lambda zeta,eta,evi: np.dot(evi,N0(zeta,eta))

def gauss2D(integrand,p,el,i,j,c1 = 0,c2 = 0,multibasis = False,degree = 4):
    p1,p2,p3,p4 = p[el[0]],p[el[1]],p[el[3]],p[el[2]]
    xi = [p1[0],p2[0],p3[0],p4[0]]
    yi = [p1[1],p2[1],p3[1],p4[1]]

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
                integral += w1*w2*det(ev1,ev2,xi,yi)*integrand(ev(ev1,ev2,xi),ev(ev1,ev2,yi),c1,c2,i,j)
            else:
                integral += w1*w2*det(ev1,ev2,xi,yi)*integrand(ev(ev1,ev2,xi),ev(ev1,ev2,yi),c2,i,j)
    return integral

#maskeringsfunksjon for plotting
def apply_mask(triang,p, alpha=0.1,coord_mask = False):
    if coord_mask:
        x = p[triang.triangles,0].mean(axis=1) 
        y = p[triang.triangles,1].mean(axis=1)
        cond1 = np.logical_and(x < .4, y > .2)
        cond2 = np.logical_and(x > .6,y >.2)
        #cond2 = np.logical_and(y < .15, y > .05)
        #mask = np.logical_and(cond1,cond2)
        # apply masking
        triang.set_mask(cond2)
    else:
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
                a_el[i,j] += gauss2D(int_func,points,el[0],i,j,c1 = c1,c2 = c2,multibasis =multibasis)
            else:
                a_el[i,j] += gauss2D(int_func,points,el[0],i,j,c2 = c2)
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

#samlefunksjon for å genere Noder, elementer, kanter og indre noder, gitt type domene
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
    elif typ == 2:
        points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005),origpts[:,1]<.60005))]
        elements,lin_set = typeTwoDom(points,N)
        non_homog_dir,homog_dir,neumann = typeTwoBdry(points,N)
    elif typ == 3:
        points = origpts[np.logical_or(np.logical_and(origpts[:,1] <.20005,origpts[:,0] < .60005),np.logical_and(np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005),origpts[:,1]<.60005))]
        elements,lin_set = typeThreeDom(points,N)
        non_homog_dir,homog_dir,neumann = typeThreeBdry(points,N)
    elif typ == 4:
        points = origpts[np.logical_or(np.logical_and(origpts[:,1] <.20005,origpts[:,0] < .60005),np.logical_and(np.logical_and(origpts[:,0] > .39995,origpts[:,0] < .60005),origpts[:,1]<.60005))]
        points = np.vstack((-points[:,1]+.6,points[:,0]-.4)).T
        elements,lin_set = typeThreeDom(points,N)
        neumann,homog_dir,non_homog_dir = typeThreeBdry(points,N)
    elif typ == 5:
        points = origpts[origpts[:,1] <.20005]
        elements,lin_set = typeFiveDom(points,N)
        non_homog_dir,homog_dir,neumann = typeZeroBdry(points,N)
        homog_dir = np.concatenate((np.array(np.where(np.logical_and(np.logical_and(points[:,0]< 0.550005,points[:,0]> 0.449995),np.logical_and(points[:,1]< 0.150005,points[:,1]> 0.049995))))[0],homog_dir))
    elif typ == 6:
        points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,1] <.40005,origpts[:,0] <.60005))]
        elements,lin_set = typeSixDom(points,N)
        non_homog_dir,homog_dir,neumann = typeSixBdry(points,N)
    elif typ == 7:
        points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,1] <.40005,origpts[:,0] <.60005))]
        points[:,0] = 1 - points[:,0]
        elements,lin_set = typeSixDom(points,N)
        neumann,homog_dir,non_homog_dir = typeSixBdry(points,N)

    #definerer indre noder
    inner = np.array([i for i in range(len(points))])
    inner = inner[~np.isin(inner, np.concatenate((non_homog_dir,homog_dir)))]
    return points,elements,lin_set,non_homog_dir,homog_dir, neumann,inner

#fjerner ønskede rader og kolonner fra inputmatrisa
def matrixShaver(Mat,rows,cols):
    Out = Mat[rows]
    Out = Out[:,cols]
    return Out

#hjelpefunksjon
def solHelper(sol,lift,inner,points,non_homog):
    uxinner = sol[:len(inner)]
    uyinner = sol[len(inner):2*len(inner)]
    p = sol[2*len(inner):]
    ux = np.zeros(len(points))
    uy = np.zeros(len(points))
    ux[inner] = uxinner
    ux[non_homog] = lift
    uy[inner] = uyinner
    return ux,uy,p

def solHelper2(sol_r,RB_mat,points,non_homog,inner,lift):
    RB1 = RB_mat[:len(inner)]
    RB2 = RB_mat[len(inner):2*len(inner)]
    RB3 = RB_mat[2*len(inner):]
    temp = RB_mat.shape[1]

    uxr = sol_r[:temp]
    uyr = sol_r[temp:2*temp]
    pr = sol_r[2*temp:]
    uxinner = RB1@uxr
    uyinner = RB2@uyr
    p = RB3@pr

    ux = np.zeros(len(points))
    uy = np.zeros(len(points))
    ux[inner] = uxinner
    ux[non_homog] = lift
    uy[inner] = uyinner
    return ux,uy,p

#hjelpefunksjon
def plotHelp(points,N,mu_max):
    tri = mtri.Triangulation(points[:,0],points[:,1])
    apply_mask(tri,points,alpha= ((1+mu_max)*0.3)/(2**(N)),coord_mask=True)
    return tri

#plottefunksjoner
def contourPlotter(u,tri,title = "title",fname = "filename",newfig = True,save = True,cbar = True):
    if newfig:
        plt.figure()
        plt.title(title)
    ax1 = plt.tricontourf(tri,u,levels = 50,cmap = 'rainbow')
    #ax2 = plt.tricontour(tri,u,levels = 20,colors = 'black',linewidths=0.25)
    plt.axis('scaled')
    if cbar:
        plt.colorbar(ax1)
    if save:
        plt.savefig(fname, dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

def quiverPlotter(ux,uy,points,title = "title",fname = "filename",newfig = True, save = True):
    if newfig:
        plt.figure()
        plt.title(title)
    ax2 = plt.quiver(points[:,0],points[:,1],ux,uy)
    plt.axis('scaled')
    if save:
        plt.savefig(fname, dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

def initialize(inlet_velocity,N = 4,typ = 0,mu3=0,mu4=0):
    #variabler, mu1 er amplitude på hastighetsprofil, mu2 er dynamsik viskositet
    mu1 = 150

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

    #type domene: 0, 1, 2, 3, 4, 5
    points,elements,lin_set,non_homog,homog,neu,inner = createDomain(N,typ)
    o1 = [0,.4]
    o2 = [.4,.6]
    o3 = [.6,1]

    cond1 = np.logical_and(points[:,0] >= o1[0],points[:,0] < o1[1])
    cond2 = np.logical_and(points[:,0] >= o2[0],points[:,0] < o2[1])
    cond3 = np.logical_and(points[:,0] >= o3[0],points[:,0] <= o3[1])

    points[cond1,1] = mu3*(points[cond1,1]-.1) + points[cond1,1]
    points[cond2,1] = mu3*5*(o2[1]-points[cond2,0])*(points[cond2,1]-.1) + mu4*5*(points[cond2,0]-o2[0])*(points[cond2,1]-.1) + points[cond2,1]
    points[cond3,1] = mu4*(points[cond3,1]-.1) + points[cond3,1]

    #bygger stivhets- og divergensmatrisene
    A = createA(a_bilin,points,elements)
    Dx = createD(b_bilin_x,points,elements)
    Dy = createD(b_bilin_y,points,elements)

    #definerer lifting-funksjonen
    rg = inlet_velocity

    #fjerner nødvendige rader og kollonner
    Ai = matrixShaver(A,inner,inner)
    Dxi = matrixShaver(Dx,lin_set,inner)
    Dyi = matrixShaver(Dy,lin_set,inner)
    Gx = matrixShaver(Dx,lin_set,non_homog)
    G = matrixShaver(A,inner,non_homog)

    #lager høyresiden
    fx = -G@rg
    fy = np.zeros_like(fx)
    fp = -Gx@rg
    rhs = np.concatenate((fx,fy,fp))

    #bygger blokkmatrisa og løser
    Block = sp.bmat([[Ai,None,Dxi.T],[None,Ai,Dyi.T],[Dxi,Dyi,None]]).tocsr()
    u_bar = solver(Block,rhs)
    ux,uy,p = solHelper(u_bar,rg,inner,points,non_homog)

    print("total out volumeflow")
    if typ == 2:
        print(sum(np.sqrt(ux[neu]**2 + uy[neu]**2))/len(neu)*.4)
    else:
        print(sum(np.sqrt(ux[neu]**2+uy[neu]**2))/len(neu)*(.2+.2*mu4))
    print("total in volumeflow")
    print(sum(np.sqrt(ux[non_homog]**2 + uy[non_homog]**2))/len(non_homog)*(.2+.2*mu3))

    #generer triangulering og maskerer for enklere plotting 
    return ux,uy,p, points,elements,lin_set,neu

def get_velocity_type(N,typ,mu3):
    points,elements,lin_set,non_homog,homog,neu,inner = createDomain(N,typ)
    return points[non_homog][:,1] + mu3*(points[non_homog][:,1]-.1)

def omega(p):
    x = p[0]
    y = p[1]
    if x <= .45 and y <= .05:
        return 0
    elif y < .05 and x > .45 and x < .55:
        return 1
    elif y <= .05 and x >= .55:
        return 2
    elif x < .45 and y > .05 and y < .15:
        return 3
    elif x > .55 and y > .05 and y < .15:
        return 4
    elif x <= .45 and y >= .15:
        return 5
    elif y > .15 and x > .45 and x < .55:
        return 6
    elif y >= .15 and x >= .55:
        return 7
    else:
        return -1

#genererer stivhetsmatrisa
def createSubA(int_func,points,elements,domain):
    A = sp.lil_matrix((len(points),len(points)))
    for el in elements:
        if omega(points[el[1][4]]) == domain:
            a_el = submat(points,el,int_func)
            for i,ai in enumerate(el[1]):
                for j,aj in enumerate(el[1]):
                    A[ai,aj] += a_el[i,j]
        elif omega(points[el[1][4]]) == -1:
            print("invalid point!")
            return A
    return A

def createSubD(int_func,points,elements,domain):
    B = sp.lil_matrix((len(points),len(points)))
    for el in elements:
        if omega(points[el[1][4]]) == domain:
            a_el = submat(points,el,int_func,multibasis=True)
            for i,ai in enumerate(el[0]):
                for j,aj in enumerate(el[1]):
                    B[ai,aj] += a_el[i,j]
        elif omega(points[el[1][4]]) == -1:
            print("invalid point!")
            return B
    return B

def sparseSolver(Ax_set,Ay_set,Dx_set,Dy_set,Ax_rhs_set,Ay_rhs_set,Dx_rhs_set,q1,q2,q3,q4,mu,points,lin_set,non_homog,inner):
    A = sp.lil_matrix((len(inner),len(inner)))
    A_rhs = sp.lil_matrix((len(inner),len(non_homog)))
    for Ax,Ay,Ax_rhs,Ay_rhs,qx,qy in zip(Ax_set,Ay_set,Ax_rhs_set,Ay_rhs_set,q1,q2):
        A += qx*Ax
        A += qy*Ay
        A_rhs += qx*Ax_rhs
        A_rhs += qy*Ay_rhs

    Dx = sp.lil_matrix((len(lin_set),len(inner)))
    Dy = sp.lil_matrix((len(lin_set),len(inner)))
    D_rhs = sp.lil_matrix((len(lin_set),len(non_homog)))
    for D_x,D_y,Dx_rhs,qx,qy in zip(Dx_set,Dy_set,Dx_rhs_set,q3,q4):
        Dx += qx*D_x
        Dy += qy*D_y
        D_rhs += qx*Dx_rhs
    
    y_n = points[non_homog,1]

    c = -mu[3]/((1+mu[2])**2)
    rg = y_n*0
    for i,y in enumerate(y_n):
        if omega([0,y]) == 0:
            rg[i] = (20*mu[2]*y + 10*y)*(20*mu[2]*y + 10*y -2 -2*mu[2])
        if omega([0,y]) == 3:
            rg[i] = (10*y + mu[2])*(10*y -2 -mu[2])
        if omega([0,y]) == 5:
            rg[i] = (20*mu[2]*y + 10*y - 2*mu[2])*(20*mu[2]*y + 10*y -2 -4*mu[2])

    rg = c*rg

    #lager høyresiden
    fx = -A_rhs@rg
    fy = np.zeros_like(fx)
    fp = -D_rhs@rg
    rhs = np.concatenate((fx,fy,fp))

    #bygger blokkmatrisa og løser
    Block = sp.bmat([[A,None,Dx.T],[None,A,Dy.T],[Dx,Dy,None]]).tocsr()
    u_bar = solver(Block,rhs)
    return u_bar

def reducedSolver(Ax_r1,Ax_r2,Ay_r1,Ay_r2,Dx_r,Dy_r,DxT_r,DyT_r,Ax_rhs_r,Ay_rhs_r,Dx_rhs_r,q1,q2,q3,q4,mu,points,non_homog):
    A1 = sp.lil_matrix(Ax_r1[0].shape)
    A2 = sp.lil_matrix(Ax_r2[0].shape)
    A_rhs = sp.lil_matrix(Ax_rhs_r[0].shape)
    for Ax1,Ax2,Ay1,Ay2,Ax_rhs,Ay_rhs,qx,qy in zip(Ax_r1,Ax_r2,Ay_r1,Ay_r2,Ax_rhs_r,Ay_rhs_r,q1,q2):
        A1 += qx*Ax1
        A1 += qy*Ay1
        A2 += qx*Ax2
        A2 += qy*Ay2
        A_rhs += qx*Ax_rhs
        A_rhs += qy*Ay_rhs

    Dx = sp.lil_matrix(Dx_r[0].shape)
    Dy = sp.lil_matrix(Dy_r[0].shape)
    DxT = sp.lil_matrix(DxT_r[0].shape)
    DyT = sp.lil_matrix(DyT_r[0].shape)
    D_rhs = sp.lil_matrix(Dx_rhs_r[0].shape)
    for D_x,D_y,D_xT,D_yT,Dx_rhs,qx,qy in zip(Dx_r,Dy_r,DxT_r,DyT_r,Dx_rhs_r,q3,q4):
        Dx += qx*D_x
        Dy += qy*D_y
        DxT += qx*D_xT
        DyT += qy*D_yT
        D_rhs += qx*Dx_rhs
    
    y_n = points[non_homog,1]

    c = -mu[3]/((1+mu[2])**2)
    rg = y_n*0
    for i,y in enumerate(y_n):
        if omega([0,y]) == 0:
            rg[i] = (20*mu[2]*y + 10*y)*(20*mu[2]*y + 10*y -2 -2*mu[2])
        if omega([0,y]) == 3:
            rg[i] = (10*y + mu[2])*(10*y -2 -mu[2])
        if omega([0,y]) == 5:
            rg[i] = (20*mu[2]*y + 10*y - 2*mu[2])*(20*mu[2]*y + 10*y -2 -4*mu[2])

    rg = c*rg

    #lager høyresiden
    fx = -A_rhs@rg
    fy = np.zeros_like(fx)
    fp = -D_rhs@rg
    rhs = np.concatenate((fx,fy,fp))

    #bygger blokkmatrisa og løser
    Block = sp.bmat([[A1,None,DxT],[None,A2,DyT],[Dx,Dy,None]]).tocsr()
    u_bar = solver(Block,rhs)
    return u_bar


#x_e = [0,-1,2,-2,3,0,1,1]

#Nmat = N(1,1)

#dx = lambda x: [x[1]-x[0],x[2]-x[3]]
#dy = lambda y: [y[3]-y[0],y[2]-y[1]]

#x = [0,2,2,0]
#y = [0,-2,0,2]

#m = [3,4,5]

#print(max(m))



'''
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


mu3 = 1
mu5 = 2

J1 = np.asmatrix([[mu3+1,0],[0,2*mu5 +1]])

points = [3,3,3]
A = [sp.lil_matrix((len(points),len(points))) for i in range(5)]
A[0][0,0] = 1
print(A[0].todense())
omega_list  = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
print(len(omega_list))

def sparsesolver3(bilin,lin,N,dirichlet,qs):

    bilin_sum = sp.lil_matrix((N,N))
    for i in range(len(qs)):
        bilin_sum += bilin[i]*qs[i]

    #fjerner dirichlet-noder
    bilin_sum[dirichlet,:] = 0
    bilin_sum[dirichlet,dirichlet] = 1
    lin[dirichlet] = 0
    #løser
    sparse_A = sp.csr_matrix(bilin_sum)
    sparse_sol = solver(sparse_A,lin)
    solution = np.asarray(sparse_sol)
    #legger inn løsningen av ligningssystemet på riktig plass, i.e. setter dirichlet-nodene til verdi 0.
    solution[dirichlet] = 0
    return solution

'''


def typeSixBdry(points,N):
    non_homo_dir1 = [i*(5*(2**N)+1) for i in range(1,(2**N)+1)]    
    non_homo_dir2 = [i*(3*(2**N)+1)+((2**N)+1)*(5*(2**N)+1) for i in range((2**N)-1)]
    non_homo_dir = np.concatenate((non_homo_dir1,non_homo_dir2)) 
    homo_dir1 = [i for i in range(5*(2**N)+1)]
    homo_dir2 = [i for i in range(((2**N)-1)*(3*(2**N)+1)+((2**N)+1)*(5*(2**N)+1),len(points))]
    homo_dir3 = [i for i in range((2**N)*(5*(2**N)+1)+3*(2**N),(2**N)*(5*(2**N)+1)+5*(2**N)+1)]
    homo_dir4 = [i*(3*(2**N)+1)+((2**N)+1)*(5*(2**N)+1)-1 for i in range(1,(2**N))]
    homo_dir = np.concatenate((homo_dir1,homo_dir2,homo_dir3,homo_dir4))
    homo_neu = [i*(5*(2**N)+1)-1 for i in range(2,2**N+1)]
    return np.asarray(non_homo_dir), np.asarray(np.sort(homo_dir)), np.asarray(homo_neu)

def typeSixDom(points,N):
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
    ti = i
    while r != (2**N) +1:
        if r == 1:
            lin = np.asarray([i,i+2,i+4*2*(2**N)+2,i+4*2*(2**N)+4])
            sq = np.asarray([i, i+1,i+2,i+5*(2**N)+1,i+5*(2**N)+2,i+5*(2**N)+3,i+4*2*(2**N)+2,i+4*2*(2**N)+3,i+4*2*(2**N)+4])
            el = [lin,sq]
            elements.append(el)
            lin_vec = np.concatenate((lin_vec,lin))
        else:
            lin = np.asarray([i,i+2,i+2*3*(2**N)+2,i+2*3*(2**N)+4])
            sq = np.asarray([i,i+1,i+2,i+3*(2**N)+1,i+3*(2**N)+2,i+3*(2**N)+3,i+2*3*(2**N)+2,i+2*3*(2**N)+3,i+2*3*(2**N)+4])
            el = [lin,sq]
            elements.append(el)
            lin_vec = np.concatenate((lin_vec,lin))
        i += 2
        if i == ti + 3*(2**N):
            if r == 1:
                i += 5*(2**N)+2
                r += 2
                ti = i
            else:
                i += 3*(2**N)+2
                r += 2
                ti = i
    return elements, np.array(list(set(lin_vec)),dtype=int)

N =3

x = np.linspace(0,1,int(5*(2**N)+1))
y = np.linspace(0,1,int(5*(2**N)+1))
X,Y = np.meshgrid(x,y)

px = X.ravel()
py = Y.ravel()
origpts = np.vstack((px,py)).T

#points = origpts[np.logical_or(origpts[:,1] <.20005,np.logical_and(origpts[:,1] <.40005,origpts[:,0] <.60005))]
#points[:,0] = 1 - points[:,0]
#elements,lin_set = typeSixDom(points,N)
#non_homog_dir,homog_dir,neumann = typeSixBdry(points,N)

#definerer basisfunksjoner
phi = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y + c[i][3]*x*y + c[i][4]*(x**2) + c[i][5]*(y**2) + c[i][6]*(x**2)*y + c[i][7]*x*(y**2) +c[i][8]*(x**2)*(y**2)
phi_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y + 2*c[i][4]*x + 2*c[i][6]*x*y + c[i][7]*(y**2) + 2*c[i][8]*x*(y**2)
phi_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x + 2*c[i][5]*y + c[i][6]*(x**2) + 2*c[i][7]*x*y+ 2*c[i][8]*(x**2)*y

zeta = lambda x,y,c,i: c[i][0] + c[i][1]*x + c[i][2]*y +c[i][3]*x*y
zeta_dx = lambda x,y,c,i: c[i][1] + c[i][3]*y
zeta_dy = lambda x,y,c,i: c[i][2] + c[i][3]*x

a_bilin = lambda x,y,c,i,j: (phi_dx(x,y,c,j)*phi_dx(x,y,c,i) + phi_dy(x,y,c,j)*phi_dy(x,y,c,i))
b_bilin_x = lambda x,y,c1,c2,i,j: -phi_dx(x,y,c2,j)*zeta(x,y,c1,i)
b_bilin_y = lambda x,y,c1,c2,i,j: -phi_dy(x,y,c2,j)*zeta(x,y,c1,i)

#generer noder, elementer, kanter og indre noder
N = 4
#type domene: 0, 1, 2, 3, 4, 5
typ = 6
points,elements,lin_set,non_homog,homog,neu,inner = createDomain(N,typ)

#bygger stivhets- og divergensmatrisene
A = createA(a_bilin,points,elements)
Dx = createD(b_bilin_x,points,elements)
Dy = createD(b_bilin_y,points,elements)

#definerer lifting-funksjonen
y_n = points[non_homog][:,1]

rg = 100*(y_n)*(0.4-y_n)

#fjerner nødvendige rader og kollonner
Ai = matrixShaver(A,inner,inner)
Dxi = matrixShaver(Dx,lin_set,inner)
Dyi = matrixShaver(Dy,lin_set,inner)
Gx = matrixShaver(Dx,lin_set,non_homog)
G = matrixShaver(A,inner,non_homog)

#lager høyresiden
fx = -G@rg
fy = np.zeros_like(fx)
fp = -Gx@rg
rhs = np.concatenate((fx,fy,fp))

#bygger blokkmatrisa og løser
Block = sp.bmat([[Ai,None,Dxi.T],[None,Ai,Dyi.T],[Dxi,Dyi,None]]).tocsr()
u_bar = solver(Block,rhs)
ux,uy,p = solHelper(u_bar,rg,inner,points,non_homog)

#generer triangulering og maskerer for enklere plotting 
tri1 = plotHelp(points,N,1)
tri2 = plotHelp(points[lin_set],N-1,1)


#figur 0, domene
plt.figure()
plotElements(points,elements)
plt.title('Domain w/bilinear and biquadratic elements')
plt.axis('scaled')
plt.savefig("figur0_type"+str(typ), dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
#figur1, hastighetsfelt og trykk
contourPlotter(p,tri2,title = "Velocity and pressure",save = False)
quiverPlotter(ux[lin_set],uy[lin_set],points[lin_set],fname="figur1_type"+str(typ), newfig = False)
#figur2, x-hastighet
contourPlotter(ux,tri1,title="x-velocity, $u_x$",fname="figur2_type"+str(typ))
#figur3, y-hastighet
contourPlotter(uy,tri1,title="y-velocity, $u_y$",fname="figur3_type"+str(typ))
#figur4, hastighetsmagnitude
contourPlotter(np.sqrt(ux**2 + uy**2),tri1,title="Velocity-magnitude, $|u|$",fname="figur4_type"+str(typ))
#figur5, hastighetsfelt
quiverPlotter(ux,uy,points,fname="figur5_type"+str(typ),title= "Velocity-field")
#figur6, trykk
contourPlotter(p,tri2,title="Pressure, p",fname="figur6_type"+str(typ))

plt.close('all')


#plt.figure()
#plotElements(points,elements)
#plt.title('Domain w/bilinear and biquadratic elements')
#for i,p in enumerate(points):
#    plt.annotate(i,p)
#plt.axis('scaled')
#plt.savefig("aaa_type"+str(1), dpi=500, facecolor='w', edgecolor='w',orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

