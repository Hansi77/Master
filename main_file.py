import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import third_attempt as arch

if __name__ == "__main__":
    N = 5
    vel = 2 #inlet max velocity
    #STRAIGHT PIPE
    #Type 0: [length 1+mu, width 0.2(1+mu)]
    #LEFT CORNER PIPE
    #Type 3: [length scale pre corner 0.4(1+mu),inlet width 0.2(1+mu),outlet width 0.2(1+mu),length scale post corner 0.4(1+mu)]
    #RIGHT CORNER PIPE
    #Type 4: [length scale pre corner 0.4(1+mu),inlet width 0.2(1+mu),outlet width 0.2(1+mu),length scale post corner 0.4(1+mu)]
    #PARTIALLY BLOCKED PIPE
    #Type 5: [length pre-hole 0.45(1+mu),length post-hole 0.45(1+mu), width 0.2(1+mu)]
    #CONTRACTION PIPE
    #Type 6: [width outlet 0.2(1+mu),width inlet 0.4(1+mu)]
    #EXPANSION PIPE
    #Type 7: [width inlet 0.2(1+mu),width outlet 0.4(1+mu)]
    #types = [5,6,3,7,3,5]
    #mus = [[0,0,1],[.5,0],[-.25,.5,0,-.25],[0,0],[-.25,1,.5,.25],[1,1,0.5]] #mu-list without the velocity parameter
    #types = [5,5,3,5,5]
    #mus = [[0,0,0],[0,0,0],[0,0,0,0],[0,0,0],[0,0,0]]
    #types = [0,0,0,0,0]
    #mus = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    types = [0,3,3,0,4,4,0,3,3,0,4,4,0]
    mus = [[0,0],[0,0,0,-.5],[-.5,0,0,0],[0,0],[0,0,0,-.5],[-.5,0,0,0],[0,0],[0,0,0,-.5],[-.5,0,0,0],[0,0],[0,0,0,-.5],[-.5,0,0,0],[0,0]]
    if types[0] == 0:
        print("INLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mus[0][1]), "m**2/s")
    elif types[0] == 3 or types[0] == 4:
        print("INLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mus[0][1]), "m**2/s")
    elif types[0] == 5:
        print("INLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mus[0][-1]), "m**2/s")
    elif types[0] == 6:
        print("INLET \"VOLUME\" FLOW:", (2/3)*vel*0.4*(1+mus[0][1]), "m**2/s")
    elif types[0] == 7:
        print("INLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mus[0][0]), "m**2/s")

    if len(types) != len(mus):
        print("Different sizes of inndata lists! exiting program")
        sys.exit(1)

    ux = []
    uy = []
    p = []

    start_time = time.time()
    i = 0
    rotate = 0
    xi = 0
    yi = 0
    rot_mat = np.asarray([[0,-1],[1,0]])
    rot_down = 0

    for typ,mu in zip(types,mus):

        rotate = rotate%4
        mu.append(vel)
        ux_i,uy_i,p_i,points_i,lin_points_i, out =  arch.get_archetype(N,typ,mu)
        if typ == 0:
            outlet = np.concatenate(([0],ux_i[out],[0]))
            outpts = points_i[out,1]
            dx = abs(outpts[1]-outpts[0])
            outpts = outpts - outpts[0] + dx
            outletpts = np.concatenate(([outpts[0] - dx] ,outpts,[outpts[-1] + dx]))
            poly = np.polyfit(outletpts,outlet,deg=2)
            #vel = (1/2)*poly[0]*(0.2*(1+mu[2]))**2 + (3/4)*poly[1]*0.2*(1+mu[2]) #+(3/2)*poly[2]
            vel = abs(-(poly[1]**2)/(4*poly[0])+poly[2])
            if i == len(types)-1:
                print("OUTLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mu[1]), "m**2/s")
        elif typ == 3 or typ == 4:
            outlet = np.concatenate(([0],uy_i[out],[0]))
            outpts = points_i[out,0]
            dx = abs(outpts[1]-outpts[0])
            outpts = outpts - outpts[0] + dx
            outletpts = np.concatenate(([outpts[0] - dx] ,outpts,[outpts[-1] + dx]))
            poly = np.polyfit(outletpts,outlet,deg=2)
            #vel = (1/2)*poly[0]*(0.2*(1+mu[2]))**2 + (3/4)*poly[1]*0.2*(1+mu[2]) #+(3/2)*poly[2]
            vel = abs(-(poly[1]**2)/(4*poly[0])+poly[2])
            if i == len(types)-1:
                print("OUTLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mu[2]), "m**2/s")
        elif typ == 5:
            outlet = np.concatenate(([0],ux_i[out],[0]))
            outpts = points_i[out,1]
            dx = abs(outpts[1]-outpts[0])
            outletpts = np.concatenate(([outpts[0] - dx],outpts,[outpts[-1] + dx]))
            poly = np.polyfit(outletpts,outlet,deg=2)
            #vel = (1/2)*poly[0]*(0.2*(1+mu[-2]))**2 + (3/4)*poly[1]*0.2*(1+mu[-2]) #+(3/2)*poly[2]
            vel = abs(-(poly[1]**2)/(4*poly[0])+poly[2])
            if i == len(types)-1:
                print("OUTLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mu[-2]), "m**2/s")
        elif typ == 6:
            outlet = np.concatenate(([0],ux_i[out],[0]))
            outpts = points_i[out,1]
            dx = abs(outpts[1]-outpts[0])
            outletpts = np.concatenate(([outpts[0] - dx],outpts,[outpts[-1] + dx]))
            poly = np.polyfit(outletpts,outlet,deg=2)
            #vel = (1/2)*poly[0]*(0.2*(1+mu[0]))**2 + (3/4)*poly[1]*0.2*(1+mu[0]) #+(3/2)*poly[2]
            vel = abs(-(poly[1]**2)/(4*poly[0])+poly[2])
            if i == len(types)-1:
                print("OUTLET \"VOLUME\" FLOW:", (2/3)*vel*0.2*(1+mu[0]), "m**2/s")
        elif typ == 7:
            outlet = np.concatenate(([0],ux_i[out],[0]))
            outpts = points_i[out,1]
            dx1 = abs(outpts[1]-outpts[0])
            dx2 = abs(outpts[-1]-outpts[-2])
            outletpts = np.concatenate(([outpts[0] - dx1],outpts,[outpts[-1] + dx2]))
            poly = np.polyfit(outletpts,outlet,deg=2)
            #vel = (1/2)*poly[0]*(0.2*(1+mu[-2]))**2 + (3/4)*poly[1]*0.2*(1+mu[-2]) #+(3/2)*poly[2]
            vel = abs(-(poly[1]**2)/(4*poly[0])+poly[2])
            if i == len(types)-1:
                print("OUTLET \"VOLUME\" FLOW:", (2/3)*vel*0.4*(1+mu[-2]), "m**2/s")
        
        if rotate == 0:
            xi_new = points_i[:,0].max()
            yi_new = points_i[:,1].max()
            if typ == 4:
                yi_new = points_i[out,1].min()
                xi_new = points_i[out-1,0].min()
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            if typ == 3 or typ == 4:
                yi += yi_new
            xi += xi_new
                
        elif rotate == 1:
            temp = ux_i
            ux_i = -uy_i
            uy_i = temp
            xi_new = points_i[:,1].max()
            yi_new = points_i[:,0].max()
            if typ == 4:
                xi_new = points_i[out,1].min()
                yi_new = points_i[out-1,0].min()
            for j in range(len(points_i)):
                points_i[j] = rot_mat@points_i[j]
            for j in range(len(lin_points_i)):
                lin_points_i[j] = rot_mat@lin_points_i[j]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            if typ == 3 or typ == 4:
                xi -= xi_new
            yi += yi_new
            
        elif rotate == 2:
            ux_i = -ux_i
            uy_i = -uy_i
            yi_new = points_i[:,1].max()
            xi_new = points_i[:,0].max()
            if typ == 4:
                yi_new = points_i[out,1].min()
                xi_new = points_i[out-1,0].min()
            for j in range(len(points_i)):
                points_i[j] = rot_mat@rot_mat@points_i[j]
            for j in range(len(lin_points_i)):
                lin_points_i[j] = rot_mat@rot_mat@lin_points_i[j]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            if typ == 3 or typ == 4:
                yi -= yi_new
            xi -= xi_new

        elif rotate == 3:
            temp = ux_i
            ux_i = uy_i
            uy_i = -temp
            yi_new = points_i[:,0].max()
            xi_new = points_i[:,1].max()
            if typ == 4:
                yi_new = points_i[out-1,0].min()
                xi_new = points_i[out,1].min()
            for j in range(len(points_i)):
                points_i[j] = rot_mat@rot_mat@rot_mat@points_i[j]
            for j in range(len(lin_points_i)):
                lin_points_i[j] = rot_mat@rot_mat@rot_mat@lin_points_i[j]
            points_i[:,0] += xi
            points_i[:,1] += yi
            lin_points_i[:,0] += xi
            lin_points_i[:,1] += yi
            if typ == 3 or typ == 4:
                xi += xi_new
            yi -= yi_new  

        if i == 0:
            points = points_i
            lin_points = lin_points_i
        else:
            points = np.concatenate((points,points_i))
            lin_points = np.concatenate((lin_points,lin_points_i))
        if typ == 3:
            rotate += 1
        elif typ == 4:
            rotate -= 1

        ux = np.concatenate((ux,ux_i))
        uy = np.concatenate((uy,uy_i))
        if typ == 7:
            p += p_i.max()
        else:
            p += p_i[0]
        p = np.concatenate((p,p_i))
        i += 1

    print("TIME:", time.time()-start_time)

    p_tri = arch.plotHelp(lin_points,N-1,1.5,coord_mask = False)
    v_tri = arch.plotHelp(points,N,1.5,coord_mask = False)

    #figur2, x-hastighet
    arch.contourPlotter(ux,v_tri,title="x-velocity, $u_x$",fname="combo_figur1",HD = True)
    #figur3, y-hastighet
    arch.contourPlotter(uy,v_tri,title="y-velocity, $u_y$",fname="combo_figur2",HD = True)
    #figur4, hastighetsmagnitude
    arch.contourPlotter(np.sqrt(ux**2 + uy**2),v_tri,title="Velocity-magnitude, $|u|$",fname="combo_figur3",HD = True)
    #figur6, trykk
    arch.contourPlotter(p,p_tri,title="Pressure, p",fname="combo_figur4",HD = True)

    plt.close('all')