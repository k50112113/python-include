import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MakeAnimation as MA
import math


##########################################################
"""
Plotting
"""
##########################################################

def reverse_colormap(cmap):
    reverse = []
    k = []   
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
    LinearL = dict(zip(k,reverse))
    cmap_r = mpl.colors.LinearSegmentedColormap('reverse_coolwarm', LinearL)
    return cmap_r
    
def setupFigure(figsize=(10,8), axsize=.7, dim=2, elev=13., azim=0, setlabeltick=True):
    # Modidfy matplotlibrc
    
    fig = plt.figure(figsize=figsize)
    if dim == 3:
        ax = fig.add_axes([.5*(1-axsize), .5*(1-axsize), axsize, axsize], projection='3d')
        ax.clear()
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis="x",which ="major",length=9,width=2,labelsize=20, pad=10)
        ax.tick_params(axis="y",which ="major",length=9,width=2,labelsize=20, pad=10)
        ax.tick_params(axis="x",which ="minor",length=6,width=2,labelsize=20, pad=10)
        ax.tick_params(axis="y",which ="minor",length=6,width=2,labelsize=20, pad=10)
        ax.tick_params(axis="z",which ="major",length=9,width=2,labelsize=20, pad=10)
        ax.tick_params(axis="z",which ="minor",length=6,width=2,labelsize=20, pad=10)
        if setlabeltick == True:
            ax.set_xlabel("$CV_1$",fontsize=26,labelpad=24)
            ax.set_ylabel("$CV_2$",fontsize=26,labelpad=24)
            ax.set_zlabel("$CV_3$",fontsize=26,labelpad=24)
            ax.set_xlim3d([-1.1, 1.1])
            ax.set_ylim3d([-1.1, 1.1])
            ax.set_zlim3d([-1.1, 1.1])
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
            ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
            ax.set_zticks([-1.0,-0.5,0.0,0.5,1.0])
    elif dim == 2:
        ax = fig.add_axes([.7*(1-axsize), .5*(1-axsize), axsize, axsize])
        ax.tick_params(axis="x",which ="major",length=9,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="y",which ="major",length=9,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="x",which ="minor",length=6,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="y",which ="minor",length=6,width=2,labelsize=24, pad=10)
        if setlabeltick == True:
            ax.set_xlabel("$CV_1$",fontsize=26)
            ax.set_ylabel("$CV_2$",fontsize=26)
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
            ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both",direction="in",which ="both",top=True,right=True)
    
    return fig, ax, plt

def plot(CV,colorcode=None,vmin=-1,vmax=-1,cmap=None,\
         filename="",filetype="", show=True, title="", colorbar_title="",\
         reversecmap=False, elev=13., azim=0, cont=False, significant_digits=1,\
         norm=None):
    dim = CV.shape[1]
    if cmap and reversecmap:
        cmap = reverse_colormap(cmap)
    
    if dim == 3:
        if filetype == "mp4":
            figsize=(13,12)
            axsize = .9
        elif str(colorcode) == "None":
            figsize=(12,12)
            axsize = .9
        else:
            axsize = .8
            figsize=(12,12)
    else:
        axsize = 0.7  
        if str(colorcode) != "None":
            figsize=(10,8)
        else:
            figsize=(8,8) 
            
    fig, ax, plt = setupFigure(figsize=figsize,axsize=axsize,dim=dim, elev=elev, azim=azim)
    title_ = ax.set_title(title, fontsize=40)
    
    if str(colorcode) != "None" and vmin == -1 and vmin == -1:
        import Utility as Ut
        vmin, vmax = Ut.get_best_colorbar_range(colorcode.min(),colorcode.max(),significant_digits=1)
    yticklabels = [vmin + i/4*(vmax-vmin) for i in range(5)]
    if cont == False:
        if dim == 3:
            direction = np.array([np.cos(azim/180*np.pi),np.sin(azim/180*np.pi),0.0])
            zorder = np.inner(CV,direction)
            arg = np.argsort(zorder)
            if str(colorcode) != "None":
                im = ax.scatter(CV[arg][:,0],CV[arg][:,1],CV[arg][:,2],s=1,c=colorcode[arg],cmap=cmap, vmin=vmin, vmax=vmax)
                if filetype != "mp4":
                    cb = fig.colorbar(im,pad=0.01,shrink=0.55,ticks=yticklabels)
                    cb.ax.tick_params(labelsize=20)
                    if colorbar_title != "":
                        cb.set_label(colorbar_title)
            else:
                ax.scatter(CV[arg][:,0],CV[arg][:,1],CV[arg][:,2],s=1,c='k')
            
            if filename != "":
                if filetype == "":
                    plt.savefig(filename)
                elif filetype == "both":
                    plt.savefig(filename+".pdf")
                    plt.savefig(filename+".png")
                elif filetype == "mp4":
                    MA.make_animation_rotate(fig,ax,filename,CV,colorcode,cmap,vmin,vmax,elev=elev)  
                else:
                    plt.savefig(filename+"."+filetype)
            elif show == True:
                plt.show()
            else:
                return fig, ax, im, title_
        else:
            im = ax.scatter(CV[:,0],CV[:,1],s=1,c=colorcode,cmap=cmap,vmin=vmin,vmax=vmax)
            if str(colorcode) != "None":
                cb = fig.colorbar(im,ticks=yticklabels)
                cb.ax.tick_params(labelsize=20)
                if colorbar_title != "":
                    cb.set_label(colorbar_title)
            else:
                im.set_color(c='k')
            #plt.tight_layout()
            if filename != "":
                if filetype == "":
                    plt.savefig(filename)
                elif filetype == "both":
                    plt.savefig(filename+".pdf")
                    plt.savefig(filename+".png")
                else:
                    plt.savefig(filename+"."+filetype)
            elif show == True:
                plt.show()
            else:
                return fig, ax, plt
    else:
        nbin = round(CV.shape[0]**(1/dim))
        if dim == 2:
            
            cv1 = CV[:,0].reshape(nbin,nbin)
            cv2 = CV[:,1].reshape(nbin,nbin)
            colorcode = colorcode.reshape(nbin,nbin)
            
            surf = ax.contourf(cv1,cv2,colorcode, vmin=vmin, vmax=vmax, cmap=cmap, levels=500, extend='both', norm=norm)
            for c in surf.collections:
                c.set_edgecolor("face")
            cb = fig.colorbar(surf,ax=ax,pad=0.08,ticks=yticklabels,extend='both')
            cb.ax.tick_params(labelsize=20)
            if colorbar_title != "":
                cb.set_label(colorbar_title, fontsize=20)
            
            if filename != "":
                if filetype == "":
                    plt.savefig(filename)
                elif filetype == "both":
                    plt.savefig(filename+".pdf")
                    plt.savefig(filename+".png")
                else:
                    plt.savefig(filename+"."+filetype)
            elif show == True:
                plt.show()
            else:
                return fig, ax, plt
        elif dim == 3:

            cv1 = CV[:,0].reshape(nbin,nbin,nbin)
            cv2 = CV[:,1].reshape(nbin,nbin,nbin)
            cv3 = CV[:,2].reshape(nbin,nbin,nbin)
            colorcode_flat = np.exp(colorcode)
            colorcode = colorcode.reshape(nbin,nbin,nbin)
            min_tmp = colorcode_flat.min()
            for index in range(len(colorcode_flat)):
                if colorcode_flat[index] == min_tmp:
                    colorcode_flat[index] = 0

            iso_prob = 0.98
            spacing = CV[:,2][1]-CV[:,2][0]
            left = colorcode_flat.min()
            right = colorcode_flat.max()
            
            mid = (left+right)/2
            sum_tmp = np.sum(colorcode_flat[np.where(colorcode_flat>mid)[0]])*spacing**3
            itrmax = 500
            itr = 0
            while abs(sum_tmp - iso_prob)/iso_prob > 1E-4 and itr < itrmax:
                itr += 1
                if sum_tmp > iso_prob:
                    left = mid
                else:
                    right = mid
                mid = (left+right)/2
                sum_tmp = np.sum(colorcode_flat[np.where(colorcode_flat>mid)[0]])*spacing**3
            iso_val=np.log(mid)
            
            from skimage import measure
            # verts, faces, norm, val = measure.marching_cubes_lewiner(volume = colorcode, level= iso_val, spacing=(spacing, spacing, spacing)) #deprecated
            verts, faces, norm, val = measure.marching_cubes(volume = colorcode, level= iso_val, spacing=(spacing, spacing, spacing))
            surf = ax.plot_trisurf(verts[:, 0]-1, verts[:,1]-1, faces, verts[:, 2]-1,color='k', lw=0,alpha=0.2)
            
            if filename != "":
                if filetype == "":
                    plt.savefig(filename)
                elif filetype == "both":
                    plt.savefig(filename+".pdf")
                    plt.savefig(filename+".png")
                elif filetype == "mp4":
                    MA.make_animation_rotate_cont(fig,ax,filename,elev=elev)  
                else:
                    plt.savefig(filename+"."+filetype)
            elif show == True:
                plt.show()
            else:
                return fig, ax, plt
               
            

def read_CV(filename,feature=[],Nmax=-1):              
    CV = []
    with open(filename,"r") as fin:
        for aline in fin:
            linelist = aline.strip().split()
            
            if len(feature):
                CV.append([float(linelist[i]) for i in feature])
            else:
                CV.append([float(i) for i in linelist])
                
            if len(CV) == Nmax and Nmax > -1:
                break     
    return np.array(CV)

def read_hist(filename):
    data = []
    with open(filename,"r") as fin:
        for aline in fin:
            linelist = aline.strip().split()
            data.append([float(i) for i in linelist])
    data = np.array(data).transpose()
    dim = data.shape[0] - 1
    nbin = round(data.shape[1]**(1/dim))
    return data, dim, nbin #data[0~n-1] axes, data[n] hist

##########################################################
"""

"""
##########################################################

def get_cont_cv_space(nbin,dim=2,lowerbond=-1,upperbond=1):
    #CV (nbin^dim x dim)
    cvaxis = np.linspace(lowerbond,upperbond,nbin)
    if dim == 2:
        cvmesh = np.meshgrid(cvaxis,cvaxis, indexing='ij')
    elif dim == 3:
        cvmesh = np.meshgrid(cvaxis,cvaxis,cvaxis, indexing='ij')
    CV = cvmesh[0].flatten()
    for a_cvmesh in cvmesh[1:]:
        CV = np.vstack((CV,a_cvmesh.flatten()))
    CV = CV.transpose()
    return CV
    
##########################################################
"""
Color coding
"""
##########################################################

from queue import PriorityQueue
import LocalFrameTransform as LFT
import Descriptor as DSPTR

def get_mpl():
    return mpl
###################################################################
def innerproduct_3D(V1,V2):
    V1len = np.linalg.norm(V1,axis=1)
    V2len = np.linalg.norm(V2,axis=1)
    dot = V1[:,0]*V2[:,0] + V1[:,1]*V2[:,1] + V1[:,2]*V2[:,2]
    return dot/V1len/V2len
    
def compute_dihedral(A,B,C,D):
    #trans state = 180 deg
    Vx = A-B
    Vy = C-B
    N1 = np.cross(Vx,Vy)
    Vx = B-C
    Vy = D-C
    N2 = np.cross(Vx,Vy)
    dihedral = np.arccos(innerproduct_3D(N1,N2))*180/np.pi
    return dihedral

def compute_bond_angle(A,B,C):
    Vx = A-B
    Vy = C-B
    bond_angle = np.arccos(innerproduct_3D(Vx,Vy))*180/np.pi
    return bond_angle
    
###############################################################################
#descriptor (N x 4): x1, y1, x2, y2...
def double_pendulum_angle(descriptor,theta_index=1):
    x1 = descriptor[:,0]
    y1 = descriptor[:,1]
    if theta_index == 1:
        theta = np.angle(-y1+x1*1j,deg=True)
        return theta,-180,180,mpl.cm.seismic
    elif theta_index == 2:
        x2p = descriptor[:,2] - x1
        y2p = descriptor[:,3] - y1
        theta = np.angle(-y2p+x2p*1j,deg=True)
        return theta,-180,180,mpl.cm.seismic
def double_pendulum_potential(descriptor):
    y1 = descriptor[:,1]
    y2 = descriptor[:,3]
    return y1+y2,-1,1,mpl.cm.jet    
def double_pendulum_time(descriptor):
    return np.arange(descriptor.shape[0]),-1,1,mpl.cm.jet        
###############################################################################
#descriptor (N x 12): A1_x, A1_y, A2_x, A2_Y...
'''
    5   4
    A   A
 6 A  A  A 3
    A   A
    1   2
'''
def hex2d_distance(descriptor,id):
    x = descriptor[:,(id-1)*2]
    y = descriptor[:,(id-1)*2+1]
    return (x**2+y**2)**0.5,-1,-1,mpl.cm.jet

###############################################################################
def alkane_all_dihedral(descriptor,n,phi_index=1,one_local_frame=True):
    #phi_index: 1 = 1-2-3-4, 2 = 2-3-4-5...
    coord = DSPTR.descriptor_to_alkane_all(descriptor,n=n,one_local_frame=one_local_frame)
    dih = compute_dihedral(coord[phi_index-1],coord[phi_index],coord[phi_index+1],coord[phi_index+2])
    return dih,0,180,mpl.cm.jet
def alkane_all_bond_angle(descriptor,n,theta_index=1,one_local_frame=True):
    #phi_index: 1 = 1-2-3, 2 = 2-3-4...
    # Equilibrium Angle = 112.7 deg
    coord = DSPTR.descriptor_to_alkane_all(descriptor,n,one_local_frame=one_local_frame)
    ang = compute_bond_angle(coord[theta_index-1],coord[theta_index],coord[theta_index+1])
    return ang,112.7-30,112.7+30,mpl.cm.seismic
def alkane_all_ee(descriptor,n,one_local_frame=True):
    #phi_index: 1 = 1-2-3-4, 2 = 2-3-4-5...
    coord = DSPTR.descriptor_to_alkane_all(descriptor,n=n,one_local_frame=one_local_frame)
    V = coord[n-1]-coord[0]
    V = np.linalg.norm(V,axis=1)    
    return V,-1,-1,mpl.cm.jet    
def alkane_all_CH1_dihedral(descriptor,n,one_local_frame=True):
    #n    0    1    2
    #H    C    C    C
    coord = DSPTR.descriptor_to_alkane_all(descriptor,n=n,one_local_frame=one_local_frame)
    dih = compute_dihedral(coord[n],coord[0],coord[1],coord[2])
    return dih,0,180,mpl.cm.jet
def alkane_all_CHn_dihedral(descriptor,n,one_local_frame=True):
    #n-3  n-2  n-1  3n-1
    #C    C    C    H
    N123 = np.array([0,0,1])
    coord = DSPTR.descriptor_to_alkane_all(descriptor,n=n,one_local_frame=one_local_frame)
    dih = compute_dihedral(coord[n-3],coord[n-2],coord[n-1],coord[3*n+1])
    return dih,0,180,mpl.cm.jet
###############################################################################
def alkane_bb_dihedral(descriptor,n,phi_index=1,one_local_frame=True):
    #phi_index: 1 = 1-2-3-4, 2 = 2-3-4-5...
    coord = DSPTR.descriptor_to_alkane_bb(descriptor,n,one_local_frame=one_local_frame)
    dih = compute_dihedral(coord[phi_index-1],coord[phi_index],coord[phi_index+1],coord[phi_index+2])
    return dih,0,180,mpl.cm.jet  
def alkane_bb_bond_angle(descriptor,n,theta_index=1,one_local_frame=True):
    #phi_index: 1 = 1-2-3, 2 = 2-3-4...
    # Equilibrium Angle = 112.7 deg
    coord = DSPTR.descriptor_to_alkane_bb(descriptor,n,one_local_frame=one_local_frame)
    ang = compute_bond_angle(coord[theta_index-1],coord[theta_index],coord[theta_index+1])
    return ang,112.7-30,112.7+30,mpl.cm.seismic   
def alkane_bb_ee(descriptor,n,one_local_frame=True):
    #phi_index: 1 = 1-2-3-4, 2 = 2-3-4-5...
    coord = DSPTR.descriptor_to_alkane_bb(descriptor,n=n,one_local_frame=one_local_frame)
    V = coord[n-1]-coord[0]
    V = np.linalg.norm(V,axis=1)    
    return V,-1,-1,mpl.cm.jet   
###############################################################################
def cycloalkane_bb_dihedral(descriptor,n,phi_index=1):
    #phi_index: 1 = 1-2-3-4, 
    #           2 = 2-3-4-5,
    #           3 = 3-4-5-6,
    #           4 = 4-5-6-1,
    #           5 = 5-6-1-2,
    #           6 = 6-1-2-3
    p1234 = np.array([phi_index-1,phi_index,phi_index+1,phi_index+2])%n
    coord = DSPTR.descriptor_to_cycloalkane_bb(descriptor)
    dih = compute_dihedral(coord[p1234[0]],coord[p1234[1]],coord[p1234[2]],coord[p1234[3]])
    return dih,0,180,mpl.cm.jet
    
    
###############################################################################
#descriptor (N x 36): site A nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
#                     site B nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
#                     site C nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
#                     site D nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
#
'''
  -> +z

  A   B
  O   O

    O
+x / \
  H   H
  
O       O
D       C
'''
def h2o_Bz(descriptor):
    Bz = descriptor[:,11]
    return Bz,-5,5,mpl.cm.seismic
def h2o_Az(descriptor):
    Az = descriptor[:,2]
    return Az,-5,5,mpl.cm.seismic
    
def h2o_dihedral_HOH_DC(descriptor):
    dih = []
    for adata in descriptor:
        OC = np.array(adata[18:21])
        OD = np.array(adata[27:30])
        NDC = -np.cross(OC,OD)
        NHOH = np.array([0,0,1])
        dih.append((np.inner(NHOH,NDC)/np.linalg.norm(NHOH)/np.linalg.norm(NDC)))    
    dih = np.arccos(np.array(dih))*180/np.pi
    return dih,0,60,mpl.cm.seismic

def h2o_dihedral_HOH_AB(descriptor):
    dih = []
    for adata in descriptor:
        OA = np.array(adata[0:3])
        OB = np.array(adata[9:12])
        NAB = np.cross(OA,OB)
        NHOH = np.array([0,0,1])
        dih.append((np.inner(NAB,NHOH)/np.linalg.norm(NAB)/np.linalg.norm(NHOH)))
    dih = np.arccos(np.array(dih))*180/np.pi
    return dih,0,180,mpl.cm.seismic

def h2o_dihedral_AB_CD(descriptor):
    dih = []
    for adata in descriptor:
        OA = np.array(adata[0:3])
        OB = np.array(adata[9:12])
        OC = np.array(adata[18:21])
        OD = np.array(adata[27:30])
        NAB = np.cross(OA,OB)
        NCD = np.cross(OC,OD)
        dih.append(abs(np.inner(NAB,NCD)/np.linalg.norm(NAB)/np.linalg.norm(NCD)))
    dih = np.arccos(np.array(dih))*180/np.pi
    return dih,0,90,mpl.cm.jet
    
def h2o_length_OA_OB(descriptor):
    dis = []
    for adata in descriptor:
        OA = np.array(adata[0:3])
        OB = np.array(adata[9:12])
        OC = np.array(adata[18:21])
        OD = np.array(adata[27:30])
        l = np.linalg.norm(OA-OB)
        dis.append(l)
    return np.array(dis),-1,-1,mpl.cm.jet
###############################################################################    
def h2o_permutation_4O(descriptor):
    #descriptor (N x 54): 1st nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    #                     2nd nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    #                     ...
    #                     6th nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    
    
    cos_30deg = np.cos(30*np.pi/180)
    bond_length = 0.9572
    host_O = np.array([0,0,0])
    host_H1 = np.array([bond_length,0,0])
    host_H2 = np.array([bond_length*np.cos(104.52*np.pi/180),bond_length*np.sin(104.52*np.pi/180),0])
    
    perm = []
    for adata in descriptor:
        nn_O = []
        nn_H = []
        for i in range(0,len(adata),9):
            nn_O.append(adata[i:i+3])
            nn_H.append([adata[i+3:i+6],adata[i+6:i+9]])
        nn_O = np.array(nn_O)
        nn_H = np.array(nn_H)
    
        O_x_max = PriorityQueue()
        O_y_max = PriorityQueue()
        O_z_max = PriorityQueue()
        O_z_min = PriorityQueue()
        
        cos1_max = PriorityQueue()
        cos2_max = PriorityQueue()
        for i in range(4):
            
            v = nn_O[i] - host_O
            oh1 = host_H1 - host_O
            oh2 = host_H2 - host_O
            r = np.sum(v**2)**0.5
            
            cos1 = np.inner(oh1,v)/(bond_length*r)
            cos2 = np.inner(oh2,v)/(bond_length*r)
            
            cos1_max.put((-cos1,i))
            cos2_max.put((-cos2,i))
            
        
        alist = [0,1,2,3]
        a, b = cos1_max.get()[1], cos2_max.get()[1]
        if a != b:
            alist.remove(a)
            alist.remove(b)
            if nn_O[alist[0]][2] > nn_O[alist[1]][2]:
                c = alist[0]
                d = alist[1]
            else:
                c = alist[1]
                d = alist[0]
            perm.append((a+1)*1000+(b+1)*100+(c+1)*10+(d+1))
        else:
            perm.append(0)
    typelist, typecount = np.unique(perm, return_counts=True)
    for i in range(len(perm)):
        index = np.where(typelist==perm[i])[0][0]
        perm[i] = index
    return perm,typelist

def h2o_csi(descriptor):
    #descriptor (N x 54): 1st nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    #                     2nd nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    #                     ...
    #                     6th nn O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z,
    #i = host
    #j = neighbor
    #dji = distance between the host and the nearest non H-bonded neighbor
    #dj'i = distance between the host and the furthest H-bonded neighbor
    #csi = dji - dj'i
    #Shi and Tanaka, PNAS (2018)
    
    cos_30deg = np.cos(30*np.pi/180)
    bond_length = 0.9572
    host_O = np.array([0,0,0])
    host_H1 = np.array([bond_length,0,0])
    host_H2 = np.array([bond_length*np.cos(104.52*np.pi/180),bond_length*np.sin(104.52*np.pi/180),0])
    
    csi = []
    for adata in descriptor:
        nn_O = []
        nn_H = []
        for i in range(0,len(adata),9):
            nn_O.append(adata[i:i+3])
            nn_H.append([adata[i+3:i+6],adata[i+6:i+9]])
        nn_O = np.array(nn_O)
        nn_H = np.array(nn_H)
    
        hbond = PriorityQueue()
        nhbond = PriorityQueue()
        for i in range(len(nn_O)):
            v = nn_O[i] - host_O
            oh1 = host_H1 - host_O
            oh2 = host_H2 - host_O
            r = np.sum(v**2)**0.5
            
            #O-H ... O (O-H belongs to host)
            cos1 = np.inner(oh1,v)/(bond_length*r)
            cos2 = np.inner(oh2,v)/(bond_length*r)
            
            #O-H ... O (O-H belongs to neighbor)
            oh1 = nn_H[i][0] - nn_O[i]
            oh2 = nn_H[i][1] - nn_O[i]
            cos3 = np.inner(oh1,-v)/(bond_length*r)
            cos4 = np.inner(oh2,-v)/(bond_length*r)
            
            if r <= 3.5 and (cos1 > cos_30deg or cos2 > cos_30deg or cos3 > cos_30deg or cos4 > cos_30deg):
                hbond.put((-r,i))
            else:
                nhbond.put((r,i))
                
        if nhbond.empty() or hbond.empty():
            csi.append(-3)
        else:
            b=nhbond.get()
            a=hbond.get()
            csi.append(b[0]+a[0])
    csi = np.array(csi)
    return csi
###############################################################################
def single_h2o_internal(descriptor,mode):
    #descriptor (N x 3): H1_x, H2_x, H2_y
    #mode: 1 = H-O-H bond angle, 2 = OH bond sum, 3 = OH bond difference
    colorcode = np.zeros(len(descriptor))
    for i in range(len(colorcode)):
        v1 = np.array([descriptor[i][0],0,0])
        v2 = np.array([descriptor[i][1],descriptor[i][2],0])
        if mode == 1:
            colorcode[i] = np.inner(v1,v2)
        elif mode == 2:
            colorcode[i] = np.sum(v1**2)**0.5+np.sum(v2**2)**0.5
        elif mode == 3 :
            colorcode[i] = np.sum(v1**2)**0.5-np.sum(v2**2)**0.5
    
    if mode == 1:
        colorcode = np.arccos(colorcode)*180/np.pi
    
    return colorcode

def single_h2o_dynamic(descriptor,nevery):
    #descriptor (N x 9): O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z
    colorcode = np.zeros(len(descriptor))
    for i in range(len(colorcode)):
        colorcode[i] = int(float(i)/nevery)
    return colorcode
