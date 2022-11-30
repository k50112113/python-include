import numpy as np
from LocalFrameTransform import *

def read_descriptor(filename,feature=[],Start=0,Nmax=-1,ignore_first=1):
    descriptor = []

    with open(filename,"r") as fin:
        line = 0
        for aline in fin:
            if "Frame" not in aline:
                if line >= Start:
                    linelist = aline.strip().split()
                    if len(feature)>0:
                        descriptor.append([float(linelist[i]) for i in feature])
                    else:
                        descriptor.append([float(i) for i in linelist[ignore_first:]])
                        
                    if len(descriptor) == Nmax and Nmax > -1:
                        break     
                line += 1
    return np.array(descriptor,dtype='float32')


def make_xyz(coord,type,filename):
    with open(filename,"w") as fout:
        for i in range(coord.shape[0]):
            fout.write('''     %d
 i =        %d, time =        %.3f, E =        0
'''%(coord.shape[1],i,i))
            for j in range(coord.shape[1]):
                fout.write("%s "%(type[j]))
                fout.write("%f %f %f\n"%(coord[i][j][0],coord[i][j][1],coord[i][j][2]))
            
def descriptor_to_cycloalkane_all(descriptor,filename="",transpose=False):

    copy_of_descriptor = np.copy(descriptor)
    Num = copy_of_descriptor.shape[0]
    coord = []
    for i in range(0,copy_of_descriptor.shape[1],3):
        coord.append(copy_of_descriptor[:,i:i+3])
    coord = np.array(coord)
    type = ['C','H','H']*(len(coord)//3)
    if filename != "":
        make_xyz(np.transpose(coord,axes=(1,0,2)),type,filename) 
    if transpose == True:
        return np.transpose(coord,axes=(1,0,2))
    return coord
def descriptor_to_cycloalkane_bb(descriptor,filename="",transpose=False):

    copy_of_descriptor = np.copy(descriptor)
    Num = copy_of_descriptor.shape[0]
    coord = []
    coord.append(np.vstack((copy_of_descriptor[:,0],np.zeros(Num),np.zeros(Num))).transpose())
    coord.append(np.zeros((Num,3)))
    coord.append(np.vstack((copy_of_descriptor[:,1:3].transpose(),np.zeros(Num))).transpose())
    for i in range(3,copy_of_descriptor.shape[1],3):
        coord.append(copy_of_descriptor[:,i:i+3])
    coord = np.array(coord)
    type = ['C']*len(coord)
    if filename != "":
        make_xyz(np.transpose(coord,axes=(1,0,2)),type,filename) 
    if transpose == True:
        return np.transpose(coord,axes=(1,0,2))
    return coord    
###############################################################################
#alkane all (N x 3+3(n-3)+3(2n+2)): C1(x) C3(x,y) 7*H(x,y,z) C4(x,y,z) 2*H(x,y,z) C5(x,y,z) 2*H(x,y,z)... + H(x,y,z)
'''
  
   \    \ /   \ /
  -C1    C3    C5
  /  \  /  \  /  ...
      C2    C4
     / \   / \
    
    C2 at (0,0,0)
    C1 at x-axis
    C3 at xy-plane
'''
def descriptor_to_alkane_all(descriptor,n,filename="",one_local_frame=True,transpose=False):
    
    copy_of_descriptor = np.copy(descriptor)
    Num = copy_of_descriptor.shape[0]
    coord = []
    coord_H = []
    coord.append(np.vstack((copy_of_descriptor[:,0],np.zeros(Num),np.zeros(Num))).transpose())
    coord.append(np.zeros((Num,3)))
    coord.append(np.vstack((copy_of_descriptor[:,1:3].transpose(),np.zeros(Num))).transpose())
    h_index = -1
    dscp_index = 3
    for i in range(7):
        coord_H.append(copy_of_descriptor[:,dscp_index:dscp_index+3])
        h_index += 1
        dscp_index += 3
    coord.append(copy_of_descriptor[:,dscp_index:dscp_index+3])
    dscp_index += 3
    for i in range(2):
        coord_H.append(copy_of_descriptor[:,dscp_index:dscp_index+3])
        h_index += 1
        dscp_index += 3
    
    c_index = 3    
    for i in range(5,n+1): 
        coord.append(copy_of_descriptor[:,dscp_index:dscp_index+3])
        coord_H.append(copy_of_descriptor[:,dscp_index+3:dscp_index+6])
        coord_H.append(copy_of_descriptor[:,dscp_index+6:dscp_index+9])
        c_index += 1
        h_index += 2
        dscp_index += 9   
        if one_local_frame == False:
            Vx = coord[c_index-3]-coord[c_index-2]
            Vy = coord[c_index-1]-coord[c_index-2]
            Vo = coord[c_index-2]
            for j in range(Num):
                T = get_trans_matrix(Vx[j],Vy[j])
                coord[c_index][j] = trans(T.transpose(),coord[c_index][j])[0]+Vo[j]
                coord_H[h_index-1][j] = trans(T.transpose(),coord_H[h_index-1][j])[0]+Vo[j]
                coord_H[h_index][j] = trans(T.transpose(),coord_H[h_index][j])[0]+Vo[j]
    
    coord_H.append(copy_of_descriptor[:,dscp_index:dscp_index+3])    
    h_index += 1
    if n > 4 and one_local_frame == False:
        for j in range(Num):
            T = get_trans_matrix(Vx[j],Vy[j])
            coord_H[h_index][j] = trans(T.transpose(),coord_H[h_index][j])[0]+Vo[j]
    
    type = ['C']*len(coord)+['H']*len(coord_H)
    coord += coord_H
    coord = np.array(coord)
    if filename != "":
        make_xyz(np.transpose(coord,axes=(1,0,2)),type,filename)
    if transpose == True:
        return np.transpose(coord,axes=(1,0,2))
    return coord
#alkane bb (N x 3+3(n-3)): C1(x) C3(x,y) C4(x,y,z) C5(x,y,z)...
'''
   C1    C3    C5
     \  /  \  /  ...
      C2    C4
     
    C2 at (0,0,0)
    C1 at x-axis
    C3 at xy-plane
'''
def descriptor_to_alkane_bb(descriptor,n,filename="",one_local_frame=True,transpose=False):
    
    copy_of_descriptor = np.copy(descriptor)
    Num = copy_of_descriptor.shape[0]
    coord = []
    coord.append(np.vstack((copy_of_descriptor[:,0],np.zeros(Num),np.zeros(Num))).transpose())
    coord.append(np.zeros((Num,3)))
    coord.append(np.vstack((copy_of_descriptor[:,1:3].transpose(),np.zeros(Num))).transpose())
    coord.append(copy_of_descriptor[:,3:6])
    c_index = 3
    dscp_index = 6
    for i in range(5,n+1): 
        coord.append(copy_of_descriptor[:,dscp_index:dscp_index+3])
        c_index += 1
        dscp_index += 3
        if one_local_frame == False:
            Vx = coord[c_index-3]-coord[c_index-2]
            Vy = coord[c_index-1]-coord[c_index-2]
            Vo = coord[c_index-2]
            for j in range(Num):
                T = get_trans_matrix(Vx[j],Vy[j])
                coord[c_index][j] = trans(T.transpose(),coord[c_index][j])[0]+Vo[j]
        
        
    type = ['C']*len(coord)
    coord = np.array(coord)
    if filename != "":
        make_xyz(np.transpose(coord,axes=(1,0,2)),type,filename) 
    if transpose == True:
        return np.transpose(coord,axes=(1,0,2))
    return coord