import numpy as np

def minimum_image(r,lbox):
    #wrap vector or wrap -L/2~L/2 box
    return r - lbox*np.round(r / lbox)
    #return (r + lboxh) % lbox - lboxh

def wrap(r, lbox):
    #wrap 0~L box
    return r%lbox

def dis(v):
    return np.sum(v**2)**0.5

def normalize(v):
    return v/dis(v)
    
def trans(T,v):
    return np.array(np.matmul(T,v))

def get_trans_matrix(vref1,vref2):
    vref1=normalize(vref1)
    vref2=normalize(vref2)
    vref3=np.cross(vref1,vref2)
    vref3=normalize(vref3)
    vref2=np.cross(vref3,vref1)
    vref2=normalize(vref2)
    return np.matrix([vref1,vref2,vref3])

def get_inverse_matrix(T):
    return np.linalg.inv(T)

def displacement_table(coord, lbox):
    N = coord.shape[0]
    r = np.zeros((N,N,3))
    for i in range(N-1):
        for j in range(i+1,N):
            tmp_r = coord[i] - coord[j]
            r[i][j] =  tmp_r
            r[j][i] = -tmp_r        
    return minimum_image(r,lbox)
    
def distance_table(coord, lbox, allpair=True, atom1_list=[], atom2_list=[], cutoff=-1):
    N = coord.shape[0]
    if allpair == True:
        atom1_list = range(N-1)
        atom2_list = range(i+1,N)
        r = np.zeros((N,N))-1
    else:
        if len(atom1_list) == 0 or len(atom2_list) == 0:
            print("atom list not found.")
            return 0   
    r = np.zeros((N,N))-1
    for i in atom1_list:
        for j in atom2_list:
            tmp_r = dis(minimum_image(coord[i]-coord[j],lbox))
            if tmp_r < cutoff or cutoff == -1:
                r[i][j] = tmp_r
                r[j][i] = tmp_r
    return r
    