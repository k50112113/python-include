import numpy as np

def kabsch(reference, target, ref_center, tar_center):
    #reference, target (N x dim)
    P = target-tar_center
    Q = reference-ref_center
    R = compute_rotation_matrix(P, Q)
    return np.transpose(np.matmul(R,np.transpose(P)))

def compute_rotation_matrix(P, Q):
    dim = P.shape[1]
    H = np.matmul(np.transpose(P),Q)
    U,S,Vh = np.linalg.svd(H)
    V = np.transpose(Vh)
    d = np.sign(np.linalg.det(np.matmul(V, np.transpose(U))))
    I = np.identity(dim)
    I[dim-1][dim-1] = d
    R = np.matmul(V,np.matmul(I,np.transpose(U)))
    return R

