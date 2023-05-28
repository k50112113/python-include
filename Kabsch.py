import numpy as np

def get_transpose_axes(m):
    if len(m.shape) == 2:  return (1, 0)
    return (0, 2, 1)

# def kabsch(reference, target, ref_center = 0, tar_center = 0):
#     #reference  (N*dim)
#     #target     (N*dim)
#     #ref_center (dim)
#     #tar_center (dim)
#     P = target    - tar_center
#     Q = reference - ref_center
#     R = compute_rotation_matrix(P, Q)
#     return np.matmul(P, R.transpose(get_transpose_axes(R)))

# def compute_rotation_matrix(P, Q):
#     dim = P.shape[1]
#     H = np.matmul(np.transpose(P),Q)
#     U, S, Vh = np.linalg.svd(H)
#     V = np.transpose(Vh)
#     d = np.sign(np.linalg.det(np.matmul(V, np.transpose(U))))
#     I = np.identity(dim)
#     I[dim-1][dim-1] = d
#     R = np.matmul(V,np.matmul(I,np.transpose(U)))
#     return R

def kabsch(reference, target, ref_center = None, tar_center = None):
    #reference  (N*dim or B*N*dim)
    #target     (N*dim or B*N*dim)
    #ref_center (dim   or B*dim)
    #tar_center (dim   or B*dim)
    if tar_center is not None:
        P = target    - np.expand_dims(tar_center, -2)
    else:
        P = target
    if ref_center is not None:
        Q = reference - np.expand_dims(ref_center, -2)
    else:
        Q = reference
    R = compute_rotation_matrix(P, Q)
    return np.matmul(P, R.transpose(get_transpose_axes(R)))

def compute_rotation_matrix(P, Q):
    #P (N*dim or B*N*dim)
    #Q (N*dim or B*N*dim)
    #R (N*dim or B*N*dim)
    if len(P.shape) == 3 and len(Q.shape) == 3 and len(P) != len(Q):
        print("Batch size must be the same")
        exit()
    dim = P.shape[-1]
    Paxes = get_transpose_axes(P)
    H = np.matmul(P.transpose(Paxes),Q)
    U, S, Vh = np.linalg.svd(H)
    Uaxes = get_transpose_axes(U)
    V = Vh.transpose(get_transpose_axes(Vh))
    d = np.sign(np.linalg.det(np.matmul(V, U.transpose(Uaxes))))
    if max(len(P.shape), len(Q.shape)) == 2:
        I = np.identity(dim)
        I[dim-1, dim-1] = d
    else:
        if len(P.shape) == 3: l = len(P)
        else: l = len(Q)
        I = np.tile(np.identity(dim), (l, 1, 1))
        I[:, dim-1, dim-1] = d
    R = np.matmul(V, np.matmul(I, U.transpose(Uaxes)))
    return R