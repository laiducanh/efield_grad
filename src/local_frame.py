import numpy as np

def get_local_axes(A, B, C):
    ''' Local frame defined by coordinates of 3 atoms A, B, C '''
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    r1 = B - A
    r2 = C - A

    z = r1 / np.linalg.norm(r1)
    y = np.cross(r1, r2) / np.linalg.norm(np.cross(r1, r2))
    x = np.cross(y, z)
    
    # transformation matrix
    U = np.vstack((x, y, z)).T # columns are bases of local frame

    if np.linalg.det(U) < 0:
        x = -x
        U = np.vstack((x, y, z)).T
    
    return U, x, y, z

def d_vector(v:np.ndarray):
    ''' Compute derivative of normlalized vector v_hat = v / ||v|| w.r.t d_v '''

    v_hat = v / np.linalg.norm(v)
    return (np.identity(3) - np.outer(v_hat, v_hat)) / np.linalg.norm(v)


def d_cross(v1:np.ndarray, v2:np.ndarray):
    ''' Compute derivative of a cross product w = v1 x v2 w.r.t d_v1 and d_v2 '''
    dw_dv1 = np.array([
        [0, v2[2], -v2[1]],
        [-v2[2], 0, v2[0]],
        [v2[1], -v2[0], 0]
    ])
    dw_dv2 = np.array([
        [0, -v1[2], v1[1]],
        [v1[2], 0, -v1[0]],
        [-v1[1], v1[0], 0]
    ])
    return dw_dv1, dw_dv2

def dU_ABC(A, B, C):
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    U, x, y, z = get_local_axes(A, B, C)

    # d_v1 / d_R = d_v1/d_v * d_v/d_R
    # v1 = v / ||v||
    v = B-A
    dz_dRA = -d_vector(v)
    dz_dRB = d_vector(v)
    dz_dRC = np.zeros((3,3))

    # d_v2 / d_R = d_v2/d_v * d_v/d_R = d_v2/d_v * (d_v/d_r1 * d_r1/d_R + d_v/d_r2 * d_r2/d_R)
    # v2 = v / ||v||
    # v = r1 x r2
    r1 = B-A
    r2 = C-A
    v = np.cross(r1, r2)
    dv_dr1, dv_dr2 = d_cross(r1, r2)
    dy_dRA = d_vector(v) @ (dv_dr1 + dv_dr2) * (-1)
    dy_dRB = d_vector(v) @ dv_dr1
    dy_dRC = d_vector(v) @ dv_dr2

    # d_v3 / d_R = d_v3/d_v1 * d_v1/d_R + d_v3/d_v2 * d_v2/d_R
    # v3 = v1 x v2
    dx_dy, dx_dz = d_cross(y, z)
    dx_dRA = dx_dy @ dy_dRA + dx_dz @ dz_dRA
    dx_dRB = dx_dy @ dy_dRB + dx_dz @ dz_dRB
    dx_dRC = dx_dy @ dy_dRC + dx_dz @ dz_dRC

    dU_dRA = np.stack([dx_dRA.T, dy_dRA.T, dz_dRA.T], axis=2)
    dU_dRB = np.stack([dx_dRB.T, dy_dRB.T, dz_dRB.T], axis=2)
    dU_dRC = np.stack([dx_dRC.T, dy_dRC.T, dz_dRC.T], axis=2)

    return dU_dRA, dU_dRB, dU_dRC

def dU_dR(coords:np.ndarray, atoms:list) -> np.ndarray:
    N = coords.shape[0]
    dU_de = np.zeros((N, 3, 3, 3))
    dU_dRA, dU_dRB, dU_dRC = dU_ABC(*coords[atoms])
    dU_de[atoms[0], :, :, :] = dU_dRA
    dU_de[atoms[1], :, :, :] = dU_dRB
    dU_de[atoms[2], :, :, :] = dU_dRC

    return dU_de

