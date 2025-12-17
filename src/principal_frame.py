import numpy as np
from scipy.optimize import linear_sum_assignment
from .utilis import *

def get_inertia(coords:np.ndarray, masses:np.ndarray) -> np.ndarray:
    " Get the moment of inertia matrix from Mole object "

    coords = np.asarray(coords)
    masses = np.asarray(masses)
    com = get_com(coords, masses)
    coords_ = coords - com
    
    I = np.zeros((3, 3))       
    for r, m in zip(coords_, masses):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    
    return symmetrize_matrix(I)

def eig_blocks(eigvals, tol=1e-8):
    """
    Given sorted eigenvalues (1D array), return list of index lists (blocks)
    that group indices whose eigenvalues are equal within tol.
    """
    used = np.zeros(3, dtype=bool)
    blocks = []
    for i in range(3):
        if used[i]:
            continue
        same = [j for j in range(3) if abs(eigvals[j] - eigvals[i]) < tol]
        for j in same:
            used[j] = True
        blocks.append(same)
    return blocks

def get_principal_axes(coords:np.ndarray, masses:np.ndarray, old_axes:np.ndarray=None) -> tuple:
    " Get three principal axes from coordinates and masses based on previous principal axes "

    # diagonalize inertial matrix
    I = get_inertia(coords, masses)
    moments, axes = np.linalg.eigh(I)
    
    # sort desceding order so the smallest moment aligns the last axis
    # orientation of the electric field will be defined from the last column of Rotation matrix
    idx = np.argsort(-moments)
    moments = moments[idx] 
    axes = axes[:, idx]

    # if encouter improper rotation
    # then swap sign of an eigenvector and eigenvalue
    if np.linalg.det(axes) < 0:
        axes[:,2] *= -1

    if old_axes is not None:
        blocks = eig_blocks(moments)
        for block in blocks:
            if len(block) > 1:
            # Given previous axes, the principal axes are projection 
            # of previous principal axes onto degenerate subspace
                proj = axes[:, block] @ axes[:, block].T
                axes[:, block] = proj @ old_axes[:, block]

        # Raise if eigenvalues swap order
        O = np.abs(old_axes.T @ axes)
        row_ind, col_ind = linear_sum_assignment(-O)
        perm = col_ind[np.argsort(row_ind)]
        if not np.allclose(perm, range(3)):
            raise RuntimeError(f'Eigenvalues swapped, possible swap: {perm}')  

        # Re-Orthogonalization, maybe not necessary
        axes, _ = np.linalg.qr(axes)

        for i in range(3):
            if np.dot(old_axes[:, i], axes[:, i]) < 0:
                axes[:, i] *= -1

    return moments, axes

def dI_dR(coords:np.ndarray, masses:np.ndarray) -> np.ndarray:
    " Compute derivatives of inertial matrix w.r.t nuclear coordinates "
    N = len(masses)
    dI = np.zeros((N, 3, 3, 3)) # Natom * xyz * row * col
    coords = np.asarray(coords)
    masses = np.asarray(masses)
    coords -= get_com(coords, masses)

    for at in range(N):
        for xyz in range(3):
            for i in range(3):
                for j in range(3):
                    dI[at, xyz, i, j] = masses[at] * (
                        2 * (i == j) * coords[at, xyz] - \
                          (i == xyz) * coords[at, j] - \
                          (j == xyz) * coords[at, i]
                         )
            dI[at,xyz] = symmetrize_matrix(dI[at,xyz])
    return dI

def dU_dR(coords:np.ndarray, masses:np.ndarray, old_paxes:np.ndarray=None) -> np.ndarray:
    " Compute derivatives of eigenvectors of the principal axes w.r.t nuclear coordinates "

    N = len(masses)
    moments, axes = get_principal_axes(coords, masses, old_paxes)
    dI = dI_dR(coords, masses)
    dU = np.zeros_like(dI)
 
    for at in range(N):
        for xyz in range(3):
            for i in range(3):
                for j in range(3):
                    if j == i: continue
                    vi = axes[:,i]
                    vj = axes[:,j]
                    denom = moments[i]-moments[j]
                    if abs(denom) < 1e-8: 
                        continue
                    dU[at,xyz,:,i] += vj * (vj.T @ dI[at,xyz] @ vi) / denom
            
    return dU

