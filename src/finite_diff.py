from pyscf import gto
import numpy as np
from .principal_frame import get_principal_axes
from .local_frame import get_local_axes
from .utilis import process_efield

def dipint_deriv_fd(mol:gto.Mole):
    natm = mol.natm
    nao = mol.nao_nr()
    delta = 1e-5
    coords = mol.atom_coords()
    dipder_fd = np.zeros((natm, 3, 3, nao, nao))
    for at in range(natm):
        for xyz in range(3):
            disp = np.zeros_like(coords)
            disp[at,xyz] = delta
            # +delta displacement
            mol_plus = mol.copy()
            mol_plus.set_geom_(coords+disp, unit='Bohr')
            d_plus = mol_plus.intor('int1e_r', comp=3)

            # -delta displacement
            mol_minus = mol.copy()
            mol_minus.set_geom_(coords-disp, unit='Bohr')
            d_minus = mol_minus.intor('int1e_r', comp=3)

            dipder_fd[at, :, xyz, :, :] = (d_plus - d_minus) / (2 * delta)
            
    return dipder_fd

def efield_deriv_fd(mol:gto.Mole, efield, R, old_paxes=None, atoms=None):
    N = mol.natm
    delta = 1e-5
    coords = mol.atom_coords()
    masses = mol.atom_mass_list()
    efield_de_fd = np.zeros((N,3,3))

    if atoms is None:
        _, eigvec = get_principal_axes(coords, masses, old_paxes)

    for at in range(N):
        for xyz in range(3):
            disp = np.zeros_like(coords)
            disp[at,xyz] = delta

            # +delta displacement
            if atoms is None:
                _, U_plus = get_principal_axes(coords+disp, masses, eigvec)
            else:
                U_plus = get_local_axes(*(coords+disp)[atoms])[0]
            
            # -delta displacement
            if atoms is None:
                _, U_minus = get_principal_axes(coords-disp, masses, eigvec)
            else:
                U_minus = get_local_axes(*(coords+disp)[atoms])[0]
         
            field_minus = process_efield(U_minus, efield, R)
            field_plus = process_efield(U_plus, efield, R)

            efield_de_fd[at,xyz] = (field_plus-field_minus)/(2*delta)
           
    return efield_de_fd



