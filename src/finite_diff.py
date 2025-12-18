from pyscf import gto, cc
import numpy as np
from .principal_frame import get_principal_axes
from .local_frame import get_local_axes
from .utilis import process_efield
from .scf import EFieldRHF

def dipint_grad_fd(mol:gto.Mole):
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

def efield_grad_fd(mol:gto.Mole, efield, R, old_paxes=None, atoms=None):
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
                U_minus = get_local_axes(*(coords-disp)[atoms])[0]
         
            field_minus = process_efield(U_minus, efield, R)
            field_plus = process_efield(U_plus, efield, R)

            efield_de_fd[at,xyz] = (field_plus-field_minus)/(2*delta)
           
    return efield_de_fd


def rhf_grad_fd(mol:gto.Mole, efield, R, atoms=None):
    N = mol.natm
    delta = 1e-6
    coords = mol.atom_coords()
    energy_de = np.zeros((N, 3))
    mf = EFieldRHF(mol, efield, R, atoms)
    mf.verbose = 0
    mf._set_old_paxes()
    for at in range(N):
        for xyz in range(3):
            disp = np.zeros_like(coords)
            disp[at,xyz] = delta
            # +delta displacement            
            mol.set_geom_(coords+disp, unit='Bohr')
            mf_plus = EFieldRHF(mol, efield, R, atoms)
            mf_plus.old_paxes = mf.old_paxes
            mf_plus.verbose = 0
            e_plus = mf_plus.kernel()

            # -delta displacement
            mol.set_geom_(coords-disp, unit='Bohr')
            mf_minus = EFieldRHF(mol, efield, R, atoms)
            mf_minus.old_paxes = mf.old_paxes
            mf_minus.verbose = 0
            e_minus = mf_minus.kernel()

            energy_de[at,xyz] = (e_plus - e_minus) / (2 * delta)

    return energy_de

def ccsd_grad_fd(mol:gto.Mole, efield, R, atoms=None):
    N = mol.natm
    delta = 1e-6
    coords = mol.atom_coords()
    energy_de = np.zeros((N, 3))
    mf = EFieldRHF(mol, efield, R, atoms)
    mf.verbose = 0
    mf._set_old_paxes()
    for at in range(N):
        for xyz in range(3):
            disp = np.zeros_like(coords)
            disp[at,xyz] = delta
            # +delta displacement            
            mol.set_geom_(coords+disp, unit='Bohr')
            mf_plus = EFieldRHF(mol, efield, R, atoms)
            mf_plus.old_paxes = mf.old_paxes
            mf_plus.verbose = 0
            e_plus = mf_plus.kernel()
            mycc = cc.CCSD(mf_plus).set_frozen()
            mycc.verbose = 0
            e_corr, _, _ = mycc.kernel()
            e_plus += e_corr

            # -delta displacement
            mol.set_geom_(coords-disp, unit='Bohr')
            mf_minus = EFieldRHF(mol, efield, R, atoms)
            mf_minus.old_paxes = mf.old_paxes
            mf_minus.verbose = 0
            e_minus = mf_minus.kernel()
            mycc = cc.CCSD(mf_minus).set_frozen()
            mycc.verbose = 0
            e_corr, _, _ = mycc.kernel()
            e_minus += e_corr

            energy_de[at,xyz] = (e_plus - e_minus) / (2 * delta)

    return energy_de