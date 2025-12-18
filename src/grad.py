from pyscf import gto
import numpy as np
from .utilis import get_dipm, process_efield, int1e_deriv
from .principal_frame import get_principal_axes
from .principal_frame import dU_dR as pf_dU
from .local_frame import get_local_axes
from .local_frame import dU_dR as lf_dU
from .finite_diff import efield_grad_fd, dipint_grad_fd

def grad_efield(mol:gto.Mole, dm:np.ndarray, efield:float, R:np.ndarray, old_paxes:np.ndarray, atoms=None):
    """ dm is (relaxed) density matrix 
        efield is the electric field strength 
        theta, phi are in degrees
    """
    N = mol.natm
    nao = mol.nao_nr()
    g = np.zeros((N, 3))
    ndip, edip = get_dipm(mol, dm)
    dip_moment = ndip - edip
    print('Correlated total dipole moment', dip_moment)

    if atoms is None:
        _, U = get_principal_axes(mol.atom_coords(), mol.atom_mass_list(), old_paxes)
    else:
        U = get_local_axes(*mol.atom_coords()[atoms])[0]
    EFIELD = process_efield(U, efield, R)
    print('Electric field: Ex = {:.5f}, Ey = {:.5f}, Ez = {:.5f}'.format(*EFIELD))
        
    # derivatives of electric field in the principal axes
    if atoms is None:
        dU = pf_dU(mol.atom_coords(), mol.atom_mass_list(), old_paxes)
    else:
        dU = lf_dU(mol.atom_coords(), atoms)
    efield_de = process_efield(dU, efield, R)
    efield_de_fd = efield_grad_fd(mol, efield, R, old_paxes, atoms)
    print('Check derivatives of electric field', np.linalg.norm(efield_de-efield_de_fd))
    g += np.einsum('ijk, k->ij', efield_de, dip_moment)

    # derivatives of electronic dipole moment integrals
    eint = mol.intor('int1e_irp', comp=9)
    dip_de = np.zeros(((N,) + eint.shape))
    for at in range(N):
        dip_de[at] = int1e_deriv(at, eint, mol)
    dip_de = dip_de.reshape(N, 3, 3, nao, nao) # Natom * mu * xyz * nao * nao
    dip_de_fd = dipint_grad_fd(mol)
    print("Check derivatives of electronic dipole moment integrals", np.linalg.norm(dip_de-dip_de_fd))
    # contract with Density
    dip_de = np.einsum('mn, ijkmn->ijk', dm, dip_de)
    # add derivatives of nuclear dipole moment integrals
    for at in range(N):
        for xyz in range(3):
            dip_de[at, xyz, xyz] -= mol.atom_charge(at)

    g -= np.einsum('ijk, j->ik', dip_de, EFIELD)
    
    return g