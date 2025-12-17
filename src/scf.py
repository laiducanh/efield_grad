from pyscf import gto
from pyscf.scf import hf
from pyscf.lib import logger
import numpy as np
from .principal_frame import get_principal_axes, eig_blocks
from .local_frame import get_local_axes
from .utilis import process_efield

class RHF(hf.RHF):
    conv_tol = 1e-10
    conv_tol_grad = 1e-8  
    _keys = hf.RHF._keys
    _keys.update({'efield_strength', 'efield_R', 'efield_atoms', 'old_paxes'})
    def __init__(self, mol:gto.Mole, efield=0.0, Rotation=np.eye(3), atoms=None):
        """ if `atoms` is specified, local frame will be defined by `atoms` """
        super().__init__(mol)
        self.efield_strength = efield
        self.efield_R = Rotation
        self.old_paxes = None
        if atoms is None:
            self.efield_atoms = None
        else:
            self.efield_atoms = list(atoms)
        self.mol = mol
    
    def _check_principal_axes(self, tol=1e-8):
        """ return True if the moment of inertia is (near) degenerate """
        moments, _ = get_principal_axes(self.mol.atom_coords(), self.mol.atom_mass_list(), self.old_paxes)
        if self.verbose >= 2:
            logger.note(self, f'Reference axes: {self.old_paxes}')
        for block in eig_blocks(moments, tol):
            if len(block) == 3 and self.verbose >= 2:
                logger.warn(self, 'Degenerate moments of inertia: I{} = I{} = I{}.'.format(*block))
            elif len(block) == 2 and self.verbose >= 2:
                logger.warn(self, 'Degenerate moments of inertia: I{} = I{}.'.format(*block))
    
    def _get_efield(self):
        # update electric field vector
        mol = self.mol
        if self.efield_atoms is None:
            _, axes = get_principal_axes(mol.atom_coords(), mol.atom_mass_list(), self.old_paxes)
        else:
            axes = get_local_axes(*self.mol.atom_coords()[self.efield_atoms])[0]
        return process_efield(axes, self.efield_strength, self.efield_R)

    def _set_old_paxes(self, old_axes=None):
        if old_axes is None:
            mol = self.mol
            _, self.old_paxes = get_principal_axes(mol.atom_coords(), mol.atom_mass_list(), self.old_paxes)
        else:
            self.old_paxes = old_axes
        # print("Store old paxes\n", self.old_paxes)

    # override core Hamiltonian added electronic dipole moment component
    def get_hcore(self, mol:gto.Mole=None):
        if mol is None: mol = self.mol
        EFIELD = self._get_efield()
        if self.verbose >= 1: # default self.verbose = 3
            print("Hamiltonian is modified with electric field: " \
            "Ex = {:.5f}, Ey = {:.5f}, Ez = {:.5f}".format(*EFIELD))        
        
        H0 = super().get_hcore(mol) # field-free core Hamiltonian
        dip_ints = mol.intor_symmetric('int1e_r', comp=3) # dipole moment integrals
        H_field = H0 - np.einsum('i, ijk->jk', EFIELD, dip_ints) # field-perturbed Hamiltonian

        return H_field

    # overide nuclear energy added nuclear dipole moment component
    def energy_nuc(self):
        mol = self.mol
        EFIELD = self._get_efield()
        nu_dip = np.einsum('i, ij->j', mol.atom_charges(), mol.atom_coords())
        return super().energy_nuc() + np.dot(EFIELD, nu_dip) 
    
    def scf(self, dm0=None, **kwargs):
        if self.efield_atoms is None:
            self._check_principal_axes()
        return super().scf(dm0, **kwargs)