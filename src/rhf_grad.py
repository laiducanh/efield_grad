from pyscf.lib import logger
from pyscf.grad import rhf
from .grad import grad_efield
from .finite_diff import rhf_grad_fd
from .utilis import finalize
from .scf import EFieldRHF
import numpy as np

class EFieldRHFGradients(rhf.Gradients):
    _keys = rhf.Gradients._keys
    _keys.update({'efield_strength', 'efield_R', 'efield_atoms'})
    def __init__(self, method:EFieldRHF):
        super().__init__(method)
        
        self.efield_strength = method.efield_strength
        self.efield_R = method.efield_R
        self.efield_atoms = method.efield_atoms
    
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
            
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        
        if self.base.do_disp():
            self.de += self.get_dispersion()
   
        # Add electric field contribution
        dm = self.base.make_rdm1()
        g = grad_efield(self.mol, dm, self.efield_strength, 
                        self.efield_R, self.base.old_paxes, self.efield_atoms)
        self.base._set_old_paxes()
        if self.verbose >= logger.DEBUG:
            finalize(self, g)
        self.de += g

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        
        de_fd = rhf_grad_fd(self.mol, self.efield_strength, self.efield_R, self.efield_atoms)
        logger.note(self, 'Check with finite difference %.10f' % np.linalg.norm(de_fd-self.de))
        # self._write(self.mol, de_fd, self.atmlst)

        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()

        return self.de