from pyscf.grad import ccsd
from pyscf.lib import logger
import numpy as np
from .utilis import relaxed_dm, finalize
from .grad import grad_efield, ccsd_deriv_fd

class EFieldCCSDGradients(ccsd.Gradients):
    _keys = ccsd.Gradients._keys
    _keys.update({'efield_strength', 'efield_R', 'efield_atoms'})
    def __init__(self, method, efield=0.0, Rotation=np.eye(3), atoms=None):
        super().__init__(method)
        self.efield_strength = efield
        self.efield_R = Rotation
        if atoms is None:
            self.efield_atoms = None
        else:
            self.efield_atoms = list(atoms)

    def kernel(self, t1=None, t2=None, l1=None, l2=None, eris=None,
                     atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        mycc = self.base
        if t1 is None:
            if mycc.t1 is None:
                mycc.run()
            t1 = mycc.t1
        if t2 is None: t2 = mycc.t2
        if l1 is None: l1 = mycc.l1
        if l2 is None: l2 = mycc.l2
        if eris is None:
            eris = mycc.ao2mo()
        if t1 is None or t2 is None:
            t1, t2 = mycc.kernel(eris=eris)[1:]
        if l1 is None or l2 is None:
            l1, l2 = mycc.solve_lambda(eris=eris)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        de = self.grad_elec(t1, t2, l1, l2, eris, atmlst, verbose=log)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        
        # Add electric field contribution
        dm = relaxed_dm(mycc, self.mol, t1, t2, l1, l2, eris, atmlst)    
        g = grad_efield(self.mol, dm, self.efield_strength, 
                        self.efield_R, mycc._scf.old_paxes, self.efield_atoms)
        mycc._scf._set_old_paxes()
        if self.verbose >= logger.DEBUG:
            finalize(self, g)
        self.de += g

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        
        # de_fd = ccsd_deriv_fd(self.mol, self.efield_strength, self.efield_R)
        # logger.note(self, 'Check with finite difference %.10f' % np.linalg.norm(de_fd-self.de))
        # self._write(self.mol, de_fd, self.atmlst)
        
        self._finalize()
        return self.de