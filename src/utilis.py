import numpy as np
from pyscf import gto, lib, scf, cc
from pyscf.cc import ccsd_rdm
from pyscf.grad.ccsd import _rdm2_mo2ao, _load_block_tril, reduce, _response_dm1
from pyscf.grad.mp2 import _shell_prange, _index_frozen_active, has_frozen_orbitals
from pyscf.lib import logger

def get_com(coords:np.ndarray, masses:np.ndarray) -> tuple[float, float, float]:
    "Get center of mass from coordinates and masses "

    coords = np.asarray(coords)
    masses = np.asarray(masses)

    return np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)

def symmetrize_matrix(X:np.ndarray):
    return 0.5 * (X + X.T)

def relaxed_dm(mycc, mol, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              d1=None, d2=None):
    " Extract relaxed density matrix from PySCF CCSD calculation "
    if eris is not None:
        if abs(eris.fock - np.diag(eris.fock.diagonal())).max() > 1e-3:
            raise RuntimeError('CCSD gradients does not support NHF (non-canonical HF)')

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2

    if d1 is None:
        d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    
    fdm2 = lib.H5TmpFile()
    if d2 is None:
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)

    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mycc.mo_occ > 0)
    with_frozen = has_frozen_orbitals(mycc)
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)

# Roughly, dm2*2 is computed in _rdm2_mo2ao
    mo_active = mo_coeff[:,np.hstack((OA,VA))]
    _rdm2_mo2ao(mycc, d2, mo_active, fdm2)  # transform the active orbitals
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    Imat = np.zeros((nao,nao))

# 2e AO integrals dot 2pdm
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = np.zeros((3,nao,nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = _load_block_tril(fdm2['dm2'], ip0, ip1, nao)
            dm2buf[:,:,diagidx] *= .5
            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            dm2buf = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao,-1))
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[i] += np.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[i] -= np.einsum('ijkl,il->kj', eri1tmp, hf_dm1[ip0:ip1]) * .5
                vhf[i,ip0:ip1] += np.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[i,ip0:ip1] -= np.einsum('ijkl,jk->il', eri1tmp, hf_dm1) * .5
            eri1 = eri1tmp = None

    Imat = reduce(np.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mycc, Xvo, eris)
    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    
    # Hartree-Fock part contribution
    dm1 += hf_dm1
    
    return dm1

def get_dipm(mol:gto.Mole, dm:np.ndarray, origin=(0,0,0)) -> tuple:
    with mol.with_common_origin(origin):
        dip_ints = mol.intor_symmetric('int1e_r', comp=3)
    edip = np.einsum('ijk, jk->i', dip_ints, dm)
    ndip = np.einsum('i, ij->j', mol.atom_charges(), mol.atom_coords())
    return ndip, edip

def int1e_deriv(atom_id:float, eint, mol:gto.Mole):
    """ 
    Resemble derivatives of integrals w.r.t electron coordinates to the nuclear derivatives
    The function works for integrals of quantum operators that are independent to nuclear coordinates, 
    i.e., overlap, electronic dipole moment
    
    """
    shl0, shl1, p0, p1 = (mol.aoslice_by_atom())[atom_id]
    nint = np.zeros_like(eint)
    nint[:, :, p0:p1] -= eint[:, :, p0:p1]
    nint += np.swapaxes(nint, 1, 2)
   
    return nint

def process_efield(U:np.ndarray, efield, R):
    """ U is eigenvectors or derivatives of eigenvectors """
    
    efield = U @ R[:,2] * efield # rotate the third vector

    return efield

def finalize(grad, de):
    if grad.verbose >= logger.NOTE:
        logger.note(grad, '--------------- Electric field contribution ---------------')
        grad._write(grad.mol, de, grad.atmlst)
        logger.note(grad, '----------------------------------------------')
