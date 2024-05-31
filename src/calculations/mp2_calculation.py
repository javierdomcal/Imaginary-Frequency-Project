from base_calculation import BaseCalculation
from pyscf.geomopt import berny_solver
from pyscf import scf, gto, grad, hessian, mp
from pyscf.lib import logger
import numpy
import ctypes
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf.gto.mole import is_au
import pyscf.grad as grad
import pyscf.hessian.thermo as thermo
from functools import reduce
from pyscf.data import nist
from functools import reduce
from pyscf.grad import rhf as rhf_grad
from pyscf.mp import mp2
from pyscf.ao2mo import _ao2mo


class MP2Calculation(BaseCalculation):
    def __init__(self, molecule, basis, symm = None, opt=True):
        super().__init__(molecule, basis, symm = None)
        self.mf = scf.RHF(self.mol).run()
        self.mf = mp.MP2(self.mf)
        self.mf.verbose = 3
        if opt:
            self.mol = berny_solver.optimize(self.mf)
            self.mf = scf.RHF(self.mol).run()
            self.mf = mp.MP2(self.mf)
            self.mf.verbose = 3
            energy = self.mf.run()

        self.grad = self.mf.Gradients()
        self.grad.verbose = 3
        self.grad.kernel()
        self.t2 = None
        if self.t2 is None:
            self.t2 = self.grad.base.t2
        if self.t2 is None:
            self.t2 = self.grad.base.kernel()[1]

    def analytical_electronic_gradient(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, components=True):
        mp = self.grad.base
        mp_grad = self.grad
        t2 = self.t2
        log = logger.new_logger(mp, self.grad.verbose)
        time0 = logger.process_clock(), logger.perf_counter()

        log.debug('Build mp2 rdm1 intermediates')
        d1 = mp2._gamma1_intermediates(mp, t2)
        doo, dvv = d1
        time1 = log.timer_debug1('rdm1 intermediates', *time0)

    # Set nocc, nvir for half-transformation of 2pdm.  Frozen orbitals are exculded.
    # nocc, nvir should be updated to include the frozen orbitals when proceeding
    # the 1-particle quantities later.
        mol = mp_grad.mol
        with_frozen = not ((mp.frozen is None)
                           or (isinstance(mp.frozen, (int, numpy.integer)) and mp.frozen == 0)
                           or (len(mp.frozen) == 0))
        OA, VA, OF, VF = grad.mp2._index_frozen_active(mp.get_frozen_mask(), mp.mo_occ)
        orbo = mp.mo_coeff[:, OA]
        orbv = mp.mo_coeff[:, VA]
        nao, nocc = orbo.shape
        nvir = orbv.shape[1]

    # Partially transform MP2 density matrix and hold it in memory
    # The rest transformation are applied during the contraction to ERI integrals
        part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc**2, nvir**2),
                                numpy.asarray(orbv.T, order='F'), (0, nao, 0, nao),
                                's1', 's1').reshape(nocc, nocc, nao, nao)
        part_dm2 = (part_dm2.transpose(0, 2, 3, 1) * 4 -
                    part_dm2.transpose(0, 3, 2, 1) * 2)

        hf_dm1 = mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)

        if atmlst is None:
            atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
        diagidx = numpy.arange(nao)
        diagidx = diagidx * (diagidx + 1) // 2 + diagidx

        de = numpy.zeros((len(atmlst), 3))
        de_1 = numpy.zeros((len(atmlst), 3))
        de_2 = numpy.zeros((len(atmlst), 3))
        de_3 = numpy.zeros((len(atmlst), 3))
        de_4 = numpy.zeros((len(atmlst), 3))
        de_5 = numpy.zeros((len(atmlst), 3))
        de_6 = numpy.zeros((len(atmlst), 3))
        de_7 = numpy.zeros((len(atmlst), 3))
        de_8 = numpy.zeros((len(atmlst), 3))

        Imat = numpy.zeros((nao, nao))
        fdm2 = lib.H5TmpFile()
        vhf1 = fdm2.create_dataset('vhf1', (len(atmlst), 3, nao, nao), 'f8')

    # 2e AO integrals dot 2pdm
        max_memory = max(0, mp.max_memory - lib.current_memory()[0])
        blksize = max(1, int(max_memory * .9e6 / 8 / (nao**3 * 2.5)))

        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            ip1 = p0
            vhf = numpy.zeros((3, nao, nao))
            for b0, b1, nf in grad.mp2._shell_prange(mol, shl0, shl1, blksize):
                ip0, ip1 = ip1, ip1 + nf
                dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
                dm2buf += lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:, ip0:ip1])
                dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
                dm2buf = dm2buf + dm2buf.transpose(0, 1, 3, 2)
                dm2buf = lib.pack_tril(dm2buf.reshape(-1, nao, nao)).reshape(nf, nao, -1)
                dm2buf[:, :, diagidx] *= .5

                shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
                eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
                Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf, nao, -1), dm2buf)
                eri0 = None

                eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3, nf, nao, -1)
                de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
                de_1[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2

                dm2buf = None
    # HF part
                for i in range(3):
                    eri1tmp = lib.unpack_tril(eri1[i].reshape(nf * nao, -1))
                    eri1tmp = eri1tmp.reshape(nf, nao, nao, nao)
                    vhf[i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                    vhf[i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1[ip0:ip1]) * .5
                    vhf[i, ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                    vhf[i, ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1) * .5
                eri1 = eri1tmp = None
            vhf1[k] = vhf
            log.debug('2e-part grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
            time1 = log.timer_debug1('2e-part grad of atom %d' % ia, *time1)

    # Recompute nocc, nvir to include the frozen orbitals and make contraction for
    # the 1-particle quantities, see also the kernel function in ccsd_grad module.
        mo_coeff = mp.mo_coeff
        mo_energy = mp._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mp._scf.get_ovlp(), mo_coeff)) * -1

        dm1mo = numpy.zeros((nmo, nmo))
        if with_frozen:
            dco = Imat[OF[:, None], OA] / (mo_energy[OF, None] - mo_energy[OA])
            dfv = Imat[VF[:, None], VA] / (mo_energy[VF, None] - mo_energy[VA])
            dm1mo[OA[:, None], OA] = doo + doo.T
            dm1mo[OF[:, None], OA] = dco
            dm1mo[OA[:, None], OF] = dco.T
            dm1mo[VA[:, None], VA] = dvv + dvv.T
            dm1mo[VF[:, None], VA] = dfv
            dm1mo[VA[:, None], VF] = dfv.T
        else:
            dm1mo[:nocc, :nocc] = doo + doo.T
            dm1mo[nocc:, nocc:] = dvv + dvv.T

        dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
        vhf = mp._scf.get_veff(mp.mol, dm1) * 2
        Xvo = reduce(numpy.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
        Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

        dm1mo += grad.mp2._response_dm1(mp, Xvo)
        time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

        Imat[nocc:, :nocc] = Imat[:nocc, nocc:].T
        im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))
        time1 = log.timer_debug1('response_rdm1', *time1)

        log.debug('h1 and JK1')
        # Initialize hcore_deriv with the underlying SCF object because some
        # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
        mf_grad = mp_grad.base._scf.nuc_grad_method()
        hcore_deriv = mf_grad.hcore_generator(mol)
        s1 = mf_grad.get_ovlp(mol)

        zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
        zeta[nocc:, :nocc] = mo_energy[:nocc]
        zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
        zeta = reduce(numpy.dot, (mo_coeff, zeta * dm1mo, mo_coeff.T))

        dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
        p1 = numpy.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
        vhf_s1occ = reduce(numpy.dot, (p1, mp._scf.get_veff(mol, dm1 + dm1.T), p1))
        time1 = log.timer_debug1('h1 and JK1', *time1)

        # Hartree-Fock part contribution
        dm1[:] = 0
        dm1p = hf_dm1 + dm1 * 2
        dm1 += hf_dm1
        zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mp.mo_occ)

        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
    # s[1] dot I, note matrix im1 is not hermitian
            de[k] += numpy.einsum('xij,ij->x', s1[:, p0:p1], im1[p0:p1])
            de_2[k] += numpy.einsum('xij,ij->x', s1[:, p0:p1], im1[p0:p1])
            de[k] += numpy.einsum('xji,ij->x', s1[:, p0:p1], im1[:, p0:p1])
            de_3[k] += numpy.einsum('xji,ij->x', s1[:, p0:p1], im1[:, p0:p1])
    # h[1] \dot DM, contribute to f1
            h1ao = hcore_deriv(ia)
            de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
            de_4[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
    # -s[1]*e \dot DM,  contribute to f1
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], zeta[p0:p1])
            de_5[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], zeta[p0:p1])
            de[k] -= numpy.einsum('xji,ij->x', s1[:, p0:p1], zeta[:, p0:p1])
            de_6[k] -= numpy.einsum('xji,ij->x', s1[:, p0:p1], zeta[:, p0:p1])
    # -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2
            de_7[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2
            de[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)
            de_8[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)

        log.timer('%s gradients' % mp.__class__.__name__, *time0)
        if components:
            return de, (de_1, de_2, de_3, de_4, de_5, de_6, de_7, de_8)
        return de

    def numerical_hessian(self, step=0.0005291772):
        self.mol.symmetry = None
        natm = self.mol.natm
        grad, grad_parts = self.analytical_gradient()

        hessian = numpy.zeros((natm, natm, 3, 3))

        hessian_forces = []
        for force in grad_parts:
            hessian_forces.append(numpy.zeros((natm, natm, 3, 3)))

        for atom in range(natm):
            for j, coord in enumerate(['x', 'y', 'z']):
                coords = self.mol.atom_coords()
                coords[atom, j] += step
                mol_moved = self.mol.set_geom_(coords, inplace=False, symmetry=None)
                mol_moved.symmetry = None

                energy_moved = scf.RHF(mol_moved).run()

                newMP2 = MP2Calculation(mol_moved, self.mol.basis, symm=False, opt=False)

                grad_moved, grad_parts_moved = newMP2.analytical_gradient()

                hessian[:, atom, :, j] = (grad_moved - grad) / step

                for n, (force, force_moved) in enumerate(zip(grad_parts, grad_parts_moved)):
                    hessian_forces[n][:, atom, :, j] = (force_moved - force) / step

        return hessian, hessian_forces

    def analytical_hessian(self):
        """
        The analytical hessian for each calculation"
        """
        pass

    def numerical_gradient(self):
        """
        The numerical gradient for each calculation
        """
        pass
