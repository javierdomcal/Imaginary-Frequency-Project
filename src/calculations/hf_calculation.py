from base_calculation import BaseCalculation
from pyscf.geomopt import berny_solver
from pyscf import scf, gto, grad, hessian
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


class HFCalculation(BaseCalculation):
    def __init__(self, molecule, basis, symm = None, opt=True):
        super().__init__(molecule, basis, symm)
        self.mf = scf.RHF(self.mol).run()
        self.mf.verbose = 3
        if opt:
            self.mol = berny_solver.optimize(self.mf)
            self.mf = scf.RHF(self.mol).run()
            self.mf.verbose = 3

        self.grad = self.mf.Gradients()
        self.grad.verbose = 3
        self.grad.kernel()

    def analytical_electronic_gradient(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, components=True):
        """
        Electronic part of RHF/RKS gradients

        Args:
                        self.grad : grad.rhf.Gradients or grad.rks.Gradients object
        """

        mf = self.grad.base

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        log = logger.Logger(self.grad.stdout, self.grad.verbose)

        hcore_deriv = self.grad.hcore_generator(self.mol)
        s1 = self.grad.get_ovlp(self.mol)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0 = self.grad._tag_rdm1(dm0, mo_coeff, mo_occ)

        t0 = (logger.process_clock(), logger.perf_counter())

        log.debug('Computing Gradients of NR-HF Coulomb repulsion')
        vhf = self.grad.get_veff(self.mol, dm0)
        log.timer('gradients of 2e part', *t0)

        dme0 = self.grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

        if atmlst is None:
            atmlst = range(self.mol.natm)
        aoslices = self.mol.aoslice_by_atom()
        de = numpy.zeros((len(atmlst), 3))
        de_1 = numpy.zeros((len(atmlst), 3))
        de_2 = numpy.zeros((len(atmlst), 3))
        de_3 = numpy.zeros((len(atmlst), 3))
        de_4 = numpy.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
            de_1[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
        # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>

            de[k] += numpy.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2
            de_2[k] += numpy.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2

            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2
            de_3[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2

            de[k] += self.grad.extra_force(ia, locals())
            de_4[k] += self.grad.extra_force(ia, locals())

        if log.verbose >= logger.DEBUG:
            log.debug('gradients of electronic part')
            grad.rhf._write(log, self.mol, de, atmlst)

        if components:
            return de, (de_1, de_2, de_3, de_4)
        return de

    def numerical_hessian(self, step=0.00005291772, components=None):
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

                newHF = HFCalculation(mol_moved, self.mol.basis, symm=False, opt=False)

                grad_moved, grad_parts_moved = newHF.analytical_gradient()

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


if __name__ == "__main__":
    mol = gto.M(
        atom='''
	            H 0 0 0
	            H 0 0 0.74
	        ''',
        basis='sto-3g'
    )

    HFcalc = HFCalculation(mol, opt=True)
    freq = HFcalc.frequencies()
    print(thermo.dump_normal_mode(mol, freq))
