from abc import ABC, abstractmethod
from pyscf import scf, gto, grad, hessian


class BaseCalculation(ABC):
    def __init__(self, molecule, basis, symm = None):
        self.mol = molecule
        self.mol.basis = basis
        self.mol.symm = symm
        self.mol.build()
        self.gradient = None
        self.hessian = None
        print(self.mol.natm)

    @abstractmethod
    def numerical_gradient(self):
        """
        The numerical gradient for each calculation
        """
        pass

    @abstractmethod
    def analytical_electronic_gradient(self):
        """
        The analytical gradient for each calculation
        """
        pass

    def analytical_gradient(self, components=True):
        grad_elec, grad_parts_elec = self.analytical_electronic_gradient(components=components)
        grad_nuc = self.grad.grad_nuc()
        grad = grad_elec + grad_nuc
        grad_parts = grad_parts_elec + (grad_nuc,)
        if not components:
            return grad
        return grad, grad_parts

    @abstractmethod
    def numerical_hessian(self):
        """
        The numerical hessian for each calculation"
        """
        pass

    @abstractmethod
    def analytical_hessian(self):
        """
        The analytical hessian for each calculation"
        """
        pass

    def frequencies(self):

        hess, _ = self.numerical_hessian()

        return hessian.thermo.harmonic_analysis(self.mol, hess, exclude_trans=False, exclude_rot=False, imaginary_freq=False)
