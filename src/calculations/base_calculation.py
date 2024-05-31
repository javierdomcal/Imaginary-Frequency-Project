from abc import ABC, abstractmethod
from pyscf import scf, gto, grad, hessian


class BaseCalculation(ABC):
    """
    Abstract base class for performing electronic structure calculations.

    Attributes:
        molecule: The molecular configuration for the calculation.
        basis: The basis set used for the calculation.
        symm: The symmetry information for the molecule (optional).

    Methods:
        numerical_gradient(): Abstract method for computing the numerical gradient.
        analytical_electronic_gradient(): Abstract method for computing the analytical electronic gradient.
        analytical_gradient(components=True): Compute the analytical gradient including nuclear contributions.
        numerical_hessian(): Abstract method for computing the numerical Hessian.
        analytical_hessian(): Abstract method for computing the analytical Hessian.
        frequencies(): Compute the frequencies of the molecular configuration.
    """

    def __init__(self, molecule, basis, symm = None):
        """
        Initializes a BaseCalculation instance.

        Args:
            molecule: The molecular configuration.
            basis: The basis set for the calculation.
            symm: The symmetry information for the molecule (optional).
        """
        self.mol = molecule
        self.mol.basis = basis
        self.mol.symm = symm
        self.mol.build()
        self.gradient = None
        self.hessian = None

    @abstractmethod
    def numerical_gradient(self):
        """
        Abstract method to compute the numerical gradient.
        """
        pass

    @abstractmethod
    def analytical_electronic_gradient(self):
        """
        Abstract method to compute the analytical electronic gradient.
        """
        pass

    def analytical_gradient(self, components=True):
        """
        Compute the analytical gradient including nuclear contributions.

        Args:
            components: Boolean indicating whether to return gradient components.

        Returns:
            If components is True, returns the gradient and its components.
            Otherwise, returns the gradient.
        """
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
        Abstract method to compute the numerical Hessian.
        """
        pass

    @abstractmethod
    def analytical_hessian(self):
        """
        Abstract method to compute the analytical Hessian.
        """
        pass

    def frequencies(self):
        """
        Compute the frequencies of the molecular configuration.

        Returns:
            The frequencies of the molecular configuration.
        """

        hess, _ = self.numerical_hessian()

        return hessian.thermo.harmonic_analysis(self.mol, hess, exclude_trans=False, exclude_rot=False, imaginary_freq=False)
