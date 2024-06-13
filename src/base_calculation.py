from abc import ABC, abstractmethod
from pyscf import scf, gto, grad, hessian
import numpy


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

    def numerical_hessian(self, step=0.00005291772):
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

                energy_moved = self._run_calculation(mol_moved)

                grad_moved, grad_parts_moved = self._calculate_gradient(mol_moved)

                hessian[:, atom, :, j] = (grad_moved - grad) / step

                for n, (force, force_moved) in enumerate(zip(grad_parts, grad_parts_moved)):
                    hessian_forces[n][:, atom, :, j] = (force_moved - force) / step

        return hessian, hessian_forces

    @abstractmethod
    def _run_calculation(self, mol):
        """
        Abstract method to run the electronic structure calculation.
        """
        pass

    @abstractmethod
    def _calculate_gradient(self, mol):
        """
        Abstract method to compute the gradient.
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
