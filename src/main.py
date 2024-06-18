import os
from pyscf import gto
from hf_calculation import HFCalculation
from mp2_calculation import MP2Calculation
from retrieve_freq_mode import harmonic_analysis_moved
import output_handler


def create_molecule(basis, ghost_atoms):
    molecule = gto.M(
        atom='''
C      0.000000     1.402725     0.000000
C      1.214790     0.701362     0.000000
C      1.214790    -0.701362     0.000000
C      0.000000    -1.402725     0.000000
C     -1.214790    -0.701362     0.000000
C     -1.214790     0.701362     0.000000
H      0.000000     2.490314     0.000000
H      2.157746     1.245157     0.000000
H      2.157746    -1.245157     0.000000
H      0.000000    -2.490314     0.000000
H     -2.157746    -1.245157     0.000000
H     -2.157746     1.245157     0.000000
    ''',
        symmetry=True
    )
    # ghost_atom = 'He 0.0 0.0 0.0\n'  # Example of a ghost atom
    #ghost_atoms_coords = ghost_atom * ghost_atoms
    return molecule


def run_calculation(basis, ghost_atoms, method, ghost_basis):
    molecule = create_molecule(basis, ghost_atoms)

    if method.lower() == 'hf':
        calc = HFCalculation(molecule, basis, symm=True, opt=True)
    elif method.lower() == 'mp2':
        calc = MP2Calculation(molecule, basis, symm=True, opt=True)
    else:
        raise ValueError(f"Unsupported calculation method: {method}")

    hess, hess_comp = calc.numerical_hessian()
    print('ppppp')
    print(hess)
    freq, hessian, hessian_components, n = harmonic_analysis_moved(molecule, hess, components=hess_comp)
    results = {
        "basis": basis,
        "freq_wavenumber": freq['freq_wavenumber'],
        "n": n,
        "hessian": hessian,
        "hessian_components": hessian_components,
        "method": method,
        "ghost_atoms": ghost_atoms,
        "ghost_basis": ghost_basis
    }
    return results


def main(basis, ghost_atoms, method, ghost_basis):
    results = run_calculation(basis, ghost_atoms, method, ghost_basis)
    output_handler.save_results(results)
    formatted_results = output_handler.handle_results(results)
    return formatted_results


if __name__ == '__main__':
    basis = '6-31++G'
    ghost_atoms = 0
    method = 'mp2'
    ghost_basis = None
    null = main(basis, ghost_atoms, method, ghost_basis)
