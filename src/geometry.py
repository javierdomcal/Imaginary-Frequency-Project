import time
import numpy as np
from pyscf import gto, scf, mp
from pyscf.geomopt import berny_solver

# Define the base molecule
BASE_MOLECULE = '''
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
'''


def add_ghost_atoms(atom_str, num_ghosts):
    if num_ghosts == 0:
        return atom_str
    elif num_ghosts == 1:
        return atom_str + 'X 0.000000 0.000000 0.000000'
    elif num_ghosts == 6:
        return atom_str + """
            X 0.000000 1.402725 0.000000
            \nX 1.214790 0.701362 0.000000
            \nX 1.214790 -0.701362 0.000000
            \nX 0.000000 -1.402725 0.000000
            \nX -1.214790 -0.701362 0.000000
            \nX -1.214790 0.701362 0.000000"""
    else:
        raise ValueError("Unsupported number of ghost atoms: must be 0, 1, or 6")


def create_molecule(atom_str, basis, ghost_basis=None):
    mol = gto.M()
    mol.atom = atom_str
    natm = mol.natm
    if ghost_basis:
        mol.basis = {'X': gto.basis.load(ghost_basis, 'C'), 'C': basis, 'H': basis}
    else:
        mol.basis = basis
    mol.build()
    return mol


def optimize_geometry(mol, method):
    if method == 'hf':
        mf = scf.RHF(mol)
    elif method == 'mp2':
        mf = scf.RHF(mol).run()
        mf = mp.MP2(mf)
    else:
        raise ValueError("Unsupported calculation method: must be 'hf' or 'mp2'")

    mol_optimized = berny_solver.optimize(mf)
    return mol_optimized


def fix_core_optimize_ghosts(mol, method):

    mol.set_geom_(mol.atom_coords(), inplace = True)
    if method == 'hf':
        mf = scf.RHF(mol)
    elif method == 'mp2':
        mf = scf.RHF(mol).run()
        mf = mp.MP2(mf)
    else:
        raise ValueError("Unsupported calculation method: must be 'hf' or 'mp2'")

    params = {"constraints": "constraints.txt", }
    mol_optimized = berny_solver.optimize(mf, **params)
    return mol_optimized


def main(basis, ghost_basis, num_ghosts, calculation_method):
    try:
        # Step 1: Optimize the molecule without ghost atoms
        mol = create_molecule(BASE_MOLECULE, basis)

        start_time = time.time()
        mol_optimized = optimize_geometry(mol, calculation_method)
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"Full geometry optimization time (without ghost atoms): {optimization_time:.2f} seconds")

        # Step 2: Add ghost atoms
        coords = mol_optimized._atom
        coords_str = ''
        for atom_coords in coords:
            coords_str += f"{atom_coords[0]} {atom_coords[1][0]*0.529177249:.6f} {atom_coords[1][1]*0.529177249:.6f} {atom_coords[1][2]*0.529177249:.6f}\n"
        atom_str_with_ghosts = add_ghost_atoms(coords_str, num_ghosts)
        mol_with_ghosts = create_molecule(atom_str_with_ghosts, basis, ghost_basis)

        # Step 3: Optimize only ghost atoms
        if num_ghosts > 0:
            start_time = time.time()
            mol_with_ghosts_optimized = fix_core_optimize_ghosts(mol_with_ghosts, calculation_method)
            end_time = time.time()
            optimization_time = end_time - start_time
            print(f"Ghost atoms optimization time: {optimization_time:.2f} seconds")
        else:
            mol_with_ghosts_optimized = mol_with_ghosts

        # Print the final optimized geometry
        print("Optimized geometry:")
        print(mol_with_ghosts_optimized.atom_coords())

        # Save the optimized geometry to an output file
        with open('optimized_geometry.xyz', 'w') as f:
            for atom, coord in zip(mol_with_ghosts_optimized.atom, mol_with_ghosts_optimized.atom_coords()):
                f.write(f"{atom[0]} {coord[0]} {coord[1]} {coord[2]}\n")

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    basis = 'sto-3g'  # Basis set for real atoms
    ghost_basis = '6-31G'  # Basis set for ghost atoms
    num_ghosts = 1  # Number of ghost atoms: 0, 1, or 6
    calculation_method = 'mp2'  # Calculation method: 'hf' or 'mp2'

    main(basis, ghost_basis, num_ghosts, calculation_method)
