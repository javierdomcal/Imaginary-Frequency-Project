from pyscf import __config__
import numpy
from pyscf import gto, hessian, scf
from pyscf.hessian import thermo
from pyscf.geomopt import berny_solver
from pyscf.data import nist
import matplotlib.pyplot as plt
setattr(__config__, 'symm_geom_tol', 1e-1)


LINDEP_THRESHOLD = 1e-7


def harmonic_analysis_moved(mol, hess, exclude_trans=False, exclude_rot=False,
                            imaginary_freq=False, mass=None, components=None):
    '''Each column is one mode

    imaginary_freq (boolean): save imaginary_freq as complex number (if True)
    or negative real number (if False)
    '''
    if mass is None:
        mass = mol.atom_mass_list(isotope_avg=True)
    results = {}
    atom_coords = mol.atom_coords()
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    natm = atom_coords.shape[0]

    mass_hess = numpy.einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
    h = mass_hess.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)

    # Initialize list to store processed additional Hessians
    processed_components = []

    # Check if there are additional Hessian matrices to process
    if components is not None:
        for additional_hess in components:
            # Process each additional Hessian matrix similarly
            mass_additional_hess = numpy.einsum('pqxy,p,q->pqxy', additional_hess, mass**-.5, mass**-.5)
            additional_h = mass_additional_hess.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
            processed_components.append(additional_h)

    TR = thermo._get_TR(mass, atom_coords)
    TRspace = []
    if exclude_trans:
        TRspace.append(TR[:3])

    if exclude_rot:
        rot_const = rotation_const(mass, atom_coords)
        rotor_type = _get_rotor_type(rot_const)
        if rotor_type == 'ATOM':
            pass
        elif rotor_type == 'LINEAR':  # linear molecule
            TRspace.append(TR[3:5])
        else:
            TRspace.append(TR[3:])

    transformed_hess = []

    if TRspace:
        TRspace = numpy.vstack(TRspace)
        q, r = numpy.linalg.qr(TRspace.T)
        P = numpy.eye(natm * 3) - q.dot(q.T)
        w, v = numpy.linalg.eigh(P)
        bvec = v[:, w > LINDEP_THRESHOLD]
        h = reduce(numpy.dot, (bvec.T, h, bvec))
        force_const_au, mode = numpy.linalg.eigh(h)
        mode = bvec.dot(mode)
    else:
        force_const_au, mode = numpy.linalg.eigh(h)
        for processed_component in processed_components:
            transformed_hess.append(numpy.linalg.inv(mode) @ processed_component @ mode)
        h_diagonal = numpy.linalg.inv(mode) @ h @ mode

    freq_au = numpy.lib.scimath.sqrt(force_const_au)
    results['freq_error'] = numpy.count_nonzero(freq_au.imag > 0)
    if not imaginary_freq and numpy.iscomplexobj(freq_au):
        # save imaginary frequency as negative frequency
        freq_au = freq_au.real - abs(freq_au.imag)

    results['freq_au'] = freq_au
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    results['freq_wavenumber'] = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2

    norm_mode = numpy.einsum('z,zri->izr', mass**-.5, mode.reshape(natm, 3, -1))
    results['norm_mode'] = norm_mode
    reduced_mass = 1. / numpy.einsum('izr,izr->i', norm_mode, norm_mode)
    results['reduced_mass'] = reduced_mass

    # https://en.wikipedia.org/wiki/Vibrational_temperature
    results['vib_temperature'] = freq_au * au2hz * nist.PLANCK / nist.BOLTZMANN

    # force constants
    dyne = 1e-2 * nist.HARTREE2J / nist.BOHR_SI**2
    results['force_const_au'] = force_const_au
    results['force_const_dyne'] = reduced_mass * force_const_au * dyne  # cm^-1/a0^2

    problematic_freq = None

    # for i, vibr in enumerate(force_const_au):
    #     this_mode = mode[:, i]
    #     cartesian_this_mode = this_mode
    #     cartesian_this_mode = cartesian_this_mode.reshape(natm, 3)

    #     newMol = mol

    #     coords = mol.atom_coords(unit='bohr')
    #     coords += cartesian_this_mode

    #     newMol = newMol.set_geom_(coords, inplace=False, unit='bohr', symmetry=None)
    #     # newMol.atom_coords() += 0.1 * cartesian_this_mode
    #     newMol.symmetry = True
    #     newMol.build()

    #     print(i, newMol.topgroup, newMol.groupname, results['freq_wavenumber'][i])

    #     if ((i == 0) or (i > 5)) and newMol.topgroup == 'C2h':
    #         problematic_freq = i

    pi = numpy.pi

    rotation = numpy.array(
        [[numpy.cos(pi / 3), numpy.sin(pi / 3), 0],
         [-numpy.sin(pi / 3), numpy.cos(pi / 3), 0],
         [0, 0, 1]])

    reflexion = numpy.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]])

    relation_original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    relation_antihor = [3, 1, 5, 6, 4, 2, 9, 7, 11, 12, 10, 8]
    relation_hor = [2, 6, 1, 5, 3, 4, 8, 12, 7, 11, 9, 10]
    relation_adhoc = [6, 1, 2, 3, 4, 5, 12, 7, 8, 9, 10, 11]
    relation_inv = [4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]

    problematic_freq = []
    problematic_not = []
    print(mol.atom_coords(unit='bohr'))

    for i, vibr in enumerate(force_const_au):
        this_mode = mode[:, i]
        cartesian_this_mode = this_mode 
        cartesian_this_mode = cartesian_this_mode.reshape(natm, 3)  # + mol.atom_coords(unit='bohr')
        new_mode = numpy.zeros((natm, 3))
        unequal_mode = numpy.zeros((natm, 3))

        for atom in range(0, 12):
            row = cartesian_this_mode[atom, :]

            moved_row = numpy.dot(row, reflexion) @ rotation
            new_mode[relation_adhoc[atom] - 1, :] = moved_row

            unequal_row = numpy.dot(row, reflexion)
            unequal_mode[relation_original[atom] - 1, :] = unequal_row

        threshold = 0.001

        if numpy.allclose(new_mode, cartesian_this_mode, atol=threshold):
            problematic_freq.append(i)

            if not numpy.allclose(cartesian_this_mode, unequal_mode, atol=threshold * 0.0001):
                problematic_not.append(i)
            # break

    if problematic_freq == []:
        problematic_freq = None

    # if force_const_au[0] < -100:
    #     problematic_freq = 0
    # else:
    #     problematic_freq = 8

    # TODO: IR intensity
    return results, h_diagonal, transformed_hess, problematic_freq


def run_hessian_analysis(mol):
    mf = scf.RHF(mol).run()
    mol = berny_solver.optimize(mf)
    mf_opt = scf.RHF(mol).run()
    hess = mf_opt.Hessian().kernel()
    freq, hess, h, problematic_freq = harmonic_analysis_moved(mol, hess, exclude_trans=False, exclude_rot=False, imaginary_freq=False)
    print(problematic_freq, freq['freq_wavenumber'], problematic_freq)
    return freq, hess, h, problematic_freq


if __name__ == '__main__':

    benzene_d6d_init = gto.M(
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
        basis='6-31++G',
        symmetry=False
    )

    benzene_d6d_init.build()
    freq, hess, h, problematic_freq = run_hessian_analysis(benzene_d6d_init)
