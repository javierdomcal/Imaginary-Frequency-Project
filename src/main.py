from retrieve_freq_mode import harmonic_analysis_moved
import base_calculation
from hf_calculation import HFCalculation
from mp2_calculation import MP2Calculation
import matplotlib.pyplot as plt
from pyscf import gto
import numpy as np

basiss = ['6-311G', '6-311G*', '6-311G**',
          '6-311+G', '6-311+G*', '6-311+G**',
          '6-311++G', '6-311++G*', '6-311++G**']
# basiss = [ '6-31+G**',
#          '6-31++G', '6-31++G*', '6-31++G**']
# basiss = ['def2-TZVP', 'def2-TZVPP', 'def2-TZVPPD']
# basiss = ['6-31G']

hf_grads = {}
hf_hessians = {}

mp2_grads = {}
mp2_hessians = {}

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


def calculate_task(input_value, molecule, calc):
    # Perform some calculation
    if calc == 'hf':
        result = calculation_HF(input_value, molecule)
    elif calc == 'mp2':
        result = calculation_MP2(input_value, molecule)
    # Save result to a file or database
    with open(f'results/result_{calc}_{input_value}.txt', 'w') as f:
        for item in result:
            f.write(str(item) + '\n')
    return result


def trials_deleting(input_value, molecule, name):
    result = calculation_MP2(input_value, molecule)

    with open(f'../results/result_{name}_{input_value}.txt', 'w') as f:
        for item in result:
            f.write(str(item) + '\n')
    return result


def calculation_HF(basis, molecule, symm=True, opt=True):
    my_hf = HFCalculation(molecule, basis, symm=symm, opt=opt)
    hess, hess_comp = my_hf.numerical_hessian()
    freq, hess, h, prob = harmonic_analysis_moved(molecule, hess, components= hess_comp)
    hf_hessians = [basis]
    hf_hessians.append(freq['freq_wavenumber'][prob])
    hf_hessians.append(prob)
    hf_hessians.append(h[prob, prob])
    for hes in hess:
        hf_hessians.append(hes[prob, prob])
    return hf_hessians


def calculation_MP2(basis, molecule, symm=True, opt=True):
    my_mp2 = MP2Calculation(molecule, basis, symm=symm, opt=opt)
    hess, hess_comp = my_mp2.numerical_hessian()
    freq, hess, h, prob = harmonic_analysis_moved(molecule, hess, components= hess_comp)
    mp2_hessians = [basis]
    mp2_hessians.append(freq['freq_wavenumber'][prob])
    mp2_hessians.append(prob)
    mp2_hessians.append(h[prob, prob])
    for hes in hess:
        mp2_hessians.append(hes[prob, prob])
    return mp2_hessians


test_name = 'arizona'
for basis in basiss:
    result = trials_deleting(basis, molecule, test_name)

# tasks = [calculate_task(input_value, molecule, 'mp2') for input_value in basiss]

# results = [task for task in tasks]

# for input_value, result in zip(basiss, results):
#    print(f"Calculation for input {input_value} completed with result: {result}")
