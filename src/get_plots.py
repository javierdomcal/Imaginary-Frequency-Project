from retrieve_freq_mode import harmonic_analysis_moved
import base_calculation
from hf_calculation import HFCalculation
from mp2_calculation import MP2Calculation
import matplotlib.pyplot as plt
from pyscf import gto
import numpy as np

basiss = ['6-31G']  # , '6-31G*', '6-31G**',
# '6-31+G', '6-31+G*', '6-31+G**',
#  '6-31++G', '6-31++G*', '6-31++G**']


colors = ['green', 'green', 'green', 'yellow', 'yellow', 'yellow', 'red', 'red', 'red']

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
    symmetry=False
)


for basis in basiss:
    #my_hf = HFCalculation(molecule, basis, symm=None, opt=True)

    #hess, hess_comp = my_hf.numerical_hessian()
    #freq, hess, h, prob = harmonic_analysis_moved(molecule, hess, components= hess_comp)
    #hf_hessians[basis] = []
    # for hes in hess:
    #    hf_hessians[basis].append(hes[prob, prob])

    # my_mp2 = MP2Calculation(molecule, basis, symm=None, opt=True)

    # hess, hess_comp = my_mp2.numerical_hessian()
    # freq, hess, h, prob = harmonic_analysis_moved(molecule, hess, components= hess_comp)
    # hf_hessians[basis] = []

    # for hes in hess:
    #     mp2_hessians[basis].append(hes[prob, prob])

    # Number of basis sets and force components
    pass
num_basiss = len(basiss)

for i, basis in enumerate(basiss):

    hf_hessians[basis] = [i]
    mp2_hessians[basis] = [i**2]

#num_force_components = len(hf_hessians[basiss[0]])
num_force_components = 1

# Width of each bar
bar_width = 0.35

# X positions for the bars
x = np.arange(num_force_components)

# Plotting
fig, ax = plt.subplots()

# Plotting HF data
for i, basis in enumerate(basiss):
    ax.bar(x - bar_width / 2 + i * bar_width / num_basiss, hf_hessians[basis], width=bar_width / num_basiss,
           label=f'{basis} HF', color='b')

# Plotting MP2 data
for i, basis in enumerate(basiss):
    ax.bar(x + bar_width / 2 + i * bar_width / num_basiss, mp2_hessians[basis], width=bar_width / num_basiss,
           label=f'{basis} MP2', color='r')

# Adding labels and title
ax.set_xlabel('Force Components')
ax.set_ylabel('Values')
ax.set_title('Force Components Comparison')
ax.set_xticks(x)
ax.set_xticklabels([f'Component {i+1}' for i in range(num_force_components)])
ax.legend()

# Show plot
plt.show()
