import os
import numpy as np


def get_counter():
    counter_file = 'results/counter.txt'
    if not os.path.exists(counter_file):
        with open(counter_file, 'w') as f:
            f.write('000000')
    with open(counter_file, 'r') as f:
        counter = int(f.read().strip()) + 1
    with open(counter_file, 'w') as f:
        f.write(f'{counter:06}')
    return f'{counter:06}'


def save_results(results):
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    ns = results['n']

    counter = get_counter()
    filename = os.path.join(result_dir, f'{counter}_{results["method"]}_{results["basis"]}_{results["ghost_atoms"]}_ghost_{results["ghost_basis"]}.txt')

    with open(filename, 'w') as f:
        f.write(f'{counter}\n')
        f.write(f'{results["method"]}\n')
        f.write(f'{results["basis"]}\n')
        f.write(f'{results["ghost_atoms"]}\n')
        f.write(f'{results["ghost_basis"]}\n')
        f.write('--------------\n')

        for i, n in enumerate(ns):
            f.write(f'{i+1}th normal mode with the correct symmetry\n')
            f.write(f'n: {n}:\n')
            f.write(f'Frequency n (cm^-1): {results["freq_wavenumber"][n]}\n')
            f.write(f'Element (n,n) of hessian: {results["hessian"][n, n]}\n')
            for j, hess in enumerate(results['hessian_components']):
                f.write(f'Element (n,n) of hessian component {j+1}: {hess[n, n]}\n')
            f.write('----------------\n')

        f.write('Full list of frequencies:\n')
        np.savetxt(f, results['freq_wavenumber'])

        f.write('Full hessian matrix:\n')
        np.savetxt(f, results['hessian'])
        for j, hess in enumerate(results['hessian_components']):
            f.write(f'Full hessian component {j+1} matrix: \n')
            np.savetxt(f, hess)  # , fmt='%0.9f')

    print(f'Results saved to {filename}')


def handle_results(results):
    n = results['n'][0]

    hessian_components_nn = [array[n, n] for array in results["hessian_components"]]

    handled_results = {
        'method': results["method"],
        'basis': results["basis"],
        'ghost_atoms': results["ghost_atoms"],
        'ghost_basis': results["ghost_basis"],
        'n': n,
        'frequency_n': results["freq_wavenumber"][n],
        'hessian_nn': results["hessian"][n, n],
        'hessian_components_nn': hessian_components_nn
    }

    return handled_results


if __name__ == '__main__':
    # Example usage
    results = {
        "basis": "6-311G",
        "freq_wavenumber": [100, 200],
        "n": [0, 1, 0],
        "hessian": np.array([[1, 2], [3, 4]]),
        "hessian_components": [np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]])],
        "method": "mp2",
        "ghost_atoms": 0,
        "ghost_basis": ""
    }
    save_results(results)
