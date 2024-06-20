from main import main  # Import the main function from main.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the src directory is two levels up from utils
sys.path.append(os.path.abspath(os.path.join(__file__, '../src')))


def gather_results(basis_family, method):
    results_list = []

    for basis in basis_family:
        ghost_atoms = []  # Define as needed
        ghost_basis = None  # Define as needed

        # Run the main function and get formatted results
        formatted_results = main(basis, ghost_atoms, method, ghost_basis)

        results_list.append(formatted_results)

    return results_list


def main_6_31G_MP2():
    basis_family = ['6-31G', '6-31G*', '6-31G**', '6-31+G', '6-31+G*', '6-31+G**', '6-31++G', '6-31++G*', '6-31++G**']  # Add more if needed
    method = 'MP2'

    results_list = gather_results(basis_family, method)

    # Convert results to pandas DataFrame
    df = pd.DataFrame(results_list)
    print(df)

    # Plotting 'hessian_components_nn'
    hessian_data = []
    for result in df['results']:
        hessian_data.append(result['hessian_components_nn'])

    plt.figure(figsize=(10, 6))
    for data in hessian_data:
        plt.plot(data, label=f'basis: {df["basis"]}')

    plt.xlabel('Index')
    plt.ylabel('hessian_components_nn')
    plt.title('Hessian Components NN for 6-31G Basis Sets using MP2 Method')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main_6_31G_MP2()
