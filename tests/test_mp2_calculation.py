# test_mp2_calculation.py
import pytest
from mp2_calculation import MP2Calculation
from pyscf import gto


@pytest.fixture
def molecule():
    # Correctly defined fixture for creating a PySCF molecule
    return gto.M(
        atom='''
            H 0 0 0
            H 0 0 0.74
        ''',
        basis='sto-3g'
    )


@pytest.fixture
def setup_mp2_calculation(molecule):
    # Fixture to setup MP2Calculation with a real molecule
    # Pytest will automatically provide the 'molecule' fixture as an argument
    return MP2Calculation(molecule, opt=False)


def test_initialization(setup_mp2_calculation):
    # Pytest automatically injects the 'setup_mp2_calculation' fixture
    mp2_calc = setup_mp2_calculation
    assert mp2_calc.mol is not None
    assert mp2_calc.mf.verbose == 3


def test_geometry_optimization(molecule):
    # Directly use 'molecule' fixture for this test
    mp2_calc = MP2Calculation(molecule, opt=True)
    # Perform your test logic here
    # For example, checking that the molecule has been optimized:
    # assert <some condition related to optimization>


def test_analytical_gradient(molecule):
    # Since 'test_geometry_optimization' shouldn't be called from here,
    # we set up a new MP2Calculation instance specifically for this test
    mp2_calc = MP2Calculation(molecule, opt=True)  # Assuming no optimization needed here
    gradient = mp2_calc.analytical_gradient()
    assert gradient is not None  # Adjust based on actual expected outcomes


def test_numerical_hessian(molecule):
    # Since 'test_geometry_optimization' shouldn't be called from here,
    # we set up a new MP2Calculation instance specifically for this test
    mp2_calc = MP2Calculation(molecule, opt=True)  # Assuming no optimization needed here
    hessian, _ = mp2_calc.numerical_hessian()
    assert hessian is not None  # Adjust based on actual expected outcomes
