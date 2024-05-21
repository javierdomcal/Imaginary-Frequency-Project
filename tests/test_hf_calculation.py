# test_hf_calculation.py
import pytest
from hf_calculation import HFCalculation
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
def setup_hf_calculation(molecule):
    # Fixture to setup HFCalculation with a real molecule
    # Pytest will automatically provide the 'molecule' fixture as an argument
    return HFCalculation(molecule, opt=False)


def test_initialization(setup_hf_calculation):
    # Pytest automatically injects the 'setup_hf_calculation' fixture
    hf_calc = setup_hf_calculation
    assert hf_calc.mol is not None
    assert hf_calc.mf.verbose == 3


def test_geometry_optimization(molecule):
    # Directly use 'molecule' fixture for this test
    hf_calc = HFCalculation(molecule, opt=True)
    # Perform your test logic here
    # For example, checking that the molecule has been optimized:
    # assert <some condition related to optimization>


def test_analytical_gradient(molecule):
    # Since 'test_geometry_optimization' shouldn't be called from here,
    # we set up a new HFCalculation instance specifically for this test
    hf_calc = HFCalculation(molecule, opt=True)  # Assuming no optimization needed here
    gradient = hf_calc.analytical_gradient()
    assert gradient is not None  # Adjust based on actual expected outcomes


def test_numerical_hessian(molecule):
    # Since 'test_geometry_optimization' shouldn't be called from here,
    # we set up a new HFCalculation instance specifically for this test
    hf_calc = HFCalculation(molecule, opt=True)  # Assuming no optimization needed here
    hessian, _ = hf_calc.numerical_hessian()
    assert hessian is not None  # Adjust based on actual expected outcomes


def test_frequencies(molecule):
    hf_calc = HFCalculation(molecule, opt=True)
    freq = hf_calc.frequencies()
    assert freq is not None
