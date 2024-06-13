import pytest
from hf_calculation import HFCalculation
from pyscf import gto


@pytest.fixture
def hf_calculation_instance():
    mol = gto.M(atom='''
        H 0 0 0
        H 0 0 0.74
    ''', basis='sto-3g')
    return HFCalculation(mol, mol.basis, opt=True)


def test_frequencies(hf_calculation_instance):
    freq = hf_calculation_instance.frequencies()
    assert freq is not None
    # Agrega más aserciones según sea necesario para verificar la salida esperada

# Agrega más funciones de prueba según sea necesario
