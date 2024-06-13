import pytest
from mp2_calculation import MP2Calculation
from pyscf import gto
import numpy

# Define the molecule
mol = gto.M(
    atom='''
            H 0 0 0
            H 0 0 0.74
        ''',
    basis='sto-3g'
)


@pytest.fixture
def mp2_instance():
    return MP2Calculation(mol, basis='sto-3g', opt=False)


def test_frequencies(mp2_instance):
    freq = mp2_instance.frequencies()
    assert freq is not None
    # Agrega más aserciones según sea necesario para verificar la salida esperada
