import pytest
from make_matrixes import open_scenarios, gauss_det
import numpy as np

matrixes = open_scenarios()

a = list(map(lambda x : gauss_det(x), matrixes))
b = list(map(lambda x : np.linalg.det(x), matrixes))
comparisson = list(zip(a,b))

@pytest.mark.parametrize("test_input,expected", comparisson)
def test_calculations(test_input, expected):
    assert abs(test_input - expected)  < 1e-6

if __name__ == '__main__':
    pytest.main()
