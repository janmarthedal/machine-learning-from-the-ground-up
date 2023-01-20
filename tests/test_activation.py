import unittest
import numpy as np
from activation import IDENTITY_ACTIVATION

class TestActivation(unittest.TestCase):

    def test_identity(self):
        x = np.array([-2, -1, -0.5, 0, 0.3, 1, 3], dtype=np.float64)
        np.testing.assert_allclose(IDENTITY_ACTIVATION[0](x), x)
        np.testing.assert_allclose(IDENTITY_ACTIVATION[1](x), np.ones_like(x))

if __name__ == '__main__':
    unittest.main()
