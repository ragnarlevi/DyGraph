import numpy as np
from numpy.testing import assert_array_less
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import multivariate_t 
import sys

sys.path.append('../')

default_eigvalue_precision = float("-1e-5")

def Generate_data():
    d = 20
    A = make_sparse_spd_matrix(d, alpha=0.6)
    return multivariate_t.rvs(loc = np.zeros(d),df = 4, shape = np.linalg.inv(A), size=200), A


def assert_positive_eig(K):
    """Assert true if the calculated kernel matrix is valid."""
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    assert_array_less(default_eigvalue_precision, min_eig)

