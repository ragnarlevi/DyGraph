import sys, os
sys.path.append('src')
from pip._internal.operations.freeze import freeze

import numpy as np
import scipy
from DyGraph import sgl
from tests.utils import Generate_data, assert_positive_eig


obs_per_graph = 50
alpha = 0.01
tol = 1e-6
tol_integrate = 1e-1

def test_sgl_inner_em():

    X, A = Generate_data()
    
    dg_opt = sgl(max_iter = 2000, lamda = obs_per_graph*alpha, tol = tol)
    dg_opt.fit(X,  lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the Gaussian case"


    dg_opt = sgl(max_iter = 2000, lamda = obs_per_graph*alpha, tol = tol)
    dg_opt.fit(X,  lik_type='t', nu = None,verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = sgl(max_iter = 2000, lamda = obs_per_graph*alpha, tol = tol)
    dg_opt.fit(X,  lik_type='t', nu = [4],verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = sgl(max_iter = 20, lamda = obs_per_graph*alpha, tol = tol_integrate)
    dg_opt.fit(X,  lik_type='group-t', nu = None, verbose=True,  nr_em_itr = 1, groups = [0] * X.shape[1])
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {1e-1} for the group-t case"


    dg_opt = sgl(max_iter = 20, lamda = obs_per_graph*alpha, tol = tol_integrate)
    dg_opt.fit(X,  lik_type='group-t', nu = [[4]*X.shape[1]], verbose=True,  nr_em_itr = 1, groups = [0] * X.shape[1])
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the group-t case"

    dg_opt = sgl(max_iter = 20, lamda = obs_per_graph*alpha, tol = tol_integrate)
    dg_opt.fit(X,  lik_type='skew-group-t', nu = None, verbose=True,  nr_em_itr = 1, groups = [0] * X.shape[1], nr_workers = 2)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the skew-group-t case"
    
    theta_init = dg_opt.theta
    dg_opt = sgl(max_iter = 20, lamda = obs_per_graph*alpha, tol = tol_integrate)
    dg_opt.fit(X,  lik_type='skew-group-t', nu = [[4]*X.shape[1]], verbose=True,  nr_em_itr = 1, groups = [0] * X.shape[1], nr_workers = 2, theta_init= theta_init)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the skew-group-t case"





