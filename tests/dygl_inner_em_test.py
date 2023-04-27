import sys, os
sys.path.append('src')
import numpy as np
from DyGraph import dygl_inner_em
from tests.utils import Generate_data, assert_positive_eig


obs_per_graph = 50
alpha = 0.01
kappa = 0.1
kappa_gamma = 0.1
tol = 1e-6
tol_integrate = 1e-1

def test_dygl_inner_em():

    X, A = Generate_data()
    
    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol)
    dg_opt.fit(lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1, temporal_penalty='element-wise')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the Gaussian element-wise"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol)
    dg_opt.fit(lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1, temporal_penalty='global-reconstruction')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the Gaussian global-reconstruction"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol)
    dg_opt.fit(lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1, temporal_penalty='ridge')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the Gaussian ridge"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = 1e-2)
    dg_opt.fit(lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1, temporal_penalty='block-wise-reconstruction',  bwr_xtol = 1e-4)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<1e-2, f"Tolerance not less than {1e-2} for the Gaussian block-wise-reconstruction"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = 1e-2)
    dg_opt.fit(lik_type='gaussian', nu = None,verbose=True,  nr_em_itr = 1, temporal_penalty='perturbed-node', p_node_tol= 1e-4, p_node_max_iter = 1000)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<1e-2, f"Tolerance not less than {1e-2} for the Gaussian 'perturbed-node"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol)
    dg_opt.fit(lik_type='t', nu = None,verbose=True,  nr_em_itr = 5, temporal_penalty='element-wise')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol)
    dg_opt.fit(lik_type='t', nu = [4]*6, verbose=True,  nr_em_itr = 1, temporal_penalty='element-wise')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, tol = tol_integrate, groups = [0]*X.shape[1])
    dg_opt.fit(lik_type='group-t', nu = [[4]*X.shape[1]] * 6,verbose=True,  nr_em_itr = 1, temporal_penalty='element-wise')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the group-t case"

    dg_opt = dygl_inner_em(X, obs_per_graph=50, max_iter = 2000, lamda = alpha, kappa = kappa, kappa_gamma= kappa_gamma, tol = tol_integrate, groups = [0]*X.shape[1], lik_type='skew-group-t')
    dg_opt.fit(nu = None, verbose=True,  nr_em_itr = 1, temporal_penalty='element-wise')
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the skew-group-t case"
