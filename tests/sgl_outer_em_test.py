import sys, os
sys.path.append('src')

from DyGraph import sgl_outer_em
from tests.utils import Generate_data, assert_positive_eig


obs_per_graph = 50
alpha = 0.01
tol = 1e-6
tol_integrate = 1e-1


def test_sgl_outer_em():

    X, A = Generate_data()
    d = X.shape[1]

    dg_opt = sgl_outer_em(X, lik_type='gaussian', max_iter = 2000, lamda = alpha, tol = tol)
    dg_opt.fit(nu = None,verbose=True,  nr_admm_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the Gaussian case"


    dg_opt = sgl_outer_em(X, lik_type='t', max_iter = 2000, lamda = alpha, tol = tol)
    dg_opt.fit( nu = None,verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = sgl_outer_em(X, lik_type='t', max_iter = 2000, lamda = alpha, tol = tol)
    dg_opt.fit( nu = [4],verbose=True,  nr_em_itr = 5)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol, f"Tolerance not less than {tol} for the t case"

    dg_opt = sgl_outer_em(X, lik_type='group-t', max_iter = 200, lamda = alpha, tol = tol_integrate, groups = [0]*d)
    dg_opt.fit( nu = None, verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {1e-1} for the group-t case"


    dg_opt = sgl_outer_em(X, lik_type='group-t', max_iter = 200, lamda = alpha, tol = tol_integrate, groups = [0]*d)
    dg_opt.fit(nu = [[4]*d], verbose=True,  nr_em_itr = 1)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the group-t case"

    dg_opt = sgl_outer_em(X, lik_type='skew-group-t', max_iter = 200, lamda = alpha, tol = tol_integrate, groups = [0]*d)
    dg_opt.fit(nu = None, verbose=True,  nr_em_itr = 1, nr_workers = 2)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the skew-group-t case"
    
    theta_init = dg_opt.theta
    dg_opt = sgl_outer_em(X, lik_type='skew-group-t', max_iter = 200, lamda = alpha, tol = tol_integrate, groups = [0]*d)
    dg_opt.fit( nu = [[4]*d], verbose=True,  nr_em_itr = 1, nr_workers = 2, theta_init= theta_init)
    assert_positive_eig(dg_opt.theta[-1])
    assert dg_opt.fro_norm<tol_integrate, f"Tolerance not less than {tol_integrate} for the skew-group-t case"