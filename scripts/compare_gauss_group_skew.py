
import numpy as np
import sys,os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t as mvt 
from scipy.stats import multivariate_normal, chi2
import sklearn
from sklearn.covariance import GraphicalLasso
import scipy
from sklearn.metrics import zero_one_loss
from sklearn.metrics.pairwise import pairwise_kernels
import time
import tqdm
import networkx as nx


sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import pickle
from sklearn.datasets import make_sparse_spd_matrix


def generate_gen_skew_t(Sigma, gamma, nu, n):
    from scipy.stats import uniform

    T = lambda u: np.array([chi2.ppf(u, df = nu[k])/nu[k] for k in range(len(nu))])

    d = Sigma.shape[0]
    x = np.random.multivariate_normal(mean = np.zeros(d),cov = Sigma, size = n)
    y = np.zeros(shape = (n, d))

    V = np.zeros(shape = (n, d))

    for i in range(n):


        u = uniform.rvs()
        y[i] = np.reciprocal(T(u))*gamma  + np.sqrt(np.reciprocal(T(u)))*x[i]
        V[i] = T(u)

    return y, V



if __name__ == '__main__':
    d = 20
    s = 0.35
    alpha_prob =  0.8 #[0.03, 0.15, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99 ]
    alphas = [0.01, 0.05, 0.075]
    kappas = [0.4]
    tol = 1e-5
    nr_quad = 5

    while True:
        G =nx.erdos_renyi_graph(d, s, seed=42)


        A = np.array(nx.adjacency_matrix(G).todense())
        A[np.tril_indices(d)] = 0
        U = np.random.binomial(1,0.5,  size = (d,d))
        A = np.multiply(A, U*np.random.uniform(-0.6,-0.2, size = (d,d)) + (1-U)*np.random.uniform(0.2,0.6, size = (d,d)))
        A = A+A.T
        np.fill_diagonal(A, np.sum(np.abs(A),axis=1))

        A2 = A.copy()
        A2_SIGNS = np.sign(A2)
        A2 = np.power(np.abs(A2), 1.3)
        A2 = A2*A2_SIGNS
        np.fill_diagonal(A2,np.diag(A))
        np.round(A2[:,1],2)


        try:
            np.linalg.inv(A)
            np.linalg.inv(A2)
        except:
            continue

        obs_per_graph = 100
        n = 300
        break


    pbar = tqdm.tqdm(total = len(alphas)*len(kappas))


    # X1 = multivariate_normal.rvs(mean = np.zeros(d), cov = np.linalg.inv(A), size = n ) 
    # X2 = multivariate_normal.rvs(mean = np.zeros(d), cov = np.linalg.inv(A2), size = n ) 
    # X1 = mvt.rvs(shape = np.linalg.inv(A),  df = 4, size = n) 
    # X2 = mvt.rvs(shape = np.linalg.inv(A2),  df = 4, size = n) 
    gamma = np.array([0.05] * 10 + [0.1]*5 + [-0.05]*5)
    nu = [4]*5 + [5]*5 + [8]*5 + [10]*5
    groups= [0]*5 + [1]*5 + [2]*5 + [3]*5
    X1,_ = generate_gen_skew_t(np.linalg.inv(A),gamma = gamma,nu = nu, n = n)
    X2,_ = generate_gen_skew_t(np.linalg.inv(A2),gamma = gamma,nu = nu, n = n)
    X = np.vstack((X1,X2))
    
    nu_for_model = [nu] * 6

    prec_dict ={}
    dens_t = {i: [] for i in range(len(kappas))}
    time_t = {i: [] for i in range(len(kappas))}
    theta_t = {i: [] for i in range(len(kappas))}
    its_t = {i: [] for i in range(len(kappas))}
    tol_t = {i: [] for i in range(len(kappas))}

    dens_n = {i: [] for i in range(len(kappas))}
    time_n = {i: [] for i in range(len(kappas))}
    theta_n = {i: [] for i in range(len(kappas))}
    its_n = {i: [] for i in range(len(kappas))}
    tol_n = {i: [] for i in range(len(kappas))}

    dens_gt = {i: [] for i in range(len(kappas))}
    time_gt = {i: [] for i in range(len(kappas))}
    theta_gt = {i: [] for i in range(len(kappas))}
    its_gt = {i: [] for i in range(len(kappas))}
    tol_gt = {i: [] for i in range(len(kappas))}

    dens_st = {i: [] for i in range(len(kappas))}
    time_st = {i: [] for i in range(len(kappas))}
    theta_st = {i: [] for i in range(len(kappas))}
    gamma_st = {i: [] for i in range(len(kappas))}
    its_st = {i: [] for i in range(len(kappas))}
    tol_st = {i: [] for i in range(len(kappas))}

    no_est = True
    for i, kappa in enumerate(kappas):
        for alpha in alphas:   
            
            # t
            start = time.time()
            dg_opt_t = dg.dygl(obs_per_graph = obs_per_graph, max_iter = 2000, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
            dg_opt_t.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="t", nr_em_itr = 1,True_prec= [A,A,A,A2,A2,A2], 
                        time_index=range(X.shape[0]), nu = [4]*6, em_tol = 1e-10, verbose =True)
            elapsed = time.time()-start
            dens_t[i].append(nx.density(G))
            time_t[i].append(elapsed)
            theta_t[i].append(dg_opt_t.theta)
            tol_t[i].append(dg_opt_t.fro_norm)
            its_t[i].append(dg_opt_t.iteration)

            # Skew-t
            if no_est:
                theta_init=dg_opt_t.theta
            else:
                theta_init=dg_opt_st.theta
            start = time.time()
            dg_opt_st = dg.dygl(obs_per_graph = obs_per_graph, max_iter = 100, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, kappa_gamma=obs_per_graph*0.1, tol = tol)
            dg_opt_st.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="skew-group-t", nr_em_itr = 1, theta_init=theta_init, 
                        True_prec= [A,A,A,A2,A2,A2],
                        time_index=range(X.shape[0]), nu = nu_for_model, groups = groups, em_tol = 1e-10, nr_quad =10, verbose =True)
            elapsed = time.time()-start
            dens_st[i].append(nx.density(G))
            time_st[i].append(elapsed)
            theta_st[i].append(dg_opt_st.theta)
            gamma_st[i].append(dg_opt_st.gamma)
            tol_st[i].append(dg_opt_st.fro_norm)
            its_st[i].append(dg_opt_st.iteration) 


            # Group-t
            if no_est:
                theta_init=dg_opt_t.theta
            else:
                theta_init=dg_opt_gt.theta
            start = time.time()
            dg_opt_gt = dg.dygl(obs_per_graph = obs_per_graph, max_iter = 100, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
            dg_opt_gt.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="group-t", nr_em_itr = 1, theta_init=theta_init,
                          True_prec= [A,A,A,A2,A2,A2],
                        time_index=range(X.shape[0]), nu = nu_for_model, groups = groups, em_tol = 1e-10, nr_quad =10, verbose =True)
            elapsed = time.time()-start
            dens_gt[i].append(nx.density(G))
            time_gt[i].append(elapsed)
            theta_gt[i].append(dg_opt_gt.theta)
            tol_gt[i].append(dg_opt_gt.fro_norm)
            its_gt[i].append(dg_opt_gt.iteration)


            # Normal
            start = time.time()
            dg_opt_n = dg.dygl(obs_per_graph = obs_per_graph, max_iter = 2000, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
            dg_opt_n.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="gaussian",True_prec= [A,A,A,A2,A2,A2],
                          time_index=range(X.shape[0]), verbose =False)
            elapsed = time.time()-start
            dens_n[i].append(nx.density(G))
            time_n[i].append(elapsed)
            theta_n[i].append(dg_opt_n.theta)

            tol_n[i].append(dg_opt_n.fro_norm)
            its_n[i].append(dg_opt_n.iteration)

            no_est = False



            pbar.set_description(f"s {s}, a {np.round(alpha, 2)}, a {np.round(kappa, 2)}")
            pbar.update()

            out_dict = {'nr_obs_per_graph':obs_per_graph, 'n':n, 'temporal_penalty':'global-reconstruction', 'A1':A, 'A2':A2, 'prec': [A,A,A,A2,A2,A2], 'gamma':gamma,
            'density_t':dens_t, 'time_t':time_t,  'theta_t':theta_t,  'tol_t':tol_t,
            'density_gt':dens_gt, 'time_gt':time_gt,  'theta_gt':theta_gt,  'tol_gt':tol_gt,
            'density_st':dens_st, 'time_st':time_st,  'theta_st':theta_st,  'tol_st':tol_st, 'gamma_st':gamma_st,
            'density_n':dens_n,  'time_n':time_n,  'theta_n':theta_n, 'tol_n':tol_n,
            'alpha':alphas, 'kappa':kappas, 'max_iter':2000, 'tol':tol, 'nr_quad':nr_quad } 

            # import pickle
            with open(f'data/gauss_group_skew/skew_search_d_{d}_n_{n}_s_{s}_nrquad_{10}_erdos_hradar_hk.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
