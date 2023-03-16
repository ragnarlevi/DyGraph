
import numpy as np
import sys,os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t as mvt 
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
from utils import calc_one_zero_error, calc_f1, calc_precision, calc_recall, calc_density, calc_roc_auc, calc_balanced_accuaray



if __name__ == '__main__':
    d = 50
    alpha_prob =  0.8 #[0.03, 0.15, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99 ]
    alphas = np.linspace(0.01, 0.5, 20)
    kappas = np.linspace(0.01, 2, 20)
    tol = 1e-10

    #prec1 = make_sparse_spd_matrix(d, alpha=alpha_prob, smallest_coef=-0.9, largest_coef=0.9, norm_diag = True, random_state = 42)
    # prec2 = make_sparse_spd_matrix(d, alpha=alpha_prob, smallest_coef=-0.9, largest_coef=0.9, norm_diag = True, random_state = 412)
    # prec = np.block([[prec1, np.zeros((d,d))], [np.zeros((d,d)), prec2]])
    # prec = prec1
    # tmp = prec.copy()
    # np.fill_diagonal(tmp,0)
    # G = nx.from_numpy_array(tmp)
    # print(nx.density(G))
    # print("\n")
    nus = [4]


    for n_idx, s in enumerate([0.1, 0.2, 0.3, 0.4]):
        
        while True:
            

            G =nx.erdos_renyi_graph(d, s, seed=42)
            A = np.array(nx.adjacency_matrix(G).todense())
            A[np.tril_indices(d)] = 0
            A = np.multiply(A, np.random.uniform(-0.4,0.4, size = (d,d)))
            A = A+A.T
            np.fill_diagonal(A, np.sum(np.abs(A),axis=1))

            try:
                np.linalg.inv(A)
            except:
                continue

            obs_per_graph = 100
            prec = A
            n = 1000
            break
            
        pbar = tqdm.tqdm(total = len(alphas)*len(kappas)*len(nus))

        for nu in nus:
            X1 = mvt.rvs(shape = np.linalg.inv(prec),  df = nu, size = n) 
            X = X1
            prec_dict ={}

            dens_t = {i: [] for i in range(len(kappas))}
            time_t = {i: [] for i in range(len(kappas))}
            F_t = {i: [] for i in range(len(kappas))}
            f1_t = {i: [] for i in range(len(kappas))}
            zo_t = {i: [] for i in range(len(kappas))}
            l1_t = {i: [] for i in range(len(kappas))}
            theta_t = {i: [] for i in range(len(kappas))}
            fro_norm_t = {i: [] for i in range(len(kappas))}
            its_t = {i: [] for i in range(len(kappas))}

            dens_n = {i: [] for i in range(len(kappas))}
            time_n = {i: [] for i in range(len(kappas))}
            F_n = {i: [] for i in range(len(kappas))}
            zo_n = {i: [] for i in range(len(kappas))}
            l1_n = {i: [] for i in range(len(kappas))}
            f1_n = {i: [] for i in range(len(kappas))}
            theta_n = {i: [] for i in range(len(kappas))}
            fro_norm_n = {i: [] for i in range(len(kappas))}
            its_n = {i: [] for i in range(len(kappas))}

            for i, kappa in enumerate(kappas):
                for alpha in alphas:    



                    start = time.time()
                    dg_opt1 = dg.dygl_parallel(obs_per_graph = obs_per_graph, max_iter = 20000, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
                    dg_opt1.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="t", nr_em_itr = 1, time_index=range(X.shape[0]), nu = nu, em_tol = 1e-10, verbose =False)
                    elapsed = time.time()-start
                    dens_t[i].append(nx.density(G))
                    time_t[i].append(elapsed)
                    theta_t[i].append(dg_opt1.theta)

                    zo_t[i].append(np.mean([calc_one_zero_error(prec, dg_opt1.theta[k]) for k in range(len(dg_opt1.theta))]))
                    F_t[i].append(np.mean([scipy.linalg.norm(prec-dg_opt1.theta[k], ord = 'fro')/scipy.linalg.norm(prec, ord = 'fro') for k in range(len(dg_opt1.theta))]))
                    l1_t[i].append(np.mean([scipy.linalg.norm(prec-dg_opt1.theta[k], ord = 1)/scipy.linalg.norm(prec, ord = 1) for k in range(len(dg_opt1.theta))]))
                    f1_t[i].append(np.mean([calc_f1(prec, dg_opt1.theta[k]) for k in range(len(dg_opt1.theta))]))
                    fro_norm_t[i].append(dg_opt1.fro_norm)
                    its_t[i].append(dg_opt1.iteration)


                    start = time.time()
                    dg_opt_n = dg.dygl_parallel(obs_per_graph = obs_per_graph, max_iter = 20000, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
                    dg_opt_n.fit(X, nr_workers=10, temporal_penalty="element-wise", lik_type="gaussian", time_index=range(X.shape[0]), verbose =False)
                    elapsed = time.time()-start
                    dens_n[i].append(nx.density(G))
                    time_n[i].append(elapsed)
                    theta_n[i].append(dg_opt_n.theta)

                    zo_n[i].append(np.mean([calc_one_zero_error(prec, dg_opt_n.theta[k]) for k in range(len(dg_opt_n.theta))]))
                    F_n[i].append(np.mean([scipy.linalg.norm(prec-dg_opt_n.theta[k], ord = 'fro')/scipy.linalg.norm(prec, ord = 'fro') for k in range(len(dg_opt_n.theta))]))
                    l1_n[i].append(np.mean([scipy.linalg.norm(prec-dg_opt_n.theta[k], ord = 1)/scipy.linalg.norm(prec, ord = 1) for k in range(len(dg_opt_n.theta))]))
                    f1_n[i].append(np.mean([calc_f1(prec, dg_opt_n.theta[k]) for k in range(len(dg_opt_n.theta))]))
                    fro_norm_n[i].append(dg_opt_n.fro_norm)
                    its_n[i].append(dg_opt_n.iteration)



                    pbar.set_description(f"s {s}, a {np.round(alpha, 2)}, a {np.round(kappa, 2)}")
                    pbar.update()

                out_dict = {'nr_obs_per_graph':obs_per_graph, 'n':n, 'temporal_penalty':'global-reconstruction', 'prec':prec,
                'density_t':dens_t, 'tol':tol, 'time_t':time_t, 'zo_t':zo_t, 'F_t':F_t, 'l1_t':l1_t, 'theta_t':theta_t, 'f1_t':f1_t,
                'density_n':dens_n,  'time_n':time_n, 'zo_n':zo_n, 'F_n':F_n, 'l1_n':l1_n, 'theta_n':theta_n, 'f1_n':f1_n,
                'alpha':alphas, 'kappa':kappas, 'max_iter':5000} 

                # import pickle
                with open(f'data/distributions/mvt_gaussian_search_d_{d}_n_{n}_s_{s}_nu_{nu}_erdos.pkl', 'wb') as handle:
                    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)