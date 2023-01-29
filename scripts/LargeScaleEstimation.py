import numpy as np
import sys,os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t as mvt, norm 
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

if __name__ == '__main__':



    
    n = 1000
    # f = lambda x: -6.63299663e-06*x**2 +  7.39629630e-03*x+  2.26700337e-01
    alpha_prob = [0.03, 0.15, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99 ]

    tol = -np.inf# 1e-6
    alpha = 0.05
    kappa = 0.4
    max_iter = 1000
# "element-wise", 'global-reconstruction', 'ridge', "block-wise-reconstruction",
    for temporal_penalty in ["element-wise", 'global-reconstruction', 'ridge', "block-wise-reconstruction", "perturbed-node"]: # "element-wise", 'global-reconstruction', 'ridge',"block-wise-reconstruction",
        print("\n")

        print(temporal_penalty)
        theta_dict = {}
        prec_dict = {}
        dens = []
        time_elapsed = []

        if temporal_penalty in ["element-wise", 'global-reconstruction', 'ridge']:
            ds = [5, 10, 20, 50, 80, 100, 150, 300]
        else:
            ds = [5,10,20, 50]


        for i,d in enumerate(ds):
            print(d)

            
            prec = make_sparse_spd_matrix(d, alpha=alpha_prob[i], smallest_coef=-0.9, largest_coef=0.9, norm_diag = True, random_state = 42)
            tmp = prec.copy()
            np.fill_diagonal(tmp,0)
            G = nx.from_numpy_array(tmp)
            print(nx.density(G))
            print("\n")

            prec_dict[i] = prec


            X1 = np.random.multivariate_normal(mean = np.zeros(prec.shape[0]),cov = np.linalg.inv(prec), size = n)
            X = X1
            obs_per_graph = 100
            start = time.time()
            dg_opt1 = dg.dygl_parallel(obs_per_graph = obs_per_graph, max_iter = max_iter, lamda = obs_per_graph*alpha, kappa = obs_per_graph*kappa, tol = tol)
            dg_opt1.fit(X, nr_workers=1, temporal_penalty=temporal_penalty, lik_type="gaussian",verbose=True)
            elapsed = time.time()-start
            dens.append(nx.density(G))
            theta_dict[i] = dg_opt1.theta
            time_elapsed.append(elapsed)
        


        out_dict = {'nr_obs_per_graph':obs_per_graph, 'n':n, 'theta_dict':theta_dict, 'prec_dict':prec_dict, 'temporal_penalty':temporal_penalty, 'density':dens,
        'tol':tol, 'time_elapsed':time_elapsed, 'ds':ds, 'alpha':alpha, 'kappa':kappa, 'max_iter':max_iter, 'iter':dg_opt1.iteration, 'err':dg_opt1.fro_norm} 
        

        with open(f'data/large_scale/ls_gaussian2_{temporal_penalty}_{n}_{obs_per_graph}.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)












