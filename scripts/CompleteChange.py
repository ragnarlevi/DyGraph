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




def calc_one_zero_error(T,Estimate, ratio = True):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    error = np.sum(~(np.sign(T[np.triu_indices(T.shape[0], k = 1)]) == np.sign(Estimate[np.triu_indices(Estimate.shape[0], k = 1)])))
    if ratio:
        error = error/float(d*(d-1)/2)
    return error

def calc_f1(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    y_true = np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.f1_score(y_true,y_pred)

def calc_f1(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    y_true = np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.f1_score(y_true,y_pred)

def calc_precision(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    y_true = np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.precision_score(y_true,y_pred)

def calc_recall(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    y_true = np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.recall_score(y_true,y_pred)

def calc_density(prec):
    tmp = prec.copy()
    np.fill_diagonal(tmp,0)
    G = nx.from_numpy_array(tmp)
    # G = nx.fast_gnp_random_graph(300,0.3)
    return nx.density(G)

def calc_roc(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<1e-7] = 0.0
    Estimate[np.abs(Estimate)<1e-7] = 0.0
    y_true = np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.recall_score(y_true,y_pred)



from sklearn.datasets import make_sparse_spd_matrix
prec_0 = make_sparse_spd_matrix(5, alpha=0.3, smallest_coef=-0.2, largest_coef=0.8, norm_diag = True,random_state=42)
#print("precision")
#print(prec_0)
#print("Covariance")
S = np.linalg.inv(prec_0)
#print(S)




if __name__ == '__main__':

    tol = 1e-10
# "element-wise", 'global-reconstruction', 'ridge',
    for temporal_penalty in ["block-wise-reconstruction", "perturbed-node"]: #'global-reconstruction', 'ridge'
        print(temporal_penalty)

        if temporal_penalty in ["element-wise", 'global-reconstruction', 'ridge']:
            alpha = np.linspace(0.01,0.3, 31)
            kappa = np.linspace(0.01,0.3, 31)

        else:
            alpha = np.concatenate((np.linspace(0.01,0.3, 25), [0.35, 0.4, 0.5]))
            kappa = np.concatenate((np.linspace(0.01,0.3, 25), [0.35, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9,1]))

        for nr_obs_per_graph in [100]:

            rnd_state = np.random.RandomState(42)
            rnd_state2 = np.random.RandomState(1)

            n = 9*nr_obs_per_graph
            d = prec_0.shape[0]
            # Simulate 
            Xs = np.zeros((n,d))
            prec_list = [prec_0]
            is_psd = False
            cnt = 0
            while cnt*nr_obs_per_graph <n:
                if cnt > 0:
                    prec_tmp = prec_list[cnt-1].copy()

                    if cnt == 3:

                        prec_tmp = prec_tmp*1.5
                    
                    if cnt == 6:
                        prec_tmp = prec_tmp*0.6
                        prec_tmp[0,2] = 0.0
                        prec_tmp[2,0] = 0.0
                        prec_tmp[4,1] = 0.0
                        prec_tmp[1,4] = 0.0

                    
                    # if rnd_state.uniform() < 0.6:

                    #     prec_tmp = prec_tmp*1.5
                    #     np.fill_diagonal(prec_tmp, np.diag(prec_list[cnt-1]) )

                    # else:
                    #     prec_tmp = prec_tmp*0.8
                    #     np.fill_diagonal(prec_tmp, np.diag(prec_list[cnt-1]) )
                    
                    u,v = np.linalg.eigh(prec_tmp)
                    if np.any(u<0.0):
                        continue
                    prec_list.append(prec_tmp)

                Xs[cnt*nr_obs_per_graph:(cnt+1)*nr_obs_per_graph ] = rnd_state2.multivariate_normal(mean = np.zeros(d), cov = np.linalg.inv(prec_list[cnt]),size=(nr_obs_per_graph) )

                cnt +=1




            prec_list = np.array(prec_list)


            pbar = tqdm.tqdm(total = len(alpha)*len(kappa))

            theta = {i:[] for i in range(len(alpha))}
            l1_error = {i:[] for i in range(len(alpha))}
            F_error = {i:[] for i in range(len(alpha))}
            one_zero_error = {i:[] for i in range(len(alpha))}


            obs_per_graph_model = nr_obs_per_graph
            for i in range(len(alpha)):
                for j in range(len(kappa)):

                    dg_opt1 = dg.dygl_parallel(obs_per_graph = obs_per_graph_model, max_iter = 2000, lamda = obs_per_graph_model*alpha[i], kappa = obs_per_graph_model*kappa[j], tol = tol)
                    dg_opt1.fit(Xs, nr_workers=3, temporal_penalty=temporal_penalty, lik_type="gaussian", verbose=False)

                    theta[i].append(dg_opt1.theta)

                    tmp_F = []
                    tmp_zo = []
                    tmp_l1 = []

                    for k in range(int(n/obs_per_graph_model)):

                        tmp_zo.append(calc_one_zero_error(prec_list[k], dg_opt1.theta[k]))
                        tmp_F.append(scipy.linalg.norm(prec_list[k]- dg_opt1.theta[k],ord = 'fro'))
                        tmp_l1.append(scipy.linalg.norm(prec_list[k]- dg_opt1.theta[k],ord = 1))


                    F_error[i].append(np.mean(tmp_F))
                    one_zero_error[i].append(np.mean(tmp_zo))
                    l1_error[i].append(np.mean(tmp_l1))

                    pbar.set_description(f"{i} {j}")
                    pbar.update()





            out_dict = {'F_error':F_error, 'alpha':alpha, 'kappa':kappa, 'one_zero_error':one_zero_error, 'l1_error':l1_error, 'prec_list':prec_list, 'theta':theta, 
            'nr_obs_per_graph':nr_obs_per_graph, 'n':n, 'X':Xs, 'obs_per_graph_model':obs_per_graph_model, 'temporal_penalty':temporal_penalty, 'tol':tol} 
            import pickle

            with open(f'data/complete/complete_change_gaussian2_{temporal_penalty}_{n}_{obs_per_graph_model}.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)