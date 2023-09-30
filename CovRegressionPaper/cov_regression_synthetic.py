

import numpy as np
from scipy.stats import multivariate_normal, matrix_normal
import networkx as nx
import sys
# sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph')
# sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph/src')
import CovReg as cr


from sklearn.metrics.pairwise import rbf_kernel
import time
import pickle


def gen_data_graph(n, d, r, test = 'all'):
    
    rnd = np.random.RandomState(42)


    G = nx.fast_gnp_random_graph(r, 3/r,    seed = 1)

    v, u = np.linalg.eigh(nx.laplacian_matrix(G).todense())

    H = np.array(np.dot(u, np.diag(np.exp(-0.1*v))).dot(u.T))
    print(np.linalg.cond(np.dot(H,H)))


    #T  = np.random.normal(loc = 0, scale = 1, size = (n,r)) 
    T = np.linspace(1,10,n).reshape(-1,1)
    K = rbf_kernel(T,T, gamma = 1) + 0.001*np.identity(n)
    # print(K[:,0])

    omega = 1
    matrix_normal.random_state = rnd
    F_true = matrix_normal(rowcov = K, colcov = np.dot(H,H)*(1/omega), seed =1).rvs()

    scale = 1
    psi = scale*np.identity(d)

    gamma = rnd.normal(loc = 0, scale = 1, size = (n))
    epsilon= rnd.normal(loc = 0, scale = scale, size = (n,d))
    #B_true = np.random.normal(loc = 0, scale = 0, size = (d,r))
    #A_true = np.random.normal(loc = 0, scale = 0, size = (d,r))
    P = rnd.binomial(1,0.5,size = (d,r) )
    A_true = rnd.uniform(0.5,1,size = (d,r))*P + rnd.uniform(-1,-0.5,size = (d,r))*(1-P)
    B_true = rnd.uniform(-1,1,size = (d,r))
    B_true[np.abs(B_true)<0.5] = 0#*(np.random.uniform(size = (d,r) ) <0.5)
    A_true = A_true*(rnd.uniform(size = (d,r) ) <0.5)

    if np.isin(test, ('all_unknown', 'psi_known', 'all_known')):
        Y = np.dot(F_true, A_true.T) +  gamma[:, np.newaxis]*np.dot(F_true, B_true.T) + epsilon
    elif np.isin(test, ('mean_all_unknown', 'mean_psi_known', 'mean_all_known')):
        Y = np.dot(F_true, A_true.T)  + epsilon
    elif np.isin(test, ('cov_all_unknown', 'cov_psi_known', 'cov_all_known')):
        Y =gamma[:, np.newaxis]*np.dot(F_true, B_true.T) + epsilon
    else:
        raise ValueError("check test")

    return Y, F_true, A_true, B_true, K, nx.laplacian_matrix(G).todense(), H, psi





def run_gcovreg(ns, alphas, test, r, d):

        

    tol = {i:[] for i in range(len(ns))}
    As = {i:[] for i in range(len(ns))}
    Bs = {i:[] for i in range(len(ns))}
    Fs = {i:[] for i in range(len(ns))}
    Psis = {i:[] for i in range(len(ns))}
    times = {i:[] for i in range(len(ns))}
    obj = {i:[] for i in range(len(ns))}

    As_true = []
    Bs_true = []
    Fs_true = []
    Psis_true = []
    Ys_true = []



    for n_cnt, n in enumerate(ns):
        Y, F_true, A_true, B_true, K, L, H, psi = gen_data_graph(n, 6, 6, test = test)
        Ys_true.append(Y)
        Fs_true.append(F_true)
        As_true.append(A_true)
        Bs_true.append(B_true)
        Psis_true.append(psi)
        

        for a_cnt, alpha in enumerate(alphas):
            st = time.time()
            gpp = cr.GraphCovReg(Y,alpha, K = K, L = L, omega = 1, beta = 0.1, r =6, max_iter = 1000, tol =1e-6)
            if test == 'all_unknown':
                gpp.fit_ggp(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001)
            elif test == "psi_known":
                gpp.fit_ggp(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi)
            elif test == "all_known":
                gpp.fit_ggp(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi, c = np.hstack((A_true, B_true)))
            elif test == 'mean_all_unknown':
                gpp.fit_ggp_a_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001)
            elif test == "mean_psi_known":
                gpp.fit_ggp_a_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi)
            elif test == "mean_all_known":
                gpp.fit_ggp_a_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi, c = A_true)
            elif test == 'cov_all_unknown':
                gpp.fit_ggp_b_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001)
            elif test == "cov_psi_known":
                gpp.fit_ggp_b_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi)
            elif test == "cov_all_known":
                gpp.fit_ggp_b_only(F_method = 'direct', type_reg = 'Tikanov', reg = 0.001, psi = psi, c = B_true)
            else:
                raise ValueError("unknown test")
            
            et = time.time()
            tol[n_cnt].append(gpp.tol_vec)
            As[n_cnt].append(gpp.A)
            Bs[n_cnt].append(gpp.B)
            Fs[n_cnt].append(gpp.F)
            Psis[n_cnt].append(gpp.Psi)
            times[n_cnt].append(et-st)
            obj[n_cnt].append(gpp.marg_lik())


            out_dict = {'alphas':alphas, 'ns':ns, 'tol':tol,
                'Ys_true':Ys_true,'As_true':As_true,'Bs_true':Bs_true,'Psis_true':Psis_true,'Fs_true':Fs_true,
                'As':As,'Bs':Bs,'Psis':Psis,'Fs':Fs, 'times':times,
                'omega':1, 'beta':0.1, 'r':r, 'd':d}
            
            params_name = f'r{r}_d{d}'
            file = 'data/GGP/' + test + params_name + '.pkl'
            with open(file, 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':



    # Time test secant vs direct
    ns =  [100]
    alphas = np.linspace(0, 0.15, 20)


    print(np.isin('mean_all_unknown', ('mean_all_unknown', 'mean_psi_known', 'cov_all_known')))
    
    run_gcovreg(ns, alphas, test = 'all_unknown', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'psi_known', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'all_known', r = 6, d = 24)


    run_gcovreg(ns, alphas, test = 'all_unknown', r = 6, d = 6)
    run_gcovreg(ns, alphas, test = 'psi_known', r = 6, d = 6)
    run_gcovreg(ns, alphas, test = 'all_known', r = 6, d = 6)


    run_gcovreg(ns, alphas, test = 'mean_all_unknown', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'mean_psi_known', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'mean_all_known', r = 6, d = 24)

    run_gcovreg(ns, alphas, test = 'cov_all_unknown', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'cov_psi_known', r = 6, d = 24)
    run_gcovreg(ns, alphas, test = 'cov_all_known', r = 6, d = 24)







    # Y, F_true, A_true, B_true, K, L, H, psi = gen_data_graph(200, 6, 6)
    # F_error = []
    # B_error = []
    # A_error = []
    # Psi_error =  []
    # obj = []
    # times = []
    # for i in range(500):
    #     st = time.time()
    #     gpp = cr.GraphCovReg(Y,0.025, K = K, L = L, omega = 1, beta = 0.1, r =6, max_iter = 100)
    #     gpp.fit_ggp(F_method = 'direct', type_reg = 'Tikanov', reg = 0.1, F_start = np.random.uniform(-1,1, (200, 6)))
    #     et = time.time()
    #     F_error.append(np.linalg.norm(gpp.F-F_true))
    #     A_error.append(np.linalg.norm(gpp.A-A_true))
    #     B_error.append(np.linalg.norm(gpp.B-B_true))
    #     Psi_error.append(np.linalg.norm(gpp.Psi-psi))
    #     obj.append(gpp.marg_lik())
    #     times.append(et-st)


    #     out_dict = {'alphas':alphas, 'ns':n, 'tol':tol,
    #         'F_error':F_error, 'B_error':B_error, 'A_error':A_error, 'Psi_error':Psi_error, 'obj':obj,
    #         'times':times,
    #         'omega':1, 'beta':0.1, 'r':6, 'd':6}
    #     with open(f'data/covreg/multi_start_r6_d6_a025_n500.pkl', 'wb') as handle:
    #         pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




    # multi start test














