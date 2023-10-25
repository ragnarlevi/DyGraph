

import pandas as pd
import numpy as np
import sys



sys.path.insert(0, 'C:/Users/User/Code/DyGraph')
sys.path.insert(0, 'C:/Users/User/Code/DyGraph/src')
import CovReg as cr
import tqdm

from scipy.optimize import minimize
from scipy.stats import t as student_t

from sklearn import linear_model


def calc_shapley_value(B,X):
    Sigma = np.cov(X.T)
    val_y = np.dot(B, Sigma).dot(B.T)

    shapleys = dict()
    d = val_y.shape[0]
    index = np.arange(X.shape[1])

    for i in range(d):
        for j in range(i,d):
            shapleys[str(i)+','+ str(j)] = []
            for k in range(X.shape[1]):
                t1 = B[i,k]*B[j,k]*Sigma[k,k]
                t2 = 0.5*np.sum(B[i,index != k]*B[j,k]*Sigma[k,index != k])
                t3 = 0.5*np.sum(B[j,index != k]*B[i,k]*Sigma[k,index != k])
                shapleys[str(i)+','+ str(j)].append(t1+t2+t3)

    return shapleys





if __name__ == '__main__':

    tol = 1e-6
    max_iter = 2000
    nu = 5
    scale_t = 10
    # test parameters
    ns = [1000]
    rs = [5, 10, 20, 30, 40, 50, 100]
    ds = [10]
    alphas = np.concatenate(([0], np.logspace(-5,-3, 70)))

    rnd = np.random.RandomState(42)


            

    # Storetrue data and coef
    B_dict_true = dict()
    A_dict_true = dict()
    Ys_dict_cov = dict()
    #Ys_dict_meancov = dict()


    X = rnd.normal(loc = 0, scale = 1, size = (np.max(ns),np.max(rs)))


    # Generate coef matrix
    for r in rs:
        for d in ds:
            B_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            B_true[np.abs(B_true)<0.7] = 0
            B_dict_true[str(r) + '_'+str(d)] = B_true
            A_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            A_true = A_true*(rnd.uniform(size = (d,r) ) <0.6)
            A_dict_true[str(r) + '_'+str(d)] = A_true

    # Generate observations

    st_dist = student_t(df = nu, scale = scale_t)
    st_dist.random_state = np.random.RandomState(5)
    for r in rs:
        for d in ds:
            B_tmp = B_dict_true[str(r) + '_'+str(d)].copy()
            A_tmp = A_dict_true[str(r) + '_'+str(d)].copy()

            X_tmp = X[:,:r]
            gamma = rnd.normal(loc = 0, scale = 1, size = (np.max(ns)))
            epsilon = st_dist.rvs(size = (np.max(ns), d))

            Y_cov = gamma[:, np.newaxis]*np.dot(X_tmp, B_tmp.T) + epsilon
            Ys_dict_cov[str(r) + '_'+str(d)] = Y_cov.copy()


            #Y_meancov = np.dot(X_tmp, A_tmp.T) + gamma[:, np.newaxis]*np.dot(X_tmp, B_tmp.T) + epsilon
            #Ys_dict_meancov[str(r) + '_'+str(d)] = Y_meancov.copy()



    def run(method):

        print(method)

        B_est = dict()
        psi_est = dict()
        marg_liks_dict = dict()
        liks_dict = dict()
        l2_dict = dict()
        nr_param_dict = dict()


        B_est_t = dict()
        psi_est_t = dict()
        marg_liks_dict_t = dict()
        liks_dict_t = dict()
        l2_dict_t = dict()
        nr_param_dict_t = dict()

        pbar = tqdm.tqdm(total = len(rs)*len(ns)*len(alphas) )

        for r in rs:
            #print(r)
            B_est[r] = dict()
            psi_est[r] = dict()

            liks_dict[r] = dict()
            l2_dict[r] = dict()
            marg_liks_dict[r] = dict()
            nr_param_dict[r] =  dict()


            B_est_t[r] = dict()
            psi_est_t[r] = dict()

            liks_dict_t[r] = dict()
            l2_dict_t[r] = dict()
            marg_liks_dict_t[r] = dict()
            nr_param_dict_t[r] =  dict()

            for n in ns:
                B_est[r][n] = []
                psi_est[r][n] = []

                liks_dict[r][n] = []
                l2_dict[r][n] = []
                marg_liks_dict[r][n] = []
                nr_param_dict[r][n] =  []

                B_est_t[r][n] = []
                psi_est_t[r][n] = []

                liks_dict_t[r][n] = []
                l2_dict_t[r][n] = []
                marg_liks_dict_t[r][n] = []
                nr_param_dict_t[r][n] =  []



                for a in range(len(alphas)):
                    #psi = Psi_dict_cov[str(r) + '_'+str(d)][k][a]
                    #B_tmp = B_dict_cov[str(r) + '_'+str(d)][k][a]
                    #nr_param.append(np.sum(np.abs(B_tmp)>1e-3))

                    x = X[:n,:r].copy()
                    y = Ys_dict_cov[str(r) + '_'+str(d)][:n].copy()
                    # if a == 0:
                    #     C_init = 'cole'
                    # else:
                    #     C_init = B_direct_est.copy()

                    C_init = 'cole'

                    # secant
                    cov_g = cr.CovReg( Y = y, alpha = alphas[a], max_iter = max_iter, tol = tol, method = method)
                    cov_g.fit_hoff(X2 = x, verbose = False, C_init = C_init, error = 'gaussian')
                    
                    try:
                        ml = cov_g.marg_lik(X2 = x, error = 'gaussian')
                    except:
                        ml = np.nan
                    try:
                        l = cov_g.likelihood(X2 = x, error = 'gaussian')
                    except:
                        l = np.nan
                    try:
                        l2 = cov_g.l2(X2=x)
                    except:
                        l2 = np.nan
                    try:
                        npara = cov_g.nr_params()
                    except:
                        npara = np.nan


                    # if a == 0:
                    #     B_direct_est = cov.B.copy()

                    marg_liks_dict[r][n].append(ml)
                    liks_dict[r][n].append(l)
                    l2_dict[r][n].append(l2)
                    nr_param_dict[r][n].append(npara)
                    B_est[r][n].append(cov_g.B.copy())
                    psi_est[r][n].append(cov_g.Psi.copy())



                    #cov_t = cr.CovReg( Y = y, alpha = alphas[a], max_iter = max_iter, tol = tol, method = method)
                    #cov_t.fit_hoff_b_only(X2 = x, verbose = False, C_init = C_init, error = 't', nu = nu)
                    cov_t = cr.CovReg( Y = y, alpha = alphas[a], max_iter = max_iter, tol = tol, method = method)
                    cov_t.fit_hoff(X2 = x, nu = nu, verbose = False, C_init = C_init, error = 't', sample = 'mode')
                    try:
                        ml = cov_t.marg_lik(X2 = x, error = 't')
                    except:
                        ml = np.nan
                    try:
                        l = cov_t.likelihood(X2 = x, error = 't')
                    except:
                        l = np.nan
                    try:
                        l2 = cov_t.l2(X2=x)
                    except:
                        l2 = np.nan
                    try:
                        npara = cov_t.nr_params()
                    except:
                        npara = np.nan


                    # if a == 0:
                    #     B_direct_est = cov.B.copy()

                    marg_liks_dict_t[r][n].append(ml)
                    liks_dict_t[r][n].append(l)
                    l2_dict_t[r][n].append(l2)
                    nr_param_dict_t[r][n].append(npara)
                    B_est_t[r][n].append(cov_t.B.copy())
                    psi_est_t[r][n].append(cov_t.Psi.copy())



                    pbar.set_description(f"{r} {n} {np.round(ml,2)} {npara}")
                    pbar.update()


            out_dict = {'marg_liks_dict':marg_liks_dict, 'liks_dict':liks_dict, 'l2_dict':l2_dict, 'psi_est':psi_est,
                        'nr_param_dict':nr_param_dict, 'B_est':B_est, 
                        'marg_liks_dict_t':marg_liks_dict_t, 'liks_dict_t':liks_dict_t, 'l2_dict_t':l2_dict_t, 'psi_est_t':psi_est_t,
                        'nr_param_dict_t':nr_param_dict_t, 'B_est_t':B_est_t, 
                        
                        'B_true':B_dict_true, 'Ys_dict_cov':Ys_dict_cov, 'X':X}
            import pickle
            with open(f'CovRegressionPaper/data_sim/new_con_exp_shapley_t_{method}_nu{nu}_scale{scale_t}_cole.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pbar.close()



    run('secant')
    run('secant_psi_identity')
    #run('Lasso')
    #run('MultiTaskLasso')