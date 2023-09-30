

import pandas as pd
import numpy as np
import sys



sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph')
sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph/src')
import CovReg as cr
import tqdm

from scipy.optimize import minimize


from sklearn import linear_model





if __name__ == '__main__':

    tol = 1e-6
    max_iter = 1000

    # test parameters
    ns = [100, 500,1000]
    rs = [5, 10, 20, 50, 100]
    ds = [10]
    alphas = np.concatenate(([0],  np.exp(-np.linspace(7,1,30))))

    rnd = np.random.RandomState(42)


            

    # Storetrue data and coef
    B_dict_true = dict()
    A_dict_true = dict()
    Xs_dict = dict()
    Ys_dict_cov = dict()
    Ys_dict_meancov = dict()
    val_y_true = dict()

    # Store estimation
    B_dict_cov = dict()
    B_dict_cov_psi = dict()
    B_dict_meancov = dict()
    B_dict_meancov_psi = dict()
    A_dict = dict()
    A_dict_psi = dict()
    Psi_dict_meancov = dict()
    Psi_dict_cov = dict()
    al_y_true = dict()
    value_function_meancov_dict = dict()
    value_function_meancov_psi_dict = dict()
    value_function_cov_dict = dict()
    value_function_cov_psi_dict = dict()

    lik_cov_dict = dict()
    nr_param_cov_dict = dict()
    lik_cov_psi_dict = dict()
    nr_param_cov_psi_dict = dict()
    lik_meancov_dict = dict()
    nr_param_meancov_dict = dict()
    lik_meancov_psi_dict = dict()
    nr_param_meancov_psi_dict = dict()



    pbar = tqdm.tqdm(total = len(alphas)*len(ns)*len(ds)*len(rs), position = 1)
    rnd = np.random.RandomState(42)

    Xs_dict = {str(r): [] for r in rs}
    for r in rs:
        Xs_dict[str(r)] = rnd.normal(loc = 0, scale = 1, size = (np.max(ns),r))


    # Generate coef matrix
    for r in rs:
        for d in ds:
            B_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            B_true = B_true*(rnd.uniform(size = (d,r) ) <0.5)
            B_dict_true[str(r) + '_'+str(d)] = B_true
            A_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            A_true = B_true*(rnd.uniform(size = (d,r) ) <0.5)
            A_dict_true[str(r) + '_'+str(d)] = A_true

    # Generate observations

    for r in rs:
        for d in ds:
            B_tmp = B_dict_true[str(r) + '_'+str(d)].copy()
            A_tmp = A_dict_true[str(r) + '_'+str(d)].copy()

            X = Xs_dict[str(r)].copy()
            gamma = rnd.normal(loc = 0, scale = 1, size = (np.max(ns)))
            epsilon = rnd.normal(loc = 0, scale = 1, size = (np.max(ns),d))

            Y_cov = gamma[:, np.newaxis]*np.dot(X, B_tmp.T) + epsilon
            Ys_dict_cov[str(r) + '_'+str(d)] = Y_cov.copy()


            Y_meancov = np.dot(X, A_tmp.T) + gamma[:, np.newaxis]*np.dot(X, B_tmp.T) + epsilon
            Ys_dict_meancov[str(r) + '_'+str(d)] = Y_meancov.copy()

    for r in rs:
        for d in ds:

            Bs_covmean = np.zeros(shape = (len(ns), len(alphas), d, r))
            #Bs_covmean_psi = np.zeros(shape = (len(ns), len(alphas), d, r))
            Bs_cov = np.zeros(shape = (len(ns), len(alphas), d, r))
            #Bs_cov_psi = np.zeros(shape = (len(ns), len(alphas), d, r))
            As = np.zeros(shape = (len(ns), len(alphas), d, r))
            #As_psi = np.zeros(shape = (len(ns), len(alphas), d, r))
            Psis_covmean = np.zeros(shape = (len(ns), len(alphas), d, d))
            Psis_cov = np.zeros(shape = (len(ns), len(alphas), d, d))

            lik_cov = np.zeros(shape = (len(ns), len(alphas)))
            nr_param_cov = np.zeros(shape = (len(ns), len(alphas)))
            lik_cov_psi = np.zeros(shape = (len(ns), len(alphas)))
            nr_param_cov_psi = np.zeros(shape = (len(ns), len(alphas)))
            lik_meancov = np.zeros(shape = (len(ns), len(alphas)))
            nr_param_meancov = np.zeros(shape = (len(ns), len(alphas)))
            lik_meancov_psi = np.zeros(shape = (len(ns), len(alphas)))
            nr_param_meancov_psi = np.zeros(shape = (len(ns), len(alphas)))

            for n_cnt, n in enumerate(ns):
                for alpha_cnt, alpha in enumerate(alphas):
                    pbar.set_description(f"r{r} d{d} n{n} a{alpha_cnt}")

                    Y_tmp_mean_cov = Ys_dict_meancov[str(r) + '_'+str(d)][:n].copy() 
                    Y_tmp_cov = Ys_dict_cov[str(r) + '_'+str(d)][:n].copy() 
                    X_tmp = Xs_dict[str(r)][:n].copy()
                    print(X_tmp.shape)


                    # Estimate mean cov model
                    cov = cr.CovReg( Y = Y_tmp_mean_cov, alpha = alpha, max_iter = max_iter, tol = tol)
                    cov.fit_hoff(X1=X_tmp, X2 = X_tmp, verbose = False)

                    lik_meancov[n_cnt, alpha_cnt] ,nr_param_meancov[n_cnt, alpha_cnt] = cov.marg_lik(X1 = X_tmp, X2 = X_tmp)

                    Bs_covmean[n_cnt, alpha_cnt] = cov.B.copy()
                    As[n_cnt, alpha_cnt] = cov.A.copy()
                    Psis_covmean[n_cnt, alpha_cnt] = cov.Psi.copy()




                    # Estimate cov model
                    cov = cr.CovReg( Y = Y_tmp_cov, alpha = alpha, max_iter = max_iter, tol = tol)
                    cov.fit_hoff_b_only(X2 = X_tmp, verbose = False)
                    
                    lik_cov[n_cnt, alpha_cnt] ,nr_param_cov[n_cnt, alpha_cnt] = cov.marg_lik( X2 = X_tmp)


                    Psis_cov[n_cnt, alpha_cnt] = cov.Psi.copy()
                    Bs_cov[n_cnt, alpha_cnt] = cov.B.copy()

                    pbar.update()


                B_dict_cov[str(r) + '_'+str(d)] = Bs_cov.copy()
                Psi_dict_cov[str(r) + '_'+str(d)] =  Psis_cov.copy()
                lik_cov_dict[str(r) + '_'+str(d)] = lik_cov
                nr_param_cov_dict[str(r) + '_'+str(d)] = nr_param_cov



                B_dict_meancov[str(r) + '_'+str(d)] = Bs_covmean.copy()
                Psi_dict_meancov[str(r) + '_'+str(d)] = Psis_covmean.copy()
                A_dict[str(r) + '_'+str(d)] = As.copy()
                lik_meancov_dict[str(r) + '_'+str(d)] = lik_meancov
                nr_param_meancov_dict[str(r) + '_'+str(d)] = nr_param_meancov
                
    



            

                out_dict = {'alphas':alphas, 'ns':ns, 'rs':rs, 'ds':ds, 
                            'value_function_cov_dict':value_function_cov_dict, 'value_function_meancov_dict':value_function_meancov_dict, 'value_function_cov_psi_dict':value_function_cov_psi_dict, 'value_function_covmean_psi_dict':value_function_meancov_psi_dict,
                            'B_dict_cov':B_dict_cov, 'B_dict_meancov':B_dict_meancov, 'B_dict_cov_psi':B_dict_cov_psi,'B_dict_meancov_psi':B_dict_meancov_psi,
                            'Ys_dict_cov':Ys_dict_cov, 'Ys_dict_meancov':Ys_dict_meancov,
                            'B_dict_true':B_dict_true, 'A_dict_true':A_dict_true,
                            'Psi_dict_cov':Psi_dict_cov, 'Psi_dict_meancov':Psi_dict_meancov,
                            'As':A_dict, 'As_psi':A_dict_psi,
                            'Xs_dict':Xs_dict, 
                            'val_y_true':val_y_true,
                            'lik_cov_dict':lik_cov_dict, 'lik_meancov_dict':lik_meancov_dict, 'lik_cov_psi_dict':lik_cov_psi_dict, 'lik_meancov_psi_dict':lik_meancov_psi_dict,
                            'nr_param_cov_dict':nr_param_cov_dict, 'nr_param_meancov_dict':nr_param_meancov_dict, 'nr_param_cov_psi_dict':nr_param_cov_psi_dict, 'nr_param_meancov_psi_dict':nr_param_meancov_psi_dict}
                
                import pickle
                with open(f'CovRegressionPaper/data_sim/shapley_less_with_lik6.pkl', 'wb') as handle:
                    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


                    
    pbar.close()