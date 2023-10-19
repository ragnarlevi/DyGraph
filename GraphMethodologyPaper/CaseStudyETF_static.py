
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t, chi2

import networkx as nx

import DyGraph as dg
import port_measures as pm
import tqdm
import pickle
from collections import defaultdict
import scipy.integrate as integrate
from sklearn.covariance import LedoitWolf


def log_lik(mean,cov, X, liktype, nu = None, prec = None, gamma = None, n = 15):

    if liktype == "t":
        lik = np.sum(multivariate_t.logpdf(X,loc = mean, shape=cov, df = nu))
    elif np.isin(liktype, ("skew-group-t", "group-t")):
        lik  = 0.0
        #print(X)
        for i in range(X.shape[0]):
            lik += np.log(dg.generalized_skew_t( X[i], prec, nu = nu, gamma = gamma, n = n))
    else:
        lik = np.sum(multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True))

    return lik


def ebic(mean,cov,prec, X, liktype, nu, beta = 0.5):
    
    n = X.shape[0]
    n_edges = nx.from_numpy_matrix(prec).number_of_edges()
    
    log_lik_val = log_lik(mean,cov, X, liktype, nu)
    return -2*log_lik_val + np.log(n)*n_edges + 4*n_edges*beta*np.log(prec.shape[0])



def _T(u,nu):
    """
    Function to calcualte the generalized skew-t weights

    Parameters
    --------------------
    u: float 
        An observation from the uniform distribution
    nu: list
        A list containing the degree of freedom for each weight 

    Returns
    ---------------------
     :  np.array
        Array with possible weights/hidden RV.
    """
    return np.array([chi2.ppf(u, df = nu[i])/nu[i] for i in range(len(nu))])


def integrand(u,nu,i,j):
    """
    Integrand for element (i,j) in the matrix D_1 as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    S: np.array
        Covariance matrix
    gamma: np.array
        vector of current gamma estimate
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    i: int 
        Row of D_1 to be calcualted
    j: int
        Column of D_1 to be calulcate

    Returns
    ---------------------
     :  float
        Integrand value
    """
    T_vec = _T(u,nu)


    A = 1/np.sqrt(T_vec)
    return A[i]*A[j]


integrand = np.vectorize(integrand,excluded = [1,2,3])

def C_group(d, nu,m):
    """
    D1 EM matrix as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    theta: np.array
        Curren estimate of Precision matrix
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    m: list
        list containing the group membership of each feautre in x
    n: int
        Number of Gaussian-Qudrature terms for the integration

    Returns
    ---------------------
     : np.array
        D1 EM matrix
    """
    
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = (d,d))

    for i in range(D.shape[0]):
        for j in range(i, D.shape[0]):
            combintaion = ''.join(sorted(str(m[i])+str(m[j])))
            if combination_calculated[combintaion] is None:
                D[i,j] =  integrate.quad(integrand, 0, 1 ,args = (nu,i,j))[0]
                combination_calculated[combintaion] = D[i,j]
            else:
                D[i,j] = combination_calculated[combintaion]

    return np.triu(D,0) + np.triu(D,1).T


def run(lik_type ,obs_per_graph, asset_type):
    # parameters
    l = 60
    #base_kappa = 0.9
    nr_quad = 10


    with open(f'data/raw_etf2.pkl', 'rb') as handle:
        data = pickle.load(handle)

    ticker_list = data['ticker_list']
    log_returns_scaled = data['log_returns_scaled']
    price = data['price']
    groups = data['groups']
    name = f'{lik_type}_nr_quad_{nr_quad}_{asset_type}_k'


    

    alphas = np.array([0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2])[::-1]
    time_index = range(360,2980, l)
    tol = 1e-3
  
    pbar = tqdm.tqdm(total = len(time_index), position=1)

    sharpes_s = {i: [] for i in range(len(alphas))}
    sharpes_m = {i: [] for i in range(len(alphas))}
    sharpes_u = {i: [] for i in range(len(alphas))}
    ebics = {i: [] for i in range(len(alphas))}
    mdds_s = {i: [] for i in range(len(alphas))}
    mdds_m = {i: [] for i in range(len(alphas))}
    mdds_u = {i: [] for i in range(len(alphas))}
    thetas_dict= {i: [] for i in range(len(alphas))}
    Ss= {i: [] for i in range(len(alphas))}
    Cs= {i: [] for i in range(len(alphas))}
    gammas_dict = {i: [] for i in range(len(alphas))}
    nus_dict = {i: [] for i in range(len(alphas))}
    fro_norms = {i: [] for i in range(len(alphas))}
    ws_s = {i: [] for i in range(len(alphas))}
    ws_m = {i: [] for i in range(len(alphas))}
    ws_u = {i: [] for i in range(len(alphas))}
    mus = {i: [] for i in range(len(alphas))}
    mus_s = {i: [] for i in range(len(alphas))}
    mus_m = {i: [] for i in range(len(alphas))}
    mus_u = {i: [] for i in range(len(alphas))}
    vars_s = {i: [] for i in range(len(alphas))}
    vars_m = {i: [] for i in range(len(alphas))}
    vars_u = {i: [] for i in range(len(alphas))}
    rs_s = {i: [] for i in range(len(alphas))}
    rs_m = {i: [] for i in range(len(alphas))}
    rs_u = {i: [] for i in range(len(alphas))}
    omegas_s = {i: [] for i in range(len(alphas))}
    omegas_m = {i: [] for i in range(len(alphas))}
    omegas_u = {i: [] for i in range(len(alphas))}
    port_price_s = {i: [] for i in range(len(alphas))}
    port_price_m = {i: [] for i in range(len(alphas))}
    port_price_u = {i: [] for i in range(len(alphas))}
    sigmas_s = {i: [] for i in range(len(alphas))}
    sigmas_m = {i: [] for i in range(len(alphas))}
    time_forecast = {i: [] for i in range(len(alphas))}
    
    likelihoods = {i: [] for i in range(len(alphas))}
    future_likelihoods = {i: [] for i in range(len(alphas))}
    AIC = {i: [] for i in range(len(alphas))}
    future_AIC = {i: [] for i in range(len(alphas))}
    BIC = {i: [] for i in range(len(alphas))}
    future_BIC = {i: [] for i in range(len(alphas))}
    nr_params = {i: [] for i in range(len(alphas))}



    # if np.isin(lik_type, ['group-t', 'skew-group-t']):
    #     with open(f'data/t_nr_quad_{nr_quad}_{asset_type}_k_{kappa_const}_element-wise.pkl', 'rb') as handle:
    #         t_port = pickle.load(handle)
    #     theta_init = t_port['thetas'][0][0]
    # else:
    #     theta_init = None
    theta_init = None

    pbar = tqdm.tqdm(total = len(time_index)*len(alphas), position=1)
    for alpha_cnt, alpha in enumerate(alphas):

        for time_cnt, i in enumerate(time_index):

            # if (time_cnt == 0) & (alpha_cnt == 0):
            #     pass
            # elif (time_cnt ==0) & (alpha_cnt > 0):
            #     theta_init = thetas[alpha_cnt-1][0].copy()
            # else:
            #     nr_graphs_tmp = len(dg_opt.theta)
            #     theta_init = np.array([np.identity(thetas[alpha_cnt][time_cnt-1].shape[1]) for _ in range(nr_graphs_tmp) ])
            #     theta_init[:nr_graphs_tmp-1] = thetas[alpha_cnt][time_cnt-1][1:nr_graphs_tmp].copy()
            #     theta_init[-1] = np.linalg.inv(np.cov(np.array(log_returns_scaled[lwr:i]-mu).T) + 0.001*np.identity(thetas[alpha_cnt][time_cnt-1].shape[1]))



            lwr = np.max((i-180,0))
            # nr_graphs = int(np.ceil((i-lwr-obs_per_graph)/l +1))
            nr_graphs = 1
            
            if theta_init is not None:
                if nr_graphs < len(theta_init):
                    theta_init = theta_init[len(theta_init)-nr_graphs:]


            #kappa = 1*np.array([base_kappa**(pwr) for pwr in range(nr_graphs)])
            

            mu = np.mean(log_returns_scaled.iloc[lwr:i],axis = 0)
            pbar.set_description(f" {lik_type}   i {i}, alpha {alpha},shape {np.array(log_returns_scaled[lwr:i]-mu).shape}")
            
            if np.isin(lik_type, ['group-t', 'skew-group-t']): 
                max_iter = 30  
                dg_opt = dg.sgl_outer_em(np.array(log_returns_scaled[lwr:i]-mu),  
                                          max_iter = max_iter,
                                            lamda = alpha,  
                                            tol = tol, 
                                            lik_type = lik_type,
                                            groups = groups
                                            )
                dg_opt.fit(max_admm_iter = 20,
                           nr_workers=10,
                           p_node_tol= 1e-4, 
                           p_node_max_iter = 500, 
                           bwr_xtol = 1e-4,
                           nr_quad = nr_quad,
                           theta_init=theta_init,
                           verbose = False)    
            elif lik_type == "empirical":
                max_iter = 1 
                S = np.cov(np.array(log_returns_scaled[lwr:i]-mu).T)
            elif lik_type == "ledoitwolf":
                
                max_iter = 1
                cov = LedoitWolf().fit(np.array(log_returns_scaled[lwr:i]-mu))
                S = cov.covariance_
            else:
                max_iter = 500 
                dg_opt = dg.sgl_outer_em(np.array(log_returns_scaled[lwr:i]-mu),  
                                          max_iter = max_iter,
                                            lamda = alpha,  
                                            tol = tol, 
                                            lik_type = lik_type
                                            )
                dg_opt.fit(max_admm_iter = 20,
                           nr_workers=10,
                           p_node_tol= 1e-4, 
                           p_node_max_iter = 500, 
                           bwr_xtol = 1e-4,
                           nr_quad = nr_quad,
                           theta_init=theta_init,
                           verbose = False)                           

            # get precision/covariance
            if np.isin(lik_type, ['group-t', 'skew-group-t', 't', 'gaussian']):
                thetas = dg_opt.theta.copy()
                nus =  dg_opt.nu.copy()
                gammas = dg_opt.gamma.copy()
                fro_norm = dg_opt.fro_norm
            else:
                thetas = [np.linalg.pinv(S)]
                nus = [1000]
                gammas = [1]
                fro_norm = 0

            assert len(thetas) == 1, "thetas á að vera einn"

            precision_matrix = thetas[-1].copy()
            precision_matrix[np.abs(precision_matrix)<1e-5]= 0.0
            S = np.linalg.inv(precision_matrix)
            if lik_type == 't':
                C = (dg_opt.nu[-1]/(dg_opt.nu[-1]-2))
                S = C*S
            elif lik_type == 'group-t':
                C = C_group(len(ticker_list), dg_opt.nu[-1], groups)
                S = S*C
            else:
                C = 1

            # Update precision matrix 
            precision_matrix = np.linalg.inv(S) 

            # portfolio weights sharpe
            w_s, mu_s, var_s = pm.portfolio_opt(S,precision_matrix, mu, log_returns_scaled[lwr:i], type = 'sharpe')

            portfolio_s = np.dot(price.iloc[i:i + l],w_s)
            port_price_s[alpha_cnt].append(portfolio_s)
            log_returns_s = np.array(100*np.log(1+pd.DataFrame(portfolio_s).pct_change()).dropna())
            r_s = (portfolio_s[-1]-portfolio_s[0])/portfolio_s[0]
            sigma_s = np.std(log_returns_s)
            sharpe_s = pm.sharpe(r_s,sigma_s)
            sharpes_s[alpha_cnt].append(sharpe_s)
            mdds_s[alpha_cnt].append(pm.max_drawdown(portfolio_s))
            omegas_s[alpha_cnt].append(pm.omega(np.squeeze(log_returns_s)))
            ws_s[alpha_cnt].append(w_s)
            mus_s[alpha_cnt].append(mu_s)
            vars_s[alpha_cnt].append(var_s)
            rs_s[alpha_cnt].append(r_s)
            sigmas_s[alpha_cnt].append(sigma_s)

            # portfolio weights minimum variance
            w_m, mu_m, var_m = pm.portfolio_opt(S,precision_matrix, mu, log_returns_scaled[lwr:i], type = 'gmv')

            portfolio_m = np.dot(price.iloc[i:i + l],w_m)
            port_price_m[alpha_cnt].append(portfolio_m)
            log_returns_m = np.array(100*np.log(1+pd.DataFrame(portfolio_m).pct_change()).dropna())

            r_m = (portfolio_m[-1]-portfolio_m[0])/portfolio_m[0]
            sigma_m = np.std(log_returns_m)
            sharpe_m = pm.sharpe(r_m,sigma_m)
            sharpes_m[alpha_cnt].append(sharpe_m)
            mdds_m[alpha_cnt].append(pm.max_drawdown(portfolio_m))
            omegas_m[alpha_cnt].append(pm.omega(np.squeeze(log_returns_m)))
            ws_m[alpha_cnt].append(w_m)
            mus_m[alpha_cnt].append(mu_m)
            vars_m[alpha_cnt].append(var_m)
            rs_m[alpha_cnt].append(r_m)
            sigmas_m[alpha_cnt].append(sigma_m)

            # add stuff independent of portfolio
            time_forecast[alpha_cnt].append(price.index[i:i + l])
            thetas_dict[alpha_cnt].append(thetas.copy())
            Ss[alpha_cnt].append(S)
            Cs[alpha_cnt].append(C)
            gammas_dict[alpha_cnt].append(gammas.copy())
            nus_dict[alpha_cnt].append(nus.copy())
            fro_norms[alpha_cnt].append(fro_norm)
            mus[alpha_cnt].append(mu.copy())

            # Calculate likelihood
            X =np.array(log_returns_scaled)[lwr:i]
            lik_tmp = []
            w = obs_per_graph
            for j in range(len(thetas)):
                if lik_type == 't':
                    cov_tmp = (nus[j]/(nus[j]-2))*np.linalg.inv(thetas[j])
                else:
                    cov_tmp = np.linalg.inv(thetas[j])
                X_tmp = X
                lik_tmp.append(log_lik(np.mean(X, axis = 0) ,
                                       cov_tmp,  # this is not used for group-t skew-t
                                       X_tmp, 
                                       liktype = lik_type, 
                                       prec =thetas[j], 
                                       nu = nus[j], 
                                       gamma = gammas[j]))
            assert len(lik_tmp) == 1
            likelihoods[alpha_cnt].append(np.array(lik_tmp))


            if lik_type == 't':
                cov_tmp = (nus[j]/(nus[j]-2))*np.linalg.inv(thetas[j])
            else:
                cov_tmp = np.linalg.inv(thetas[j])
            future_likelihoods[alpha_cnt].append(log_lik(np.zeros(thetas[-1].shape[1]) ,
                                                  cov_tmp, 
                                                  np.array(np.array(log_returns_scaled)[i:i+20]-np.array(mu)), 
                                                  liktype = lik_type, 
                                                  nu = nus[-1],
                                                  prec = thetas[-1],
                                                  gamma = gammas[-1],
                                                  n = 15))



            nr_params_tmp = []
            for iii in range(len(thetas)):
                theta_t = thetas[iii].copy()
                theta_t[np.abs(theta_t)<1e-3] = 0
                if lik_type == 't':
                    nr_params_tmp.append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + 1.0)
                elif lik_type == 'group-t':
                    nr_params_tmp.append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + 3)
                elif lik_type == 'skew-group-t':
                    nr_params_tmp.append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + 3 + float(theta_t.shape[0]))
                else:
                    nr_params_tmp.append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0))

            assert len(nr_params_tmp) == 1

            nr_params[alpha_cnt].append(nr_params_tmp)
            # print(nr_params[alpha_cnt])

            AIC[alpha_cnt].append(2*np.sum(nr_params_tmp) -2*np.sum(lik_tmp) )
            future_AIC[alpha_cnt].append(2*(nr_params_tmp[-1] - future_likelihoods[alpha_cnt][-1]))
            BIC[alpha_cnt].append(np.sum(nr_params_tmp)*np.log(180) -2*np.sum(lik_tmp) )
            future_BIC[alpha_cnt].append(nr_params_tmp[-1]*np.log(l) - 2*future_likelihoods[alpha_cnt][-1])


            # Guess next theta
            theta_init = thetas.copy()
            #theta_init = np.vstack((theta_init, [theta_init[-1]]))
            pbar.update()



            out_dict = {'alphas':alphas, 'time_index':time_index, 'time_change':price.index[time_index], 'time_forecast':time_forecast, 'ticker_list':ticker_list, 
                        'groups':groups,   'likelihoods':likelihoods, 'future_likelihoods':future_likelihoods, 'nr_params':nr_params,
                        'AIC':AIC, 'future_AIC':future_AIC, 'BIC':BIC, 'future_BIC':future_BIC,
                        'price':price, 'X':log_returns_scaled, 'l':l, 'obs_per_graph':obs_per_graph, 'gammas':gammas_dict, 'Ss':Ss, 'Cs':Cs,
                        'tol':tol, 'max_iter':max_iter,'ebics':ebics,'thetas':thetas_dict, 'nus':nus_dict, 'fro_norms':fro_norms,'mus':mus,
                        'sharpes_s':sharpes_s,  'mdds_s':mdds_s,   'ws_s':ws_s, 'mus_s':mus_s, 'vars_s':vars_s, 'rs_s':rs_s, 
                        'omegas_s':omegas_s,'port_price_s':port_price_s, 'sigmas_s':sigmas_s,
                        'sharpes_m':sharpes_m,  'mdds_m':mdds_m,   'ws_m':ws_m, 'mus_m':mus_m, 'vars_m':vars_m, 'rs_m':rs_m, 
                        'omegas_m':omegas_m,'port_price_m':port_price_m, 'sigmas_m':sigmas_m}
        
        with open(f'data_static/{name}3.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

    # run(0.1, 't', 50, 'etf', 'element-wise')
    # run(0.4, 'skew-group-t', 50, 'etf', 'element-wise')

    for n in [60*3]:
        for k in [0.01]:
            # run('gaussian' ,n, 'etf')
            # run('t' ,n, 'etf')
            # run('ledoitwolf' ,n, 'etf')
            # run('empirical' ,n, 'etf')
            run('group-t' ,n, 'etf')
            run('skew-group-t' ,n, 'etf')






   

        # pbar.close()
        # theta_init = dg_opt.theta[:20].copy()
