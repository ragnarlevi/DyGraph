
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t, chi2

import networkx as nx
import yfinance as yf
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import port_measures as pm
import tqdm
import pickle
from collections import defaultdict
import scipy.integrate as integrate
from sklearn.covariance import LedoitWolf
from multiprocessing.pool import Pool
from pmdarima.arima import auto_arima
from arch import arch_model

def log_lik(mean,cov, X, liktype, nu = None, prec = None, gamma = None, n = 10):

    if np.isin(liktype, ['gaussian', 'LedoitWolf', 'covariance']):
        lik = np.sum(multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True))
    elif liktype == "t":
        lik = np.sum(multivariate_t.logpdf(X,loc = mean, shape=cov, df = nu))
    elif np.isin(liktype, ("skew-group-t", "group-t")):
        lik  = 0.0
        #print(X)
        for i in range(X.shape[0]):
            lik += np.log(dg.generalized_skew_t( X[i], prec, nu = nu, gamma = gamma, n = n))
    else:
        assert False, "likelihood not correct"

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
    T_vec = _T(u,nu)


    A = 1/np.sqrt(T_vec)
    return A[i]*A[j]


integrand = np.vectorize(integrand,excluded = [1,2,3])


def C_group(d, nu,m):
    
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



def integrand2(u,nu,i,j):

    T_vec = _T(u,nu)
    e_val = nu/(nu-2)


    A = 1/T_vec
    return (A[i]-e_val[i])*(A[j]-e_val[j])



integrand2 = np.vectorize(integrand2,excluded = [1,2,3])


def C_skew(d, nu,m):
    
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = (d,d))

    for i in range(D.shape[0]):
        for j in range(i, D.shape[0]):
            combintaion = ''.join(sorted(str(m[i])+str(m[j])))
            if combination_calculated[combintaion] is None:
                D[i,j] =  integrate.quad(integrand2, 0, 1 ,args = (nu,i,j))[0]
                combination_calculated[combintaion] = D[i,j]
            else:
                D[i,j] = combination_calculated[combintaion]

    return np.triu(D,0) + np.triu(D,1).T


def integrand3(u,nu,i,j, gamma):

    T_vec = _T(u,nu)
    e_val = nu/(nu-2)


    A = 1/T_vec
    A2 = 1/np.sqrt(T_vec)
    return A[i]*gamma[i]*A2[j]-e_val[i]*gamma[i]*A2[j]



integrand3 = np.vectorize(integrand3,excluded = [1,2,3,4])


def C_cov(d, nu,m, gamma):
    
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = (d,d))

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):

            D[i,j] =  integrate.quad(integrand3, 0, 1 ,args = (nu,i,j,gamma))[0]


    return D


def run(lik_type, asset_type, arima_fit = False, garch_fit = False):
    
    # parameters
    obs_per_graph = 500
    nr_quad = 10
    l = 20

    # data
    # 0 ind, energy, mat, 
    # 1 consumer, communication
    # 2 fin. real, health, tech
    if asset_type == 'ind':
        with open(f'data/case_study_etf/raw_30_assets.pkl', 'rb') as handle:
            data = pickle.load(handle)

        ticker_list = data['ticker_list']
        #log_returns = data['log_returns']
        log_returns_scaled = data['log_returns_scaled']
        price = data['price']
        groups = data['groups']

        name = f'{lik_type}_nr_quad_{nr_quad}_ind_30'
    if asset_type == 'ind100':
        with open(f'data/case_study_etf/raw_ind_100.pkl', 'rb') as handle:
            data = pickle.load(handle)

        ticker_list = data['ticker_list']
        #log_returns = data['log_returns']
        log_returns_scaled = data['log_returns_scaled']
        price = data['price']
        groups = data['groups']

        name = f'{lik_type}_nr_quad_{nr_quad}_ind_100'
    else:
        with open(f'data/case_study_etf/raw_etf.pkl', 'rb') as handle:
            data = pickle.load(handle)

        ticker_list = data['ticker_list']
        #log_returns = data['log_returns']
        log_returns_scaled = data['log_returns_scaled']
        price = data['price']
        groups = data['groups']
        name = f'{lik_type}_nr_quad_{nr_quad}_{asset_type}'

    if np.isin(lik_type, ['group-t', 'skew-group-t']):
        max_iter = 30
    else:
        max_iter = 500

    # 0 util, energy, materials, industrials
    # 1 communication, conusmer, consumer
    # health, real, fin. TECH
    
    if arima_fit:
        name = name + "_arima"
    if garch_fit:
         name = name + "_garch"


    alphas = np.array([0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4])
    time_index = range(500, 1580, l)
    tol = 1e-6
  
    pbar = tqdm.tqdm(total = len(time_index), position=1)

    sharpes_s = {i: [] for i in range(len(alphas))}
    sharpes_m = {i: [] for i in range(len(alphas))}
    ebics = {i: [] for i in range(len(alphas))}
    mdds_s = {i: [] for i in range(len(alphas))}
    mdds_m = {i: [] for i in range(len(alphas))}
    thetas= {i: [] for i in range(len(alphas))}
    Ss= {i: [] for i in range(len(alphas))}
    Cs= {i: [] for i in range(len(alphas))}
    gammas = {i: [] for i in range(len(alphas))}
    nus = {i: [] for i in range(len(alphas))}
    fro_norms = {i: [] for i in range(len(alphas))}
    ws_s = {i: [] for i in range(len(alphas))}
    ws_m = {i: [] for i in range(len(alphas))}
    mus = {i: [] for i in range(len(alphas))}
    mus_s = {i: [] for i in range(len(alphas))}
    mus_m = {i: [] for i in range(len(alphas))}
    vars_s = {i: [] for i in range(len(alphas))}
    vars_m = {i: [] for i in range(len(alphas))}
    rs_s = {i: [] for i in range(len(alphas))}
    rs_m = {i: [] for i in range(len(alphas))}
    omegas_s = {i: [] for i in range(len(alphas))}
    omegas_m = {i: [] for i in range(len(alphas))}
    port_price_s = {i: [] for i in range(len(alphas))}
    port_price_m = {i: [] for i in range(len(alphas))}
    sigmas_s = {i: [] for i in range(len(alphas))}
    sigmas_m = {i: [] for i in range(len(alphas))}
    divs_s = {i: [] for i in range(len(alphas))}
    divs_m = {i: [] for i in range(len(alphas))}

    port_price_u = {i: [] for i in range(len(alphas))}
    sharpes_u = {i: [] for i in range(len(alphas))}
    mdds_u = {i: [] for i in range(len(alphas))}
    omegas_u = {i: [] for i in range(len(alphas))}
    ws_u = {i: [] for i in range(len(alphas))}
    mus_u = {i: [] for i in range(len(alphas))}
    vars_u = {i: [] for i in range(len(alphas))}
    rs_u = {i: [] for i in range(len(alphas))}
    sigmas_u = {i: [] for i in range(len(alphas))}
    divs_u = {i: [] for i in range(len(alphas))}


    time_forecast = {i: [] for i in range(len(alphas))}
    
    likelihoods = {i: [] for i in range(len(alphas))}
    future_likelihoods = {i: [] for i in range(len(alphas))}
    AIC = {i: [] for i in range(len(alphas))}
    future_AIC = {i: [] for i in range(len(alphas))}
    BIC = {i: [] for i in range(len(alphas))}
    future_BIC = {i: [] for i in range(len(alphas))}
    nr_params = {i: [] for i in range(len(alphas))}



    if np.isin(lik_type, ['group-t', 'skew-group-t']):
        with open(f'data/case_study_etf/t_nr_quad_5_etf_static_l_{l}.pkl', 'rb') as handle:
            t_port = pickle.load(handle)
        theta_init = t_port['thetas'][4][0]
    else:
        theta_init = None

    
    pbar = tqdm.tqdm(total = len(time_index)*len(alphas), position=1)
    for alpha_cnt, alpha in enumerate(alphas):

        for i in time_index:
            lwr = np.max((i-500,0))
            nr_graphs = int(np.ceil((i-lwr-obs_per_graph)/l +1))
            
            if theta_init is not None:
                if nr_graphs < len(theta_init):
                    theta_init = theta_init[len(theta_init)-nr_graphs:]


            pbar.set_description(f"i {i}, alpha {alpha}")
            mu = np.array(np.mean(log_returns_scaled.iloc[lwr:i],axis = 0))

            if arima_fit:
                X = []
                arimas = []
                for s in range(log_returns_scaled.shape[1]):

                    out_arima = auto_arima(np.array(log_returns_scaled.iloc[lwr:i,s]-mu[s]), seasonal=False)
                    X.append(out_arima.resid())
                    arimas.append(out_arima)
                
                X = np.array(X).T
            elif garch_fit:
                X = []
                arimas = []
                mu = []
                for s in range(log_returns_scaled.shape[1]):

                    returns = np.array(log_returns_scaled.iloc[lwr:i,s])
                    am = arch_model(returns, vol = "Garch", p = 1, o = 0, q = 1, dist = 'studentst',mean = 'Constant')
                    res = am.fit()
                    X.append(res.resid)
                    arimas.append(res)  
                    mu.append(res.params['mu'])

                X = np.array(X).T
                mu = np.array(mu)

            else:
                X = np.array(log_returns_scaled.iloc[lwr:i]-mu)

            if lik_type == 'covariance':
                S = np.cov(X.T)
                


                precision_matrix = np.linalg.pinv(S) 
                gamma = None
                nu = None
                fro_norm = None
                theta = np.array([precision_matrix.copy()])
                C = 1

            elif lik_type == 'LedoitWolf':
                cov = LedoitWolf().fit(X)
                S = cov.covariance_
                


                precision_matrix = np.linalg.inv(S) 
                gamma = None
                nu = None
                fro_norm = None
                theta = np.array([precision_matrix.copy()])
                C = 1
                
            else:
                dg_opt = dg.sgl_outer_em(max_iter = max_iter, lamda = obs_per_graph*alpha, tol = tol)
                dg_opt.fit(X,  lik_type=lik_type, nu = None, verbose=True, nr_workers = 12,  max_admm_iter = 100, groups = groups, nr_quad = nr_quad, theta_init=theta_init)
                
                
                # get precision/covariance
                precision_matrix = dg_opt.theta[-1].copy()
                precision_matrix[np.abs(precision_matrix)<1e-5]= 0.0
                S = np.linalg.inv(precision_matrix)
                gamma = dg_opt.gamma[-1].copy()
                print(gamma)

                if lik_type == 't':
                    C = (dg_opt.nu[-1]/(dg_opt.nu[-1]-2))
                    S = C*S
                elif lik_type == 'group-t':
                    C = C_group(len(ticker_list), dg_opt.nu[-1], groups)
                    S = S*C
                elif lik_type == "skew-group-t":
                    C = C_group(len(ticker_list), dg_opt.nu[-1], groups)
                    skew_C = C_skew(len(ticker_list), dg_opt.nu[-1], groups)
                    cov_C = C_cov(len(ticker_list), dg_opt.nu[-1], groups, gamma)
                    S = S*C + np.multiply(skew_C, np.outer(gamma, gamma)) +  cov_C + cov_C.T
                else:
                    C = 1

                nu = dg_opt.nu[-1]
                fro_norm = dg_opt.fro_norm
                theta = dg_opt.theta.copy()

                # Update precision matrix 
                precision_matrix = np.linalg.inv(S) 

            
            # portfolio weights sharpe
            w_s, mu_s, var_s = pm.portfolio_opt(S, precision_matrix, mu, X+mu, type = 'sharpe', w_fix=True)
            portfolio_s = np.dot(price.iloc[i:i+l],w_s) # np.dot(np.array(price.iloc[i:i + l]),w_s)
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
            divs_s[alpha_cnt].append(pm.div_ratio(w_s, S))

            # portfolio weights minimum variance
            w_m, mu_m, var_m = pm.portfolio_opt(S,precision_matrix, mu, X+mu, type = 'gmv', w_fix=True)

            portfolio_m = np.dot(price.iloc[i:i+l],w_m) # np.dot(price.iloc[i:i + l],w_m)
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
            divs_m[alpha_cnt].append(pm.div_ratio(w_m, S))

            w_u, mu_u, var_u = pm.portfolio_opt(S,precision_matrix, mu,  X+mu, type = 'uniform', w_fix=True)
            portfolio_u = np.dot(price.iloc[i:i+l],w_u) #np.dot(price.iloc[i:i + l],w_u)
            port_price_u[alpha_cnt].append(portfolio_u)
            log_returns_u = np.array(100*np.log(1+pd.DataFrame(portfolio_u).pct_change()).dropna())
            r_u = (portfolio_u[-1]-portfolio_u[0])/portfolio_u[0]
            sigma_u = np.std(log_returns_u)
            sharpe_u = pm.sharpe(r_u,sigma_u)
            sharpes_u[alpha_cnt].append(sharpe_u)
            mdds_u[alpha_cnt].append(pm.max_drawdown(portfolio_u))
            omegas_u[alpha_cnt].append(pm.omega(np.squeeze(log_returns_u)))
            ws_u[alpha_cnt].append(w_u)
            mus_u[alpha_cnt].append(mu_u)
            vars_u[alpha_cnt].append(var_u)
            rs_u[alpha_cnt].append(r_u)
            sigmas_u[alpha_cnt].append(sigma_u)
            divs_u[alpha_cnt].append(pm.div_ratio(w_u, S))

            

            # add stuff independent of portfolio
            time_forecast[alpha_cnt].append(price.index[i:i + l])
            # ebics[alpha_cnt].append(ebic(np.zeros(S.shape[0]),S, precision_matrix, log_returns_scaled, liktype=lik_type, nu = dg_opt.nu, beta = 0.5))
            thetas[alpha_cnt].append(theta)
            Ss[alpha_cnt].append(S)
            Cs[alpha_cnt].append(C)
            gammas[alpha_cnt].append(gamma)
            nus[alpha_cnt].append(nu)
            fro_norms[alpha_cnt].append(fro_norm)
            mus[alpha_cnt].append(mu.copy())


            likelihoods[alpha_cnt].append(log_lik(np.zeros(theta[-1].shape[1]) ,
                                                  np.linalg.inv(theta[-1]), 
                                                  np.array(X-mu), 
                                                  liktype = lik_type, 
                                                  nu = nu,
                                                  prec = theta[-1].copy(),
                                                  gamma = gamma,
                                                  n = 10))
            
            if arima_fit:
                next_x = []
                for s in range(log_returns_scaled.shape[1]):
                    next_x.append(arimas[s].predict(l))
                next_x = np.array(next_x).T
                next_x = next_x - np.array(log_returns_scaled.iloc[i:i+l]-mu)
            else:
                next_x = np.array(log_returns_scaled.iloc[i:i+l]-mu)

            future_likelihoods[alpha_cnt].append(log_lik(np.zeros(theta[-1].shape[1]) ,
                                                  np.linalg.inv(theta[-1]), 
                                                  next_x, 
                                                  liktype = lik_type, 
                                                  nu = nu,
                                                  prec = theta[-1].copy(),
                                                  gamma = gamma,
                                                  n = 10))


            theta_t = theta[-1].copy()
            theta_t[np.abs(theta_t)<1e-2] = 0
            #print(theta_t)
            if lik_type == 't':
                nr_params[alpha_cnt].append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + 1.0)
            elif lik_type == 'group-t':
                nr_params[alpha_cnt].append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + float(len(nu)))
            elif lik_type == 'skew-group-t':
                nr_params[alpha_cnt].append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0) + float(len(nu)) + float(theta_t.shape[0]))
            else:
                nr_params[alpha_cnt].append(np.sum(theta_t[np.triu_indices(theta_t.shape[0],1)] != 0))
            #print(nr_params[alpha_cnt])
            #print(nr_params[alpha_cnt][-1])

            AIC[alpha_cnt].append(2*(nr_params[alpha_cnt][-1] - likelihoods[alpha_cnt][-1]))
            future_AIC[alpha_cnt].append(2*(nr_params[alpha_cnt][-1] - future_likelihoods[alpha_cnt][-1]))
            BIC[alpha_cnt].append((nr_params[alpha_cnt][-1]*np.log(500) - 2*likelihoods[alpha_cnt][-1]))
            future_BIC[alpha_cnt].append((nr_params[alpha_cnt][-1]*np.log(20) - 2*future_likelihoods[alpha_cnt][-1]))

            # Guess next theta
            if lik_type in ('groupt-t', 'skew-group-t', 't', 'gaussian'):
                theta_init = dg_opt.theta.copy()
                # theta_init = np.vstack((theta_init, [theta_init[-1]]))
            pbar.update()


            out_dict = {'alphas':alphas, 'time_index':time_index, 'time_change':price.index[time_index], 'time_forecast':time_forecast, 'ticker_list':ticker_list, 
                        'groups':groups, 'future_likelihoods':future_likelihoods, 'nr_params':nr_params, 'likelihoods':likelihoods,
                        'AIC':AIC, 'future_AIC':future_AIC, 'BIC':BIC, 'future_BIC':future_BIC, 'nr_quad':nr_quad,
                        'price':price, 'X':log_returns_scaled, 'l':l, 'obs_per_graph':obs_per_graph, 'gammas':gammas, 'Ss':Ss, 'Cs':Cs,
                        'tol':tol, 'max_iter':max_iter,'ebics':ebics,'thetas':thetas, 'nus':nus, 'fro_norms':fro_norms,'mus':mus,
                        'sharpes_s':sharpes_s,  'mdds_s':mdds_s,   'ws_s':ws_s, 'mus_s':mus_s, 'vars_s':vars_s, 'rs_s':rs_s, 
                        'omegas_s':omegas_s,'port_price_s':port_price_s, 'sigmas_s':sigmas_s , 'divs_s':divs_s, 
                        'sharpes_m':sharpes_m,  'mdds_m':mdds_m,   'ws_m':ws_m, 'mus_m':mus_m, 'vars_m':vars_m, 'rs_m':rs_m, 
                        'omegas_m':omegas_m,'port_price_m':port_price_m, 'sigmas_m':sigmas_m , 'divs_m':divs_m,
                        'sharpes_u':sharpes_u,  'mdds_u':mdds_u,   'ws_u':ws_u, 'mus_u':mus_u, 'vars_u':vars_u, 'rs_u':rs_u, 
                        'omegas_u':omegas_u,'port_price_u':port_price_u, 'sigmas_u':sigmas_u, 'divs_u':divs_u}
            
            with open(f'data/case_study_etf/{name}_static_l_{l}_outer_em_no_init.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # pbar.close()
        # if lik_type in ('groupt-t', 'skew-group-t', 't', 'gaussian'):
        #     theta_init = dg_opt.theta[:20].copy()

if __name__ == "__main__":

    # run("skew-group-t", "etf")
    #run("gaussian", "etf",  arima_fit=False, garch_fit=True)
    # run("t", "etf",  arima_fit=False, garch_fit=True)
    #run("skew-group-t", "etf",  arima_fit=False, garch_fit=False)
    #run("gaussian", "etf",  arima_fit=True)
    #run("t", "etf",  arima_fit=True)
    run("LedoitWolf", "etf")
    run("covariance", "etf")
    # run("skew-group-t", "etf")


