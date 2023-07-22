
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import networkx as nx
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import port_measures as pm
import matplotlib.pyplot as plt
import tqdm
import scipy

from scipy.stats import shapiro 
from statsmodels.tsa.seasonal import STL
from pmdarima.arima import auto_arima
from scipy.stats import boxcox
import statsmodels.api as sm
import pickle


# R stuff
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
HDtest = importr('HDtest')
spdep  = importr('spdep')
from multiprocessing.pool import Pool


def my_hdtest(Z, cov_Z, Y, cov_Y):

    d = Z.shape[1]
    n1 = Z.shape[0]
    n2 = Y.shape[0]

    za = (Z - np.mean(Z,0))
    vz = np.dot((za**2).T, za**2)/n1 - (2/n1 )*np.dot(za.T, za)*cov_Z + cov_Z**2

    ya = (Y - np.mean(Y,0))
    vy = np.dot((ya**2).T, ya**2)/n2 - (2/n2 )*np.dot(ya.T, ya)*cov_Y + cov_Y**2

    CLX = np.max((cov_Z-cov_Y)**2/(vz/n1+vy/n2))

    CLX_test = CLX-(4*np.log(d)-np.log(np.log(d)))
    p_val = 1 - np.exp(-np.exp(-CLX_test/2)/(np.sqrt(8*np.pi)))
    return p_val, CLX_test




def CE_it(X,H,w, used, alpha_in_CE,d, alpha, kappa, metric, B):

  

    # bootstrap refrence
    Xs = []
    for j in range(H):
        if j == 0:
            Xs.append(X[j*w:((j+1)*w + 10)][np.random.choice(w+10, w)])
        elif j == H-1:
            Xs.append(X[(j*w-10):(j+1)*w][np.random.choice(w+10, w)])
        else:
            Xs.append(X[(j*w-5):((j+1)*w + 5)][np.random.choice(w+10, w)])

    Xs = np.vstack(Xs)


    dg_opt1 = dg.dygl_inner_em(Xs, obs_per_graph = w, max_iter = 10000, lamda = alpha, kappa =kappa, tol = 1e-4)
    dg_opt1.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    theta_reference = dg_opt1.theta
    theta_reference[np.abs(theta_reference)<1e-5] = 0


    if alpha_in_CE:
        reg_matrix = np.ones((H,d,d))*alpha
    else:
        reg_matrix = np.zeros((H,d,d))
    for h in range(H):
        np.fill_diagonal(reg_matrix[h],0)
        reg_matrix[h][np.triu_indices(d,1)] = B[h,i]*999999
        reg_matrix[h] = reg_matrix[h] + reg_matrix[h].T


    if used == 'xs_x':
        X_used = X.copy()
    elif used == 'xs_xs':
        X_used = Xs.copy()
    else:
        raise ValueError("check used argument")


        
    dg_opt2 = dg.dygl_inner_em(X_used, obs_per_graph = w, max_iter = 10000, lamda = reg_matrix, kappa =kappa, tol = 1e-6)
    dg_opt2.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    dg_opt2.theta[np.abs(dg_opt2.theta)<1e-5] = 0
    
    thetas_sim = dg_opt2.theta.copy()

    for h in range(H):
        if metric == 'cai':
            _, val1 = my_hdtest(Xs[h*w:(h+1)*w], np.linalg.inv(theta_reference[h]),X_used[h*w:(h+1)*w], np.linalg.inv(dg_opt2.theta[h]))
        elif metric == 'zero-one':
            val1 = -np.sum(np.sign(theta_reference[h][np.triu_indices(d,1)])==np.sign(dg_opt2.theta[h][np.triu_indices(d,1)]))
        #val1 = np.linalg.norm(theta_reference[0]-theta1)
        
        #_, value_vals[h,it_nr-1,i] = my_hdtest(np.vstack((X1,X2)), np.linalg.inv(0.5*(theta_reference[0]+theta_reference[1])),np.vstack((X1,X2)), np.linalg.inv(0.5*(theta1+theta2)))
        #value_vals[h,it_nr-1,i] = np.linalg.norm(theta_reference[0]+theta_reference[1] - theta1 - theta2)
    return val1, thetas_sim



def CE_glasso(X, w, N, tail_prob, nr_its, alpha, kappa, alpha_in_CE, metric, used):

    d = X.shape[1]

    nr_params = int(d*(d-1)/2)

    H = int(X.shape[0]/w)

    p = np.zeros((H, nr_its+1,nr_params))
    sigmas = np.zeros((H, nr_its+1, nr_params))

    p[:,0] = 0.5

    thetas_sim = np.zeros((N, H, d,d))


    pbar = tqdm.tqdm(total = nr_its, position=1)
    d = X.shape[1]
    nr_params = int(d*(d-1)/2)

    pbar = tqdm.tqdm(total = nr_its*N)
    pool = Pool(8)
    value_vals = np.zeros((H,nr_its,N))  # store log-likelihoods
    for it_nr in range(1,nr_its+1):
        val_func = np.zeros(H)

        # Draw bernoulli
        B = np.zeros((H,N,nr_params))
        for h in range(H):
            B[h] = scipy.stats.bernoulli.rvs(p = p[h,it_nr-1], size = (N,nr_params))   


        results = pool.starmap(CE_it, tqdm.tqdm(((X,H,w, used, alpha_in_CE,d, alpha, kappa, metric, B) for _ in range(N)), total = N, position=0))

        for result_cnt, result in enumerate(results):
            value_vals[:,it_nr-1,result_cnt] = result[0]
            thetas_sim[result_cnt] = result[1]


        # Get quantile (best value)
        for h in range(H):
            value_vals[h,it_nr-1] = -value_vals[h,it_nr-1]
            val_func[h] = np.quantile(value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])], tail_prob)
        
        # update p
        for h in range(H):
            for i in range(nr_params):
                p[h,it_nr, i] = np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])] >= val_func[h])*B[h,~np.isnan(value_vals[h,it_nr-1]),i])/np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])] >= val_func[h]))
                sigmas[h,it_nr, i] = np.sqrt(np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])] >= val_func[h])*(B[h,~np.isnan(value_vals[h,it_nr-1]),i]-p[h,it_nr-1, i])**2)/np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])]>= val_func[h])))

            s = np.exp(-2/it_nr)*0.9
            p[h,it_nr] = s*p[h,it_nr] + (1-s)*p[h,it_nr-1]


        pbar.close()

    return p, sigmas, thetas_sim




if __name__ == '__main__':
    


    obs_per_graph = 30
    lik_type='gaussian'
    tol = 1e-6
    alpha = 1
    kappa = 0.05
    N = 10000
    tail_prob = 0.9
    nr_its = 100
    alpha_in_CE = False
    metric = 'cai'
    used = 'xs_xs'


    # load and scale data
    with open(f'data/AQI/cleaned_aqi.pkl', 'rb') as handle:
        ts_df = pickle.load(handle)

    # Get lat long
    wikiurl="https://en.wikipedia.org/wiki/User:Michael_J/County_table"
    table_class="wikitable sortable jquery-tablesorter"
    response=requests.get(wikiurl)
    sites = pd.read_html(response.content)[-1]
    sites = sites.loc[(sites['State'] == 'CA') & np.isin(sites['County [2]'], ts_df.columns)]
    sites.head()


    sites['Latitude'] = sites['Latitude'].str.replace('°','')
    sites['Latitude'] = sites['Latitude'].str.replace('+','')
    sites['Latitude'] = sites['Latitude'].str.replace('–','')
    sites['Latitude'] = sites['Latitude'].astype(float)
    sites['Longitude'] = sites['Longitude'].str.replace('°','')
    sites['Longitude'] = sites['Longitude'].str.replace('+','')
    sites['Longitude'] = sites['Longitude'].str.replace('–','')
    sites['Longitude'] = sites['Longitude'].astype(float)

    import math

    def distance(origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371 # km

        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c

        return d

    
    locs = np.array(sites[['Latitude', 'Longitude']])
    d = locs.shape[0]
    R = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            R[i,j] = distance(locs[i], locs[j])

    R = R/np.max(R)
    
    scaler = StandardScaler()
    ts_df_scaled = scaler.fit_transform(ts_df)
    ts_df_scaled = ts_df_scaled[:2700]

    # spatial aalpha np.round(np.exp(-np.linspace(5, -2,40)))# 
    # normal alpha np.exp(-np.linspace(5, 0.5,20))
    reg_matrix = R*alpha




    p, sigmas, thetas_sim = CE_glasso(ts_df_scaled, obs_per_graph, N, tail_prob, nr_its, reg_matrix, kappa, alpha_in_CE, metric, used)


    out_dict = {'obs_per_graph':obs_per_graph, 'lik_type':lik_type, 'tol':1e-6,
                'index':ts_df.index[:2700], 'R':R, 'X':ts_df_scaled, 'tail_prob':tail_prob, 'N':N, 'nr_its':nr_its, 'alpha':alpha, 'kappa':kappa,
                'alpha_in_CE':alpha_in_CE, 'metric':metric, 'used':used,'p':p, 'sigmas':sigmas, 'thetas_sim':thetas_sim}

    with open(f'data/AQI/AQI_sim_{lik_type}_{alpha_in_CE}_{metric}_{used}_ew.pkl', 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



