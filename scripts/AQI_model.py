
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




if __name__ == '__main__':
    


    obs_per_graph = 30
    lik_type='gaussian'
    tol = 1e-6

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

    kappas = np.exp(-np.linspace(3.5, 1,10))
    alphas = np.exp(-np.linspace(5, 0.5,20))
    thetas = {i:[] for i in range(len(kappas))}
    nus = {i:[] for i in range(len(kappas))}
    F_errors = {i:[] for i in range(len(kappas))}
    iters = {i:[] for i in range(len(kappas))}


    pbar = tqdm.tqdm(total = len(kappas)*len(alphas), position=1)
    theta_init = None
    for k_cnt, kappa in enumerate(kappas):
        for a_cnt, alpha in enumerate(alphas):

            pbar.set_description(f"kappa {kappa}, alpha {alpha}")


            dygl_aqi = dg.dygl_inner_em(X = ts_df_scaled, obs_per_graph=obs_per_graph, max_iter=10000, lamda= alpha, kappa=kappa, lik_type=lik_type, tol = tol)
            dygl_aqi.fit(temporal_penalty='element-wise', nr_workers=8, theta_init=theta_init)
            theta_init = dygl_aqi.theta.copy()
            thetas[k_cnt].append(dygl_aqi.theta)
            nus[k_cnt].append(dygl_aqi.nu)
            F_errors[k_cnt].append(dygl_aqi.F_error)
            iters[k_cnt].append(dygl_aqi.iteration)
    

            out_dict = {'thetas':thetas, 'nus':nus, 'F_errors':F_errors, 'alphas':alphas, 'kappas':kappas, 'obs_per_graph':obs_per_graph, 'lik_type':lik_type, 'tol':1e-6,
                        'index':ts_df.index[:2700], 'R':R, 'X':ts_df_scaled}

            with open(f'data/AQI/AQI_{lik_type}_model_ew.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


            pbar.update()

    pbar.close()

