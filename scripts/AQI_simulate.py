
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import networkx as nx
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')
sys.path.insert(0, 'C:/Users/User/Code/DyGraph/src')

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




def CE_it(args):
    X,H,w, used, alpha_in_CE,d, alpha, kappa, metric, B, idx = args

  

    Xs = []
    for j in range(3):
        if j == 0:
            Xs.append(X[j*w:((j+1)*w)][np.random.choice(w, w)])
        elif j == 3-1:
            Xs.append(X[(j*w):(j+1)*w][np.random.choice(w, w)])
        else:
            Xs.append(X[(j*w):((j+1)*w)][np.random.choice(w, w)])

    Xs = np.vstack(Xs)

    dg_opt1 = dg.dygl_inner_em(Xs, obs_per_graph = w, max_iter = 10000, lamda = alpha, kappa = kappa, tol = 1e-3)
    dg_opt1.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    dg_opt1.theta[np.abs(dg_opt1.theta)<1e-3] = 0
    thetas_sim = dg_opt1.theta.copy()
    theta_reference = dg_opt1.theta

    reg_matrix = np.zeros((H,d,d))
    for h in range(H):
        np.fill_diagonal(reg_matrix[h],0)
        reg_matrix[h][np.triu_indices(d,1)] = B[h]*999999# 999999*(1-np.abs(np.sign(theta_reference[h][np.triu_indices(d,1)])))#
        reg_matrix[h] = reg_matrix[h] + reg_matrix[h].T

    #X_used = Xs.copy()
    if used == 'xs_x':
        X_used = X.copy()
    elif used == 'xs_xs':
        X_used = Xs.copy()
    else:
        raise ValueError("check used argument")

    dg_opt2 = dg.dygl_inner_em(X_used, obs_per_graph = w, max_iter = 2000, lamda = reg_matrix, kappa =kappa, tol = 1e-3)
    dg_opt2.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    dg_opt2.theta[np.abs(dg_opt2.theta)<1e-3] = 0
    particle_graph = dg_opt2.theta.copy()
    particle_graph[np.abs(particle_graph)<1e-3] = 0


    val1 = np.zeros(H)
    for h in range(H):
        if metric == 'cai':
            _, val1[h] = my_hdtest(Xs[h*w:(h+1)*w], np.linalg.inv(theta_reference[h]),X_used[h*w:(h+1)*w], np.linalg.inv(dg_opt2.theta[h]))
        elif metric == 'zero-one':
            val1[h] = -np.sum(np.sign(theta_reference[h][np.triu_indices(d,1)])==np.sign(particle_graph[h][np.triu_indices(d,1)]))
        else:
            raise ValueError("check metric argument")

    #pbar.update()


    return val1, thetas_sim, idx



def CE_glasso(X, w, N, tail_prob, nr_its, alpha, kappa, alpha_in_CE, metric, used, s, sparsity, known):

    d = X.shape[1]

    nr_params = int(d*(d-1)/2)

    H = int(X.shape[0]/w)

    p = np.zeros((H, nr_its+1,nr_params))
    sigmas = np.zeros((H, nr_its+1, nr_params))

    p[:,0] = 0.5

    thetas_sim = np.zeros((N, H, d,d))
    d = X.shape[1]
    nr_params = int(d*(d-1)/2)

    #pbar = tqdm.tqdm(total = nr_its*N)
    pool = Pool(4)
    value_vals = np.zeros((H,nr_its,N))  # store log-likelihoods
    for it_nr in range(1,nr_its+1):
        print(f'it_nr {it_nr}: {w}_{N}_{lik_type}_{alpha_in_CE}_{metric}_{used}_ew_s{s}_spar{sparsity}_tp{tail_prob}_a{alpha}_k{kappa}_ni{nr_its}_k{known}')

        # Draw bernoulli
        B = np.zeros((H,N,nr_params))
        for h in range(H):
            B[h] = scipy.stats.bernoulli.rvs(p = p[h,it_nr-1], size = (N,nr_params))   


        list_of_args1 = [X]*N
        list_of_args2 = [H]*N
        list_of_args3 = [w]*N
        list_of_args4 = [used]*N
        list_of_args5 = [alpha_in_CE]*N
        list_of_args6 = [d]*N
        list_of_args7 = [alpha]*N
        list_of_args8 = [kappa]*N
        list_of_args9 = [metric]*N
        list_of_args10 = [B[:,iii]  for iii in range(N)]
        list_of_args11 = [iii for iii in range(N)]

        #X,H,w, used, alpha_in_CE,d, alpha, kappa, metric, B, idx 
        args = zip(list_of_args1, list_of_args2, list_of_args3,
                   list_of_args4, list_of_args5, list_of_args6,
                   list_of_args7, list_of_args8, list_of_args9,
                   list_of_args10, list_of_args11)



        for result in tqdm.tqdm(pool.imap_unordered(CE_it, args), total=N):
        #for result_cnt, result in enumerate(results):
            #print(result)
            value_vals[:,it_nr-1,result[2]] = result[0]
            thetas_sim[result[2]] = result[1]

        
        # Get quantile (best value)
        val_func = np.zeros(H)
        for h in range(H):
            value_vals[h,it_nr-1] = -value_vals[h,it_nr-1]
            val_func[h] = np.quantile(value_vals[h,it_nr-1], tail_prob)
                
        # update p
        for h in range(H):
            for i in range(nr_params):
                p[h,it_nr, i] = np.sum((value_vals[h,it_nr-1] >= val_func[h])*B[h,:,i])/np.sum((value_vals[h,it_nr-1] >= val_func[h]))
                sigmas[h,it_nr, i] = np.sqrt(np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])] >= val_func[h])*(B[h,~np.isnan(value_vals[h,it_nr-1]),i]-p[h,it_nr-1, i])**2)/np.sum((value_vals[h,it_nr-1,~np.isnan(value_vals[h,it_nr-1])]>= val_func[h])))

            p[h,it_nr] = s*p[h,it_nr] + (1-s)*p[h,it_nr-1]



        out_dict = {'obs_per_graph':obs_per_graph, 'lik_type':lik_type, 'tol':1e-6,
            'X':X, 'tail_prob':tail_prob, 'N':N, 'nr_its':nr_its, 'alpha':alpha, 'kappa':kappa,
        'alpha_in_CE':alpha_in_CE, 'metric':metric, 'used':used,'p':p, 'sigmas':sigmas, 'thetas_sim':thetas_sim, 'value_vals':value_vals, 's':s, 'sparsity':sparsity}

        with open(f'data/GraphHypTest/single_sim_w{w}_{N}_{lik_type}_{alpha_in_CE}_{metric}_{used}_ew_s{s}_spar{sparsity}_tp{tail_prob}_a{alpha}_k{kappa}_ni{nr_its}_k{known}.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #pbar.close()

    return p, sigmas, thetas_sim




if __name__ == '__main__':
    




    # # load and scale data
    # with open(f'data/AQI/cleaned_aqi.pkl', 'rb') as handle:
    #     ts_df = pickle.load(handle)

    # # Get lat long
    # wikiurl="https://en.wikipedia.org/wiki/User:Michael_J/County_table"
    # table_class="wikitable sortable jquery-tablesorter"
    # response=requests.get(wikiurl)
    # sites = pd.read_html(response.content)[-1]
    # sites = sites.loc[(sites['State'] == 'CA') & np.isin(sites['County [2]'], ts_df.columns)]
    # sites.head()


    # sites['Latitude'] = sites['Latitude'].str.replace('°','')
    # sites['Latitude'] = sites['Latitude'].str.replace('+','')
    # sites['Latitude'] = sites['Latitude'].str.replace('–','')
    # sites['Latitude'] = sites['Latitude'].astype(float)
    # sites['Longitude'] = sites['Longitude'].str.replace('°','')
    # sites['Longitude'] = sites['Longitude'].str.replace('+','')
    # sites['Longitude'] = sites['Longitude'].str.replace('–','')
    # sites['Longitude'] = sites['Longitude'].astype(float)

    # import math

    # def distance(origin, destination):
    #     lat1, lon1 = origin
    #     lat2, lon2 = destination
    #     radius = 6371 # km

    #     dlat = math.radians(lat2-lat1)
    #     dlon = math.radians(lon2-lon1)
    #     a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
    #         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    #     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    #     d = radius * c

    #     return d

    
    # locs = np.array(sites[['Latitude', 'Longitude']])
    # d = locs.shape[0]
    # R = np.zeros((d, d))

    # for i in range(d):
    #     for j in range(d):
    #         R[i,j] = distance(locs[i], locs[j])

    # R = R/np.max(R)
    
    # scaler = StandardScaler()
    # ts_df_scaled = scaler.fit_transform(ts_df)
    # ts_df_scaled = ts_df_scaled[:2700]

    # spatial aalpha np.round(np.exp(-np.linspace(5, -2,40)))# 
    # normal alpha np.exp(-np.linspace(5, 0.5,20))

    obs_per_graph = 50
    lik_type='gaussian'
    tol = 1e-6
    alpha = 0.05
    kappa = 0.05
    N = 5000
    tail_prob = 0.9
    nr_its = 50
    alpha_in_CE = False
    metric = 'cai'
    used = 'xs_xs'
    reg_matrix = alpha


    def gen_local_change(d, s):

        k = d
        sparsity = s
        rnd = np.random.RandomState(42)
        while True:

            # generate the symmetric sparsity mask
            mask = rnd.uniform(size=(k,k))
            mask = mask * (mask < sparsity)
            mask[np.tril_indices(k,0)] = 0
            mask = mask + mask.T + np.identity(k)
            mask[mask > 0] = 1

            # generate the symmetric precision matrix
            A = rnd.uniform( size = (k,k))
            A[np.tril_indices(k,0)] = 0
            A = A + A.T + np.identity(k)

            # apply the reqired sparsity
            A = A * mask

            # force it to be positive definite
            l,u = np.linalg.eigh(A)
            A = A - (np.min(l)-.5) * np.identity(k)

            A2 = A.copy()
            shut_index = rnd.uniform(low =0, high = 0.5, size = A.shape)
            shut_index = shut_index +shut_index.T
            np.fill_diagonal(shut_index,1)
            shut_index = shut_index <0.3
            A2[shut_index] = 0


            for i in range(k):
                for j in range(i+1, k):
                    if rnd.uniform()<0.3:
                        A2[i,j] = rnd.uniform(low = -1, high = 1)
                        A2[j,i] = A2[i,j]


            A3 = A2.copy()
            shut_index = rnd.uniform(low =0, high = 0.5, size = A.shape)
            shut_index = shut_index +shut_index.T
            np.fill_diagonal(shut_index,1)
            shut_index = shut_index <0.3
            A3[shut_index] = 0

            for i in range(k):
                for j in range(i+1, k):
                    if rnd.uniform()<0.3:
                        A3[i,j] = rnd.uniform(low = -1, high = 1)
                        A3[j,i] = A3[i,j]



            try:
                u = np.linalg.eigvals(A)
                assert np.all(u>=0)
                u = np.linalg.eigvals(A2)
                assert np.all(u>=0)
                u = np.linalg.eigvals(A3)
                assert np.all(u>=0)
            except: 
                continue

            break




        return np.array([A, A2, A3])



    # d = 10
    # sparsity = 0.6
    # As = gen_local_change(d,sparsity)
    # nr_params = int(d*(d-1)/2)
  


    # rnd = np.random.RandomState(10)
    # X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 80)
    # X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 40)
    # X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 30)
    # X = np.vstack([X1,X2,X3])


    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)

    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    #CE_glasso(X, obs_per_graph, 5000, 0.6, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)


    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.3, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.9, sparsity, False)




    # d = 10
    # sparsity = 0.3
    # As = gen_local_change(d,sparsity)
    # nr_params = int(d*(d-1)/2)
  


    # rnd = np.random.RandomState(10)
    # X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 80)
    # X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 40)
    # X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 30)
    # X = np.vstack([X1,X2,X3])

    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)

    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, False)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)


    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.3, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, False)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.9, sparsity, False)



    d = 10
    sparsity = 0.6
    As = gen_local_change(d,sparsity)
    nr_params = int(d*(d-1)/2)
  


    rnd = np.random.RandomState(10)
    X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 50)
    X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 50)
    X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 50)

    X11 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 450)
    X22 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[1]), size = 450)
    X33 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[2]), size = 450)
    X = np.vstack([X1,X11,X2, X22,X3, X33])

    for tp in [0.8, 0.6, 0.9, 0.5]:
        CE_glasso(X, 500, 5000, tp, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)

        CE_glasso(X, 500, 5000, tp, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
        CE_glasso(X, 500, 5000, tp, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)

    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.3, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.9, sparsity, True)




    # d = 10
    # sparsity = 0.3
    # As = gen_local_change(d,sparsity)
    # nr_params = int(d*(d-1)/2)
  


    # rnd = np.random.RandomState(10)
    # X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 50)
    # X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 50)
    # X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 50)
    # X = np.vstack([X1,X2,X3])

    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)

    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'cai', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'cai', used, 0.6, sparsity, True)

    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.01, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.9, nr_its, 0.025, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)


    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.3, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.05, alpha_in_CE, 'zero-one', used, 0.6, sparsity, True)
    # CE_glasso(X, obs_per_graph, 5000, 0.8, nr_its, 0.05, 0.01, alpha_in_CE, 'zero-one', used, 0.9, sparsity, True)
