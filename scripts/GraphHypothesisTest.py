import numpy as np
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph/src')
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import tqdm
import scipy
import scipy
from multiprocessing.pool import Pool


import pickle



def gen_all_all_zero(d,s, v1 = 1.1, v2 = 1.1):

    while True:


        k = d
        sparsity = s
        rnd = np.random.RandomState(42)
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
        A = A - (np.min(l)-.1) * np.identity(k)

        A2 = A.copy()
        A2_SIGNS = np.sign(A2)
        A2 = np.power(np.abs(A2), v1)
        A2 = A2*A2_SIGNS
        np.fill_diagonal(A2,np.diag(A))

        A3 = A2.copy()
        A3_SIGNS = np.sign(A3)
        A3 = np.power(np.abs(A3), v2)
        A3 = A3*A3_SIGNS
        np.fill_diagonal(A3,np.diag(A))


        try:
            np.linalg.inv(A)
            np.linalg.inv(A2)
            np.linalg.inv(A3)
        except:
            continue


        break

    return np.array([A, A2, A3])


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


def hdtest_boot(Z, cov_Z, Y, cov_Y, B):

    test_stat = np.zeros(B)

    X = np.vstack((Z,Y))

    _,t_stat_sample= my_hdtest(Z, cov_Z, Y, cov_Y)
    for i in range(B):
        
        #from sklearn.model_selection import train_test_split
        #X1_tmp, X2_tmp = train_test_split(X, test_size=0.5)

        X1_tmp = X[np.random.choice(X.shape[0], X.shape[0])]
        X2_tmp = X[np.random.choice(X.shape[0], X.shape[0])]
        p,test_stat[i] = my_hdtest(X1_tmp, cov_Z, X2_tmp, cov_Y) # out['statistics'][0] #
    
    return np.sum(test_stat >t_stat_sample)/len(test_stat),t_stat_sample,test_stat

    
def CE_remove_lik(X,N, tail_prob_init, tail_prob_final, theta_1, nr_its = 2):

    tail_direction = np.sign(tail_prob_final-tail_prob_init)

    theta_1[np.abs(theta_1)<1e-3] = 0.0


    d = X.shape[1]

    nr_params = int(d*(d-1)/2)

    p = np.array([0.5]*nr_params)

    current_tail_prob = tail_prob_init
    for _ in range(nr_its):

        liks = np.zeros(N)  # store log-likelihoods
        B = scipy.stats.bernoulli.rvs(p = p, size = (N,nr_params))  

        # find optimal theta where B_i dictates which elements get regularized to infinity
        for i in range(N):
            theta_2 = np.zeros((d,d))
            theta_2[np.triu_indices(d,1)] = (1-B[i])*theta_1[np.triu_indices(d,1)]
            theta_2 = theta_2 + theta_2.T
            np.fill_diagonal(theta_2, np.diag(theta_1))

            try:
            
                liks[i] = np.sum(scipy.stats.multivariate_normal.logpdf(X, mean =np.zeros(d), cov = np.linalg.inv(theta_2))) - np.log(X.shape[0])*(np.sum(theta_2[np.triu_indices(d,1)] != 0))
            except:
                liks[i] = -np.inf


        val_func = np.quantile(liks, current_tail_prob)

        if tail_direction >0:
            current_tail_prob = np.min((tail_prob_final, current_tail_prob + 0.05 ))
        elif tail_direction <0: 
            current_tail_prob = np.max((tail_prob_final, current_tail_prob - 0.05 ))
        else:
            current_tail_prob = tail_prob_final

        

        # update p
        for i in range(nr_params):
            p[i] = np.sum((liks >= val_func)*B[:,i])/np.sum((liks >= val_func))


    # Finally use best estimate of I to get theta
    I = 1 * (p >0.5)
    theta_2 = np.zeros((d,d))
    theta_2[np.triu_indices(d,1)] = (1-I)*theta_1[np.triu_indices(d,1)]
    theta_2 = theta_2 + theta_2.T
    np.fill_diagonal(theta_2, np.diag(theta_1))

    return theta_2



def CE_glasso(X,N, theta_reference, tail_prob, nr_its, alpha, nr_zeros_limit, alpha_in_CE, model, bootstrap_X, boot_refrence):

    S_refrence = np.linalg.inv(theta_reference)

    d = X.shape[1]

    nr_params = int(d*(d-1)/2)

    p = np.zeros((nr_its+1,nr_params))
    sigmas = np.zeros((nr_its+1, nr_params))
    if model == 'expon':
        p[0] = 0.4*alpha
    elif model == 'Bernoulli':
        p[0] = 0.5
    elif model == 'Gaussian':
        p[0]= 0.4*alpha
        sigmas[0] = p[0]*0.5

    eigvals = np.zeros((nr_its,N, d))
    nr_zeros_ce =  np.zeros((nr_its,N))

    pbar = tqdm.tqdm(total = nr_its, position=1)

    d = X.shape[1]

    nr_params = int(d*(d-1)/2)


    pbar = tqdm.tqdm(total = nr_its*N)
    for it_nr in range(1,nr_its+1):




        value_vals = np.zeros(N)  # store log-likelihoods
        if model == 'Bernoulli':
            B = scipy.stats.bernoulli.rvs(p = p[it_nr-1], size = (N,nr_params))   
        elif model == 'expon':
            B = scipy.stats.expon.rvs(scale = p[it_nr-1], size = (N,nr_params))  
        elif model == 'Gaussian':
            B = scipy.stats.norm.rvs(loc = p[it_nr-1], scale = sigmas[it_nr-1], size = (N,nr_params))  
            B[B<0] = 0
        else:
            raise ValueError("model not known")

        # find optimal theta where B_i dictates which elements get regularized to infinity
        for i in range(N):
            if model == 'expon':
                reg_matrix = np.zeros((d,d))
                reg_matrix[np.triu_indices(d,1)] = B[i]
                reg_matrix = reg_matrix + reg_matrix.T
            elif model == 'Gaussian':
                reg_matrix = np.zeros((d,d))
                reg_matrix[np.triu_indices(d,1)] = B[i]
                reg_matrix = reg_matrix + reg_matrix.T
            elif model == 'Bernoulli':
                if alpha_in_CE:
                    reg_matrix = np.ones((d,d))*alpha
                else:
                    reg_matrix = np.zeros((d,d))
                np.fill_diagonal(reg_matrix,0)
                reg_matrix[np.triu_indices(d,1)] = B[i]*999999
                reg_matrix = reg_matrix + reg_matrix.T
            if bootstrap_X:
                X_i = X[np.random.choice(X.shape[0], size = X.shape[0], replace=True)].copy()
            else:
                X_i = X

            gls = dg.sgl_inner_em(X_i, max_iter = 1000, lamda = reg_matrix, tol = 1e-4)
            gls.fit(nr_workers=1, lik_type="gaussian",verbose=False)
            theta_2 = gls.theta[-1].copy()
            eigvals[it_nr-1, i] = np.linalg.eigvals(theta_2)



            p_val, value_vals[i] = my_hdtest(X_i, np.linalg.inv(theta_2), X, S_refrence)

            edges = theta_2[np.triu_indices(d,1)]
            edges[np.abs(edges)<1e-3] = 0
            nr_zeros_ce[it_nr-1, i] = np.sum(edges==0)
            if nr_zeros_limit is not None:
                if nr_zeros_ce[it_nr-1, i] >nr_zeros_limit:
                    value_vals[i] = np.nan
            pbar.update()

        # Get quantile (best value)
        value_vals = -value_vals
        val_func = np.quantile(value_vals[~np.isnan(value_vals)], tail_prob)
        
        # update p
        for i in range(nr_params):
            p[it_nr, i] = np.sum((value_vals[~np.isnan(value_vals)] >= val_func)*B[~np.isnan(value_vals),i])/np.sum((value_vals[~np.isnan(value_vals)] >= val_func))
            sigmas[it_nr, i] = np.sqrt(np.sum((value_vals[~np.isnan(value_vals)] >= val_func)*(B[~np.isnan(value_vals),i]-p[it_nr, i])**2)/np.sum((value_vals[~np.isnan(value_vals)] >= val_func)))


        w = np.exp(-2/it_nr)*0.9
        p[it_nr] = w*p[it_nr] + (1-w)*p[it_nr-1]


    pbar.close()


    pbar.close()
    return p, eigvals, nr_zeros_ce, sigmas



def run_sim(X, alpha, kappa, obs_per_graph, cnt_a, cnt_k, nr_ce_it, N, ratio_nr_zeros, alpha_in_CE, model,bootstrap_X, tail_prob):


    d = X.shape[1]
    n = X.shape[0]
    nr_params = int(d*(d-1)/2)
    p_distribution = np.ones((int(n/obs_per_graph), nr_ce_it+1, nr_params))
    sigmas = np.ones((int(n/obs_per_graph), nr_ce_it+1, nr_params))
    eig_distribution =  np.zeros((int(n/obs_per_graph), nr_ce_it, N, d))
    nr_zeros_ce =  np.zeros((int(n/obs_per_graph), nr_ce_it, N))
    thetas = np.ones((int(n/obs_per_graph), d, d))

    dg_opt1 = dg.dygl_inner_em(X, obs_per_graph = obs_per_graph, max_iter = 10000, lamda = alpha, kappa = kappa, tol = 1e-6)
    dg_opt1.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    # Nr possible removals
    covs = []
    for i in range(len(dg_opt1.theta)):
        covs.append(np.linalg.inv(dg_opt1.theta[i]))
    for i in range(len(dg_opt1.theta)):
        thetas[i] = dg_opt1.theta[i].copy()  
        theta_reference = dg_opt1.theta[i].copy()  

        if ratio_nr_zeros is None:
            nr_zeros_limit = None
        else:
            edges = theta_reference[np.triu_indices(d,1)]
            edges[np.abs(edges)<1e-3] = 0
            nr_zeros_limit = ratio_nr_zeros*np.sum(edges==0)
        p_distribution[i], eig_distribution[i],nr_zeros_ce[i], sigmas[i] = CE_glasso(X[i*obs_per_graph:(i+1)*obs_per_graph], N,  theta_reference, tail_prob, nr_ce_it, alpha, 
                                                                          nr_zeros_limit, alpha_in_CE, model, bootstrap_X)

    return p_distribution,thetas, cnt_k, cnt_a, eig_distribution, nr_zeros_ce, sigmas



if __name__ == '__main__':

        
    d = 20
    s = 0.3
    nr_ce_it = 40
    N = 3000
    As = gen_local_change(d,s)
    nr_params = int(d*(d-1)/2)
    ratio_nr_zeros = 0.9
    alpha_in_CE = True
    model = 'Gaussian'
    bootstrap_X = False
    tail_prob = 0.95
    obs_per_graph = 200

    X1 = np.random.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = obs_per_graph)
    X2 = np.random.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = obs_per_graph)
    X3 = np.random.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = obs_per_graph)

    X = np.vstack([X1,X2,X3])
 
    n = X.shape[0]
    kappas = [0, 0.01, 0.02, 0.05, 0.1]# np.exp(-np.linspace(5, 0,20)) #[0.02, 0.05, 0.08, 0.15] #np.exp(-np.linspace(5, 0,6))
    alphas = [0.005, 0.01,0.02, 0.03, 0.05, 0.07, 0.1]#0.005, 0.01, ## np.exp(-np.linspace(5, 1.5,20)) #[0.01, 0.02, 0.03, 0.05, 0.09, 0.13]

    p_distribution = np.ones((len(kappas), len(alphas), int(n/obs_per_graph), nr_ce_it+1, nr_params))
    sigmas = np.ones((len(kappas), len(alphas), int(n/obs_per_graph), nr_ce_it+1, nr_params))
    eig_distribution = np.ones((len(kappas), len(alphas), int(n/obs_per_graph), nr_ce_it, N, d))
    nr_zeros_ce = np.ones((len(kappas), len(alphas), int(n/obs_per_graph), nr_ce_it, N))
    thetas = np.zeros((len(kappas), len(alphas), int(n/obs_per_graph), d,d)) 

    pool = Pool(10)
    results = pool.starmap(run_sim, tqdm.tqdm((( X.copy(), alpha, kappa, obs_per_graph, cnt_a, cnt_k, nr_ce_it, N, ratio_nr_zeros, alpha_in_CE, model,bootstrap_X,tail_prob) for cnt_a, alpha in enumerate(alphas) for cnt_k, kappa in enumerate(kappas)), total = len(kappas)*len(alphas), position=0))

    for result in results:

        p_distribution[result[2], result[3]] = result[0]
        thetas[result[2], result[3]] = result[1]
        eig_distribution[result[2], result[3]] = result[4]
        nr_zeros_ce[result[2], result[3]]= result[5]
        sigmas[result[2], result[3]]= result[6]

    out_dict = {'alphas':alphas, 'kappas':kappas, 'p_distribution':p_distribution, 'thetas': thetas, 'eig_distribution':eig_distribution, 'X':X, 'As':As, 'N':N, 'nr_ce_it':nr_ce_it,
                'ratio_nr_zeros':ratio_nr_zeros,'nr_zeros_ce':nr_zeros_ce, 'obs_per_graph':obs_per_graph, 'alpha_in_CE':alpha_in_CE, 'model':model, 'bootstrap_X':bootstrap_X, 'sigmas':sigmas}

    with open(f'data/GraphHypTest/local_change_rnr0_{ratio_nr_zeros}_model_{model}_alphaCE_{alpha_in_CE}_boot_{bootstrap_X}_{obs_per_graph}.pkl', 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



