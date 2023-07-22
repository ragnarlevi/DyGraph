import numpy as np
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph/src')
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import tqdm
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
    for it_nr in range(1,nr_its+1):

        value_vals = np.zeros((H,nr_its,N))  # store log-likelihoods

        # find optimal theta where B_i dictates which elements get regularized to infinity



        # Draw bernoulli
        B = np.zeros((H,N,nr_params))
        for h in range(H):
            B[h] = scipy.stats.bernoulli.rvs(p = p[h,it_nr-1], size = (N,nr_params))   

        for i in range(N):


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


            dg_opt1 = dg.dygl_inner_em(Xs, obs_per_graph = w, max_iter = 2000, lamda = alpha, kappa =kappa, tol = 1e-4)
            dg_opt1.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
            theta_reference = dg_opt1.theta
            theta_reference[np.abs(theta_reference)<1e-3] = 0

        
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


                
            dg_opt2 = dg.dygl_inner_em(X_used, obs_per_graph = w, max_iter = 2000, lamda = reg_matrix, kappa =kappa, tol = 1e-3)
            dg_opt2.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
            dg_opt2.theta[np.abs(dg_opt2.theta)<1e-5] = 0
            
            thetas_sim[i] = dg_opt2.theta.copy()

            val_func = np.zeros(H)
            for h in range(H):
                if metric == 'cai':
                    _, val1 = my_hdtest(Xs[h*w:(h+1)*w], np.linalg.inv(theta_reference[h]),X_used[h*w:(h+1)*w], np.linalg.inv(dg_opt2.theta[h]))
                elif metric == 'zero-one':
                    val1 = -np.sum(np.sign(theta_reference[h][np.triu_indices(d,1)])==np.sign(dg_opt2.theta[h][np.triu_indices(d,1)]))
                #val1 = np.linalg.norm(theta_reference[0]-theta1)
                value_vals[h,it_nr-1,i] = val1
                #_, value_vals[h,it_nr-1,i] = my_hdtest(np.vstack((X1,X2)), np.linalg.inv(0.5*(theta_reference[0]+theta_reference[1])),np.vstack((X1,X2)), np.linalg.inv(0.5*(theta1+theta2)))
                #value_vals[h,it_nr-1,i] = np.linalg.norm(theta_reference[0]+theta_reference[1] - theta1 - theta2)
            pbar.update()

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



def run_sim(X, alpha, kappa, obs_per_graph, cnt_a, cnt_k, nr_ce_it, N, alpha_in_CE, tail_prob, metric, used):


    d = X.shape[1]
    n = X.shape[0]
    nr_params = int(d*(d-1)/2)
    p_distribution = np.ones((int(n/obs_per_graph), nr_ce_it+1, nr_params))
    sigmas = np.ones((int(n/obs_per_graph), nr_ce_it+1, nr_params))

    dg_opt1 = dg.dygl_inner_em(X, obs_per_graph = obs_per_graph, max_iter = 10000, lamda = alpha, kappa = kappa, tol = 1e-6)
    dg_opt1.fit(nr_workers=1, temporal_penalty="element-wise", lik_type="gaussian",verbose=False)
    thetas_point = dg_opt1.theta.copy()
    
    p_distribution, sigmas, thetas_sim = CE_glasso(X.copy(), obs_per_graph, N, tail_prob, nr_ce_it, alpha, kappa, alpha_in_CE, metric, used)

    return p_distribution, thetas_point, sigmas, thetas_sim, cnt_k, cnt_a



if __name__ == '__main__':

        

    d = 10
    s = 0.3
    nr_ce_it = 60
    N = 4000
    As = gen_local_change(d,s)
    nr_params = int(d*(d-1)/2)
    tail_prob = 0.9
    obs_per_graph = 50
    ratio_nr_zeros = 0.9


    rnd = np.random.RandomState(10)
    X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 50)
    X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 50)
    X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 50)

    X = np.vstack([X1,X2,X3])

    n = X.shape[0]
    kappas = [0.01, 0.05, 0.1]# np.exp(-np.linspace(5, 0,20)) #[0.02, 0.05, 0.08, 0.15] #np.exp(-np.linspace(5, 0,6))
    alphas = [0.01,  0.025, 0.05, 0.07, 0.09, 0.11, 0.15, 0.2]#0.005, 0.01, ## np.exp(-np.linspace(5, 1.5,20)) #[0.01, 0.02, 0.03, 0.05, 0.09, 0.13]




    def run_case(X, alpha_in_CE, model, metric, used, prefix):
        print(f'{prefix}_{obs_per_graph}_{metric}_{used}_{alpha_in_CE}' )

       
        p_distribution = np.ones((len(kappas), len(alphas), int(n/obs_per_graph), nr_ce_it+1, nr_params))
        sigmas = np.ones((len(kappas), len(alphas), int(n/obs_per_graph)    , nr_ce_it+1, nr_params))
        thetas = np.zeros((len(kappas), len(alphas), int(n/obs_per_graph), d,d)) 
        thetas_sim = np.zeros((len(kappas), len(alphas),N, int(n/obs_per_graph),d,d)) 

        pool = Pool(12)
        results = pool.starmap(run_sim, tqdm.tqdm((( X, alpha, kappa, obs_per_graph, cnt_a, cnt_k, nr_ce_it, N, alpha_in_CE, tail_prob, metric, used) for cnt_a, alpha in enumerate(alphas) for cnt_k, kappa in enumerate(kappas)), total = len(kappas)*len(alphas)))

        for result in results:
            p_distribution[result[4], result[5]] = result[0]
            thetas[result[4], result[5]] = result[1]
            sigmas[result[4], result[5]]= result[2]
            thetas_sim[result[4], result[5]]= result[3]

        out_dict = {'alphas':alphas, 'kappas':kappas, 'p_distribution':p_distribution, 'thetas': thetas, 'thetas_sim':thetas_sim, 'X':X, 'As':As, 'N':N, 'nr_ce_it':nr_ce_it,
                    'ratio_nr_zeros':ratio_nr_zeros, 'obs_per_graph':obs_per_graph, 'alpha_in_CE':alpha_in_CE, 'model':model, 'sigmas':sigmas,
                    'used':used}

        with open(f'data/GraphHypTest/{prefix}_{obs_per_graph}_{metric}_{used}_{alpha_in_CE}.pkl', 'wb') as handle:
        #with open(f'data/GraphHypTest/test.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





    # run_case(X.copy(), alpha_in_CE = False,model = 'Gaussian',metric = 'zero-one',used = 'xs_xs',prefix = 'known_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'zero-one',used = 'xs_x',prefix = 'known_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'cai',used = 'xs_xs',prefix = 'known_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'cai',used = 'xs_x', prefix = 'known_cp')
    #run_case(alpha_in_CE = True,model = 'Gaussian',metric = 'cai',used = 'xs_xs')
    #run_case(alpha_in_CE = True,model = 'Gaussian',metric = 'cai',used = 'xs_x')



    rnd = np.random.RandomState(10)
    X1 = rnd.multivariate_normal(mean = np.zeros(d),  cov = np.linalg.inv(As[0]), size = 80)
    X2 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[1]), size = 40)
    X3 = rnd.multivariate_normal(mean = np.zeros(d),  cov =  np.linalg.inv(As[2]), size = 30)

    X = np.vstack([X1,X2,X3])



    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'zero-one',used = 'xs_xs',prefix = 'unknown_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'zero-one',used = 'xs_x',prefix = 'unknown_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'cai',used = 'xs_xs',prefix = 'unknown_cp')
    run_case(X.copy(),alpha_in_CE = False,model = 'Gaussian',metric = 'cai',used = 'xs_x', prefix = 'unknown_cp')





