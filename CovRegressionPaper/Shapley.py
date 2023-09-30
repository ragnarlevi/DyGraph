

import pandas as pd
import numpy as np
import sys



sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph')
sys.path.insert(0, 'C:/Users/ragna/Documents/Code/DyGraph/src')
import CovReg as cr
import tqdm

from scipy.optimize import minimize


from sklearn import linear_model


def cov_regression_lasso(X, Y, alpha, max_itr = 100, tol = 1e-5):
    
    d = Y.shape[1]
    n = X.shape[0]
    Y_tilde = np.vstack((Y, np.zeros((n,d))))

    r = X.shape[1]
    alpha = n*alpha
    iteration = 0
    max_itr = 100

    Psi_pre = np.identity(d)
    v = np.ones(n)
    m = np.ones(n)
    X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
    # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
    B_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))


    B_tmp = B_pre.flatten()
    B_vec = np.zeros(2*r*d)
    B_vec[:r*d] = np.abs(B_tmp) * (B_tmp>0)
    B_vec[r*d:] = np.abs(B_tmp) * (B_tmp<0)
    # B_pre = np.zeros((r,d))
    while iteration <max_itr:
        Psi_pre_inv = np.linalg.inv(Psi_pre)
        # print(np.linalg.cond(Psi_pre_inv))
        # E-step
        for i in range(n):
            v[i] = (1+np.dot(X[i].T, B_pre.T).dot(Psi_pre_inv).dot(B_pre).dot(X[i])) ** (-1)
            m[i] = v[i]*np.dot((Y[i]-0).T,Psi_pre_inv).dot(B_pre).dot(X[i])

        X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
        reg_lasso = linear_model.Lasso(alpha = alpha).fit(X_tilde,Y_tilde)
        B = reg_lasso.coef_.copy()

        Psi = np.cov((Y-np.dot(X,B.T)).T)*(n-1)/n + 0.001*np.identity(d)
        

        if np.linalg.norm(B-B_pre)<tol:
            break

        Psi_pre = Psi.copy()
        B_pre = B.copy()

        iteration+=1

    return B, Psi





def cov_regression(X, Y, alpha, max_itr = 100, tol = 1e-5):
    
    d = Y.shape[1]
    n = X.shape[0]
    Y_tilde = np.vstack((Y, np.zeros((n,d))))

    r = X.shape[1]
    alpha = n*alpha
    iteration = 0
    max_itr = 100

    Psi_pre = np.identity(d)
    v = np.ones(n)
    m = np.ones(n)
    X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
    # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
    B_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + 0.0001*np.identity(r)))




    def fun(param, X, Y,Psi_inv, d,r, alpha):
        
        B = np.reshape(param[:r*d]-param[r*d:],(d,r))
        obj = np.trace(np.dot((Y- np.dot(X,B.T)).T,(Y- np.dot(X,B.T))).dot(Psi_inv)) + alpha*param.sum() #+ 0.5*alpha*np.sum(param**2)#  + alpha*np.sum(np.abs(B))
        t_0 = Y.T-np.dot(B,X.T)
        grad = (-0.5*np.dot(Psi_inv,t_0).dot(X) ).flatten() #  0.5*alpha*param[:r*d] 
        return obj, np.concatenate((grad + alpha , - grad + alpha ), axis=None) # np.trace(np.dot((Y- np.dot(X,B.T)).T,(Y- np.dot(X,B.T))).dot(Psi_inv)) + alpha*np.sum(np.abs(B))

    B_tmp = B_pre.flatten()
    B_vec = np.zeros(2*r*d)
    B_vec[:r*d] = np.abs(B_tmp) * (B_tmp>0)
    B_vec[r*d:] = np.abs(B_tmp) * (B_tmp<0)
    # B_pre = np.zeros((r,d))
    while iteration <max_itr:
        Psi_pre_inv = np.linalg.inv(Psi_pre)
        # print(np.linalg.cond(Psi_pre_inv))
        # E-step
        for i in range(n):
            v[i] = (1+np.dot(X[i].T, B_pre.T).dot(Psi_pre_inv).dot(B_pre).dot(X[i])) ** (-1)
            m[i] = v[i]*np.dot((Y[i]-0).T,Psi_pre_inv).dot(B_pre).dot(X[i])

        X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
        # M-step
        out = minimize(fun, B_vec, args = (X_tilde, Y_tilde, np.identity(d),d,r, alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*r*d))
        B_vec = out.x
        B = np.reshape(out.x[:r*d] - out.x[r*d:], (d,r))
        #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
        #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)


        Psi = np.cov((Y-np.dot(X,B.T)).T)*(n-1)/n
        

        if np.linalg.norm(B-B_pre)<tol:
            break

        Psi_pre = Psi.copy()
        B_pre = B.copy()
        

        #print(scipy.linalg.norm(B_pre-B_true))
        iteration+=1

    return B, Psi


def cov_regression_t(X, Y, alpha, nu, max_itr = 100):
    
    d = Y.shape[1]
    n = X.shape[0]
    Y_tilde = np.vstack((Y, np.zeros((n,d))))

    r = X.shape[1]
    alpha = n*alpha
    iteration = 0
    max_itr = 100

    Psi_pre = np.identity(d)
    v = np.ones(n)
    m = np.ones(n)
    X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
    B_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + 0.0001*np.identity(r)))




    def fun(param, X, Y,Psi_inv, d,r, alpha,):
        
        B = np.reshape(param[:r*d]-param[r*d:],(d,r))
        obj = np.trace(np.dot((Y- np.dot(X,B.T)).T,(Y- np.dot(X,B.T))).dot(Psi_inv)) + alpha*param.sum() #+ 0.5*alpha*np.sum(param**2)#  + alpha*np.sum(np.abs(B))
        t_0 = Y.T-np.dot(B,X.T)
        grad = (-np.dot(Psi_inv,t_0).dot(X) ).flatten() #  0.5*alpha*param[:r*d] 
        return obj, np.concatenate((grad + alpha , - grad + alpha ), axis=None) # np.trace(np.dot((Y- np.dot(X,B.T)).T,(Y- np.dot(X,B.T))).dot(Psi_inv)) + alpha*np.sum(np.abs(B))

    B_tmp = B_pre.flatten()
    B_vec = np.zeros(2*r*d)
    B_vec[:r*d] = np.abs(B_tmp) * (B_tmp>0)
    B_vec[r*d:] = np.abs(B_tmp) * (B_tmp<0)
    # B_pre = np.zeros((r,d))
    while iteration <max_itr:
        Psi_pre_inv = np.linalg.inv(Psi_pre)
        # print(np.linalg.cond(Psi_pre_inv))
        # E-step
        for i in range(n):
            v[i] = (1+np.dot(X[i].T, B_pre.T).dot(Psi_pre_inv).dot(B_pre).dot(X[i])) ** (-1)
            m[i] = v[i]*np.dot((Y[i]-0).T,Psi_pre_inv).dot(B_pre).dot(X[i])
        M = np.einsum('nj,jk,nk->n', (Y -np.dot(X, B_pre.T)), Psi_pre_inv, (Y -np.dot(X, B_pre.T)))  # Mahalanobis distance
        tau = (nu + d)/(nu  + M)

        X_tilde =  np.vstack((m[:,np.newaxis]*np.sqrt(tau[:,np.newaxis])*X,v[:,np.newaxis]*np.sqrt(tau[:,np.newaxis])*X))
        Y_tilde = np.vstack((Y*np.sqrt(tau[:,np.newaxis]), np.zeros((n,d))))
        # M-step
        out = minimize(fun, B_vec, args = (X_tilde, Y_tilde, Psi_pre_inv,d,r, alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*r*d))
        B_vec = out.x
        B = np.reshape(out.x[:r*d] - out.x[r*d:], (d,r))
        #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
        #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)


        Psi = np.cov((Y-np.dot(X,B.T)).T)*(n-1)/n
        

        Psi_pre = Psi.copy()
        B_pre = B.copy()
        

        #print(scipy.linalg.norm(B_pre-B_true))
        iteration+=1

    return B, Psi


def cov_regression_subgrad(X, Y, alpha, max_itr = 100, step_size = 0.05):
    
    d = Y.shape[1]
    n = X.shape[0]
    Y_tilde = np.vstack((Y, np.zeros((n,d))))

    r = X.shape[1]
    alpha = n*alpha
    iteration = 0
    max_itr = 100


    Psi_pre = np.identity(d)
    v = np.zeros(n)
    m = np.zeros(n)
    v = np.ones(n)
    m = np.ones(n)
    X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
    B_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + 0.1*np.identity(r)))


    # B_pre = np.zeros((r,d))
    while iteration <max_itr:
        Psi_pre_inv = np.linalg.inv(Psi_pre)
        # print(np.linalg.cond(Psi_pre_inv))
        # E-step
        for i in range(n):
            v[i] = (1+np.dot(X[i].T, B_pre.T).dot(Psi_pre_inv).dot(B_pre).dot(X[i])) ** (-1)
            m[i] = v[i]*np.dot((Y[i]-0).T,Psi_pre_inv).dot(B_true).dot(X[i])

        X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
        # M-step

        B=B_pre.copy()
        for sub_step in range(1,1000):
            B = B - (step_size/sub_step)* (-0.5*np.dot(Psi_pre_inv, Y_tilde.T -np.dot(B,X_tilde.T)).dot(X_tilde) + alpha*np.sign(B))
            # B = soft_threshold_odd(B,alpha)


        Psi = np.cov((Y-np.dot(X,B.T)).T)*(n-1)/n
            

        Psi_pre = Psi.copy()
        B_pre = B.copy()

        iteration+=1

    return B, Psi



def soft_threshold_odd( A, lamda):

    """
    diagonal lasso penalty

    Parameters
    ------------------
    A: np.array,
    
    lamda: float,
        regularization
    """
    opt_m = (A-lamda)*(A>=lamda) + (A+lamda)*(A<=-lamda)
    

    return opt_m



def cov_regression_gen_grad(X, Y, alpha, max_itr = 100, step_size = 0.05, grad_itr = 100):
    
    d = Y.shape[1]
    n = X.shape[0]
    Y_tilde = np.vstack((Y, np.zeros((n,d))))

    r = X.shape[1]
    alpha = n*alpha
    iteration = 0
    max_itr = 100


    Psi_pre = np.identity(d)
    v = np.zeros(n)
    m = np.zeros(n)
    v = np.ones(n)
    m = np.ones(n)
    X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
    B_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + 0.1*np.identity(r)))


    # B_pre = np.zeros((r,d))
    while iteration <max_itr:
        Psi_pre_inv = np.linalg.inv(Psi_pre)
        # print(np.linalg.cond(Psi_pre_inv))
        # E-step
        for i in range(n):
            v[i] = (1+np.dot(X[i].T, B_true.T).dot(Psi_pre_inv).dot(B_true).dot(X[i])) ** (-1)
            m[i] = v[i]*np.dot((Y[i]-0).T,Psi_pre_inv).dot(B_true).dot(X[i])

        X_tilde =  np.vstack((m[:,np.newaxis]*X,v[:,np.newaxis]*X))
        # M-step
        alpha = 1
        B=B_pre.copy()
        # Gradient
        for sub_step in range(1,grad_itr+1):
            B = B - (step_size/sub_step)* (-0.5*np.dot(Psi_pre_inv, Y_tilde.T -np.dot(B,X_tilde.T)).dot(X_tilde))
            B = soft_threshold_odd(B,alpha)

        Psi = np.cov((Y-np.dot(X,B.T)).T)*(n-1)/n
            

        Psi_pre = Psi.copy()
        B_pre = B.copy()
        

        #print(scipy.linalg.norm(B_pre-B_true))
        iteration+=1

    return B, Psi


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
    max_iter = 1000

    # test parameters
    ns = [100, 500,1000]
    rs = [5, 10, 20, 50, 100]
    ds = [10]
    alphas = np.concatenate(([0], np.logspace(-4,0.5, 60)))

    rnd = np.random.RandomState(42)


            

    # Storetrue data and coef
    B_dict_true = dict()
    A_dict_true = dict()
    Xs_dict = dict()
    Ys_dict_cov = dict()
    Ys_dict_meancov = dict()

    B_est = dict()



    rnd = np.random.RandomState(42)

    Xs_dict = {str(r): [] for r in rs}
    for r in rs:
        Xs_dict[str(r)] = rnd.normal(loc = 0, scale = 1, size = (np.max(ns),r))


    # Generate coef matrix
    for r in rs:
        for d in ds:
            B_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            B_true[np.abs(B_true)<0.8] = 0
            B_dict_true[str(r) + '_'+str(d)] = B_true
            A_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
            A_true = B_true*(rnd.uniform(size = (d,r) ) <0.6)
            A_dict_true[str(r) + '_'+str(d)] = A_true

    # Generate observations

    for r in rs:
        for d in ds:
            B_tmp = B_dict_true[str(r) + '_'+str(d)].copy()
            A_tmp = A_dict_true[str(r) + '_'+str(d)].copy()

            X = Xs_dict[str(r)].copy()
            gamma = rnd.normal(loc = 0, scale = 1, size = (np.max(ns)))
            epsilon = rnd.normal(loc = 0, scale = 0.1, size = (np.max(ns),d))

            Y_cov = gamma[:, np.newaxis]*np.dot(X, B_tmp.T) + epsilon
            Ys_dict_cov[str(r) + '_'+str(d)] = Y_cov.copy()


            Y_meancov = np.dot(X, A_tmp.T) + gamma[:, np.newaxis]*np.dot(X, B_tmp.T) + epsilon
            Ys_dict_meancov[str(r) + '_'+str(d)] = Y_meancov.copy()


    liks_dict = dict()
    nr_param_dict= dict()

    for r in rs:
        print(r)
        liks_dict_tmp = dict()
        nr_param_dict_tmp= dict()
        B_est_tmp = dict()
        for n in ns:
            liks = []
            nr_param = []
            Bs = []
            for a in range(len(alphas)):
                print((r, n, a))
                #psi = Psi_dict_cov[str(r) + '_'+str(d)][k][a]
                #B_tmp = B_dict_cov[str(r) + '_'+str(d)][k][a]
                #nr_param.append(np.sum(np.abs(B_tmp)>1e-3))

                x = Xs_dict[str(r)][:n]
                y = Ys_dict_cov[str(r) + '_'+str(d)][:n]
                try:
                    cov = cr.CovReg( Y = y, alpha = alphas[a], max_iter = 1000, tol = 1e-6)
                    cov.fit_hoff_b_only(X2 = x, verbose = False, psi = 0.1*np.identity(d))
                    l, npara = cov.marg_lik(X2 = x)
                    liks.append(l)
                    nr_param.append(npara)
                    Bs.append(cov.B.copy())
                except:
                    liks.append(np.nan)
                    nr_param.append(np.nan)
                    Bs.append(np.nan)
                


                liks_dict_tmp[n] = np.array(liks).copy()
                nr_param_dict_tmp[n] = np.array(nr_param).copy()
                B_est_tmp[n] = Bs.copy()



        
        liks_dict[r] = liks_dict_tmp.copy()
        nr_param_dict[r] = nr_param_dict_tmp.copy()
        B_est[r] = B_est_tmp.copy()
        

        out_dict = {'liks_dict':liks_dict, 'nr_param_dict':nr_param_dict, 'B_est':B_est, 'B_true':B_dict_true, 'Ys_dict_cov':Ys_dict_cov, 'Xs_dict':Xs_dict}
        
        import pickle
        with open(f'CovRegressionPaper/data_sim/shapley_simple_psi0.1.pkl', 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

