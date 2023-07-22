
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import networkx as nx
import yfinance as yf
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')

import DyGraph as dg
import port_measures as pm
import matplotlib.pyplot as plt
import tqdm
import scipy
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


    
    ns = [50, 100, 500, 1000, 10000]
    rs = [5, 10, 50]
    ds = [5, 10, 50]
    alphas = np.exp(-np.linspace(15,2,60))

    B_dict = dict()
    B_dict_true = dict()
    Psi_dict = dict()
    Xs_dict = {str(r): [] for r in rs}
    Ys_dict = {str(r)+ '_' + str(d): [] for r in rs for d in ds}
    val_y_true = dict()
    value_function_dict = dict()
    pbar = tqdm.tqdm(total = len(alphas)*len(ns)*len(ds)*len(rs))

    for r in rs:
        for n_cnt, n in enumerate(ns):
            Xs_dict[str(r)].append(np.random.normal(loc = 0, scale = 1, size = (n,r)))


    for r in rs:
        for d in ds:

                
            B_true = np.random.normal(loc = 0, scale = 1, size = (d,r))
            B_true = B_true*(np.random.uniform(size = (d,r) ) <0.5)
            B_dict_true[str(r) + '_'+str(d)] = B_true
            for i in range(d):
                for j in range(i,d):
                    value_function_dict[str(r) + '_'+str(d)] = np.zeros(shape = (d,d,len(ns), len(alphas), r))
            
            Bs = np.zeros(shape = (len(ns), len(alphas), d, r))
            Psis = np.zeros(shape = (len(ns), len(alphas), d, d))

            val_y_true[str(r) + '_'+str(d)] = np.dot(B_true, np.identity(r)).dot(B_true.T)


            for n_cnt, n in enumerate(ns):
                X = Xs_dict[str(r)][n_cnt].copy()
                gamma = np.random.normal(loc = 0, scale = 1, size = (n))
                epsilon = np.random.normal(loc = 0, scale = 1, size = (n,d))
                Y = gamma[:, np.newaxis]*np.dot(X, B_true.T) + epsilon
                Ys_dict[str(r) + '_'+str(d)].append(Y.copy())
                for alpha_cnt, alpha in enumerate(alphas):
                    B, Psi = cov_regression(X,Y, alpha, 100 )
                    #B, Psi = cov_regression_lasso(X,Y,alpha,100)
                    Bs[n_cnt, alpha_cnt] = B.copy()
                    Psis[n_cnt, alpha_cnt] = Psi.copy()
                    
                    shapleys = calc_shapley_value(B,X)
                    for i in range(d):
                        for j in range(i,d):
                            value_function_dict[str(r) + '_'+str(d)][i,j,n_cnt, alpha_cnt] = shapleys[str(i)+','+ str(j)]

                    B_dict[str(r) + '_'+str(d)] = Bs
                    Psi_dict[str(r) + '_'+str(d)] = Psis
                    pbar.update()

            out_dict = {'alphas':alphas, 'ns':ns, 'rs':rs, 'ds':ds, 'value_function_dict':value_function_dict, 'Xs_dict':Xs_dict, 'Ys_dict':Ys_dict,
                        'B_dict':B_dict, 'B_dict_true':B_dict_true, 'Psi_dict':Psi_dict,'val_y_true':val_y_true}
            import pickle
            with open(f'data/Shapley/shapley7_nopsi.pkl', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


                    
    pbar.close()