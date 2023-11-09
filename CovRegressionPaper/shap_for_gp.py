



import numpy as np


import tqdm
import scipy
from scipy.optimize import minimize
import gpflow
import tensorflow as tf

from jax import value_and_grad
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

def gp_ind_obj(param,Y,r, K, scale):

    d = Y.shape[1]
    n = Y.shape[0]



    B = param.reshape((d,r))


    #C = jnp.kron(jnp.dot(B,B.T), K) + scale*jnp.identity(n*d)
    #C_inv = jnp.linalg.inv(C)

    #v,_ = jnp.linalg.eigh(C)
    #v[v<= 0] = 1e-6
    #v.at[v<= 0].set(1e-6)

    #obj = -0.5*np.sum(np.log(v)) - 0.5*np.einsum('ij,ji->', C_inv, np.outer(Y.flatten(order='F'),Y.flatten(order='F')))
    #obj = -0.5*jnp.sum(jnp.log(v)) - 0.5*jnp.trace(jnp.dot(C_inv, jnp.outer(Y.flatten(order='F'),Y.flatten(order='F'))))


    cov_xx = K + (scale**2) * jnp.identity(n)
    cov = jnp.kron(jnp.dot(B,B.T), cov_xx) + 0.01 * jnp.identity(n*d)
    LL = mvn.logpdf(Y.flatten(order = 'F'), np.zeros(n*d), cov)

    return -LL#, -grad1.flatten()-grad2.flatten()







def calc_shapley_value(B,X=None, Sigma = None):
    if Sigma is None:
        Sigma = np.cov(X.T)
    
    r = Sigma.shape[0]
    val_y = np.dot(B, Sigma).dot(B.T)

    d = val_y.shape[0]
    shapleys = np.zeros((int(d*(d+1)/2), r))
    index = np.arange(r)


    cnt=0
    shap_matrix = np.zeros((r,d,d))
    for i in range(d):
        for j in range(i,d):
            for k in range(r):
                t1 = B[i,k]*B[j,k]*Sigma[k,k]
                t2 = 0.5*np.sum(B[i,index != k]*B[j,k]*Sigma[k,index != k])
                t3 = 0.5*np.sum(B[j,index != k]*B[i,k]*Sigma[k,index != k])
                shapleys[cnt,k] = t1+t2+t3
                shap_matrix[k,i,j] = t1+t2+t3
            cnt+=1

    for k in range(r):
        shap_matrix[k] = shap_matrix[k]+shap_matrix[k].T -np.diag(shap_matrix[k])

    return shapleys, shap_matrix



rnd = np.random.RandomState(42)
n = 200
d = 10
r = 4
T = np.linspace(1,10,n).reshape(-1,1)

kernel_true = gpflow.kernels.RBF(variance=1, lengthscales=1) + gpflow.kernels.White(variance=0.1)

K = kernel_true.K(T)#rbf_kernel(T,T, gamma = 0.1) + 0.01*np.identity(n)# linear_kernel(T,T)+ 0.001
K_inv = np.linalg.inv(K)
F_true = np.zeros((n, r))
for i in range(r):
    F_true[:,i] = rnd.multivariate_normal(mean = np.zeros(n), cov = K, size = 1)


H_test = np.identity(r)

scale = 0.1
psi = scale*np.identity(d)
psi_inv = np.linalg.inv(psi)
epsilon= rnd.normal(loc = 0, scale = scale, size = (n,d))
B_true = rnd.normal(loc = 0, scale = 1, size = (d,r))
Y_gp_true = np.dot(F_true, np.dot(H_test, B_true.T)) + epsilon




B_est = dict()


for try_r in tqdm.tqdm(range(1, d)):
    print(try_r)
    out = minimize(value_and_grad(gp_ind_obj), np.random.uniform(size = d*try_r), args = (Y_gp_true,try_r, np.array(K), scale), jac = True, method= 'CG')
    B_est[try_r] = out.x.reshape((d,try_r))


import pickle
with open('shap_gp_experiment.pkl', 'wb') as handle:
    pickle.dump(B_est, handle, protocol=pickle.HIGHEST_PROTOCOL)