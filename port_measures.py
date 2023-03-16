import numpy as np
import pandas as pd
import scipy.integrate as integrate
from statsmodels.distributions.empirical_distribution import ECDF

def div_ratio(w, cov):
  # numerator is perfect correlation
  # denom is portfolio risk
  return np.inner(w, np.sqrt(np.diag(cov)))/np.sqrt(np.dot(w, cov).dot(w))

def var_div_ratio(w,data, q = 0.95):
  # w weights
  # d = data

  ind_var = np.zeros(data.shape[1])
  for col in range(data.shape[1]):
      ind_var[col] = np.quantile(-data[:,col], q)

  port_var = np.quantile(np.dot(-data, w), q)

  return port_var/np.inner(ind_var,w)

def fix_weight(w:np.array):

    if np.sum(w[w <0]) <-0.3:

        # fix negative
        w[w <0] = 0.3*w[w <0]/np.abs(np.sum(w[w <0]))
        w[w >=0] = 1.3*w[w >=0]/np.abs(np.sum(w[w >=0]))

    return w



def omega(x, level = 0):
  ecdf = ECDF(x)  
  numerator = integrate.quad(lambda x: 1-ecdf(x), level, np.inf, limit = 10000)
  denominator = integrate.quad(ecdf, -np.inf, level, limit = 10000)
  if denominator[0] == 0.0:
    return 10
  else:
    return numerator[0]/denominator[0]

def sharpe(mu, sigma, r_f = 0):

  return (mu-r_f)/sigma

def sortino(mu,x, r_f = 0):

  x_above = x.copy()
  x_above[x_above > r_f] = r_f

  return (mu -r_f)/(np.sqrt(np.mean(x_above ** 2)))

def beta(X_port, X_index):
  return np.cov(X_port,X_index)[0,1]/np.var(X_index)

def treynor(mu, beta, r_f = 0):
  return (mu-r_f)/beta

def max_drawdown(price):
  return np.min((price/np.array(pd.DataFrame(price).cummax().iloc[:,0])-1))


def portfolio_opt(S,precision_matrix, mu, X, type):


  if type == 'uniform':
      w = np.ones(S.shape[1])/S.shape[1]
      mu_p = np.mean(np.dot(X, w))
      var_p = np.dot(w,S).dot(w)
  elif type == 'sharpe':
      w = np.dot(precision_matrix, mu)/np.dot(np.ones(S.shape[0]), precision_matrix).dot(mu) 
      w = fix_weight(w)
      mu_p = np.mean(np.dot(X, w))
      var_p = np.dot(w,S).dot(w)
  elif type == 'gmv':
      w = np.dot(precision_matrix, np.ones(S.shape[0]))/np.dot(np.ones(S.shape[0]), precision_matrix).dot(np.ones(S.shape[0])) 
      w = fix_weight(w)
      mu_p = np.mean(np.dot(X, w))
      var_p = np.dot(w,S).dot(w)


  return w, mu_p, var_p 
