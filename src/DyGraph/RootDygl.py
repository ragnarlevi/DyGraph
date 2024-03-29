

# Root class for data preperation

import numpy as np
import warnings
from scipy.stats import kurtosis



class RootDygl():

    def __init__(self, X, obs_per_graph, max_iter, lamda, kappa, S = None, kappa_gamma = 0, lik_type = 'gaussian', tol = 1e-6, groups = None) -> None:

        """
        Parameters
        ------------------
        X: array of size n, d
            data with n observations and d features.
        obs_per_graph: int,
            Observations used to construct each each matrix, can be 1 or larger


        max_iter: int,
            Maximum number of iterations
        
        lambda: float,
            regularization strength used for z0 l1 off diagonal 

        kappa: float or vector,
            regularization strength used for z1 and z2 temporal penalties

        kappa: float or vector,,
            regularization strength used for z3 and z4 gamma temporal penalties

        tol: float,
            Convergence tolerance.
        groups: numpy array of size d
            Grouping for EM
        
        
        """

        assert obs_per_graph >= 0, "block size has to be bigger than 0"

        self.S = S
        if X is not None:
            self.X = np.array(X)
            self.d = X.shape[1]
            self.n = self.X.shape[0]
        else:
            self.X = None
            self.d = self.S[0].shape[1]

        self.max_iter = max_iter
        self.lamda = lamda*obs_per_graph
        self.kappa = kappa*obs_per_graph
        self.kappa_gamma = kappa_gamma*obs_per_graph
        self.tol = tol
        self.lik_type = lik_type


        if np.isin(lik_type, ('skew-group-t', 'group-t')):
            if groups is None:
                raise ValueError("groups has to be given for skew-group-t and group-t")
            elif len(groups) != self.d:
                raise ValueError("groups length has to be same as number of features")
            else:
                self.groups = np.array(groups)
        else:
            self.groups = None

        self.obs_per_graph = int(obs_per_graph)
        self.w = self.obs_per_graph

        self.rho = float(obs_per_graph+1)
        self.rho_gamma = float(obs_per_graph+1)

        if type(self.lamda) is float:
            self.lamda = self.lamda*np.ones((self.d, self.d))
            np.fill_diagonal(self.lamda,0)

        if self.obs_per_graph <1:
            raise ValueError(f"obs_per_graph has to be 1 or larger")
        

    def get_nr_graphs(self):
        """
        Calculate number of graphs
        """
        if self.X is not None:
            if self.n % self.obs_per_graph:
                warnings.warn("Observations per graph estimation not divisiable by total number of observations")
            self.nr_graphs = int(np.ceil(self.n/self.obs_per_graph))
        else:
            self.nr_graphs = len(self.S)

        


    def get_X_index(self, i):
        """
        Function to get  window index of X

        Parameters
        ----------------------------
        i: int
            Graph number
        w: int
            window size
        l: int
            rolling window jump size. Only used of type is rolling-window
        type:str
            disjoint or rolling-window

        """

        lwr = self.obs_per_graph*i
        upr = self.obs_per_graph*(i+1)

        
        return lwr, upr
    
    def return_X(self, i):
        lwr, upr = self.get_X_index(i)

        if self.X is None:
            out = None
        else:
            out = self.X[lwr:upr].copy()
  
        return out
        


    def calc_S(self, method):
        """
        Calculation of the empirical covariance matrix

        Parameters
        ---------------------
        method: str,
            Method used to estimate the covariance
        X: numpy array,
            data matrix

        """
        if self.S is None:
            self.S = []

            if method == "empirical":
                for i in range(self.nr_graphs):
                    x_tmp = self.return_X(i)
                    if x_tmp.shape[0] == 1:
                        self.S.append(np.outer(x_tmp,x_tmp))
                    else:
                        self.S.append( np.cov(x_tmp.T))
            else:
                raise ValueError(f"No method for S called {method}")
        
    def calc_nu(self,liktype):

        if liktype != 'gaussian':

            if liktype == 't':
                nu = np.zeros(self.nr_graphs)
            else:
                nu = np.zeros((self.nr_graphs, self.d))


            for i in range(self.nr_graphs):
                x_tmp = self.return_X(i)

                nu_tmp = {}
                if liktype == 't':
                    kurt = np.mean(kurtosis(x_tmp, bias=False))
                    nu[i] = 6.0 / kurt + 4.0
                    nu[i] = np.min((np.max((nu[i],3)), 100))
                else:
                    for j in np.unique(self.groups):
                        kurt = np.mean(kurtosis(x_tmp[:, self.groups == j], bias=False))
                        nu_tmp[j] = 6.0 / kurt + 4.0
                        nu_tmp[j] = np.min((np.max((nu_tmp[j],4)), 100))
                    nu[i] = np.array([nu_tmp[j] for j in self.groups])
        else:
            nu = np.ones(self.nr_graphs)


        return nu
            






