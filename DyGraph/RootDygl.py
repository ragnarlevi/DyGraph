

# Root class for data preperation

import numpy as np
import inspect
import warnings
from scipy.stats import kurtosis



class RootDygl():

    def __init__(self, obs_per_graph, max_iter, lamda, kappa, kappa_gamma = 0, tol = 1e-6, l = None, X_type = 'disjoint') -> None:

        """
        Parameters
        ------------------
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
        l: int
            If X_type = rolling-window. l is the rolling window jumpt size
        X_type: str
            disjoint or rolling-window.
        
        
        """

        assert obs_per_graph >= 0, "block size has to be bigger than on1"

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.obs_per_graph = int(obs_per_graph)
        self.w = self.obs_per_graph
        self.l = l
        self.rho = float(obs_per_graph+1)
        self.rho_gamma = float(obs_per_graph+1)
        self.X_type = X_type

        if X_type == 'rolling-window' and l is None:
            raise ValueError(f"If X_type is {X_type} l has to be an integer")
        if self.obs_per_graph <1:
            raise ValueError(f"obs_per_graph has to be 1 or larger")
        if (l is not None) and l <1:
            raise ValueError(f"l has to be 1 or larger.")
        

    def get_nr_graphs(self):
        """
        Calculate number of graphs
        """
        if self.X_type == 'disjoint':
            if self.n % self.obs_per_graph:
                warnings.warn("Observations per graph estimation not divisiable by total number of observations")
            self.nr_graphs = int(np.ceil(self.n/self.obs_per_graph))
        elif self.X_type == 'rolling-window':
            if (self.n-self.obs_per_graph) % self.l:
                warnings.warn("Rolling window does not produce graphs with equal number of observations.")
            self.nr_graphs = int(np.ceil((self.n-self.obs_per_graph)/self.l +1))
        else:
            raise ValueError(f"X_type {self.X_type} not available. Use disjoint or rolling-window")
        


    def get_X_index(self, i, w, l = None, type = "disjoint"):
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
        if type == 'disjoint':
            lwr = w*i
            upr = w*(i+1)
        elif type == 'rolling-window':
            assert l is not None, f"parameter l can not be None"
            lwr = l*i
            upr = w+l*i
        else:
            raise ValueError(f"type {type} not available. Use disjoint or rolling-window")
        
        return lwr, upr
    
    def return_X(self, i, X):
        lwr, upr = self.get_X_index(i, self.obs_per_graph, self.l, self.X_type)
        return X[lwr:upr]
        


    def calc_S(self, X, method):
        """
        Calculation of the empirical covariance matrix

        Parameters
        ---------------------
        method: str,
            Method used to estimate the covariance
        X: numpy array,
            data matrix

        """
        X = np.array(X)
        self.S = []

        if method == "empirical":
            for i in range(self.nr_graphs):
                lwr, upr = self.get_X_index(i,self.obs_per_graph,self.l, self.X_type)
                x_tmp = X[lwr:upr]
                if x_tmp.shape[0] == 1:
                    self.S.append(np.outer(x_tmp,x_tmp))
                else:
                    self.S.append( np.cov(x_tmp.T))
        else:
            raise ValueError(f"No method for S called {method}")
        
    def calc_nu(self,X, liktype, groups = None):

        if liktype != 'gaussian':

            if liktype == 't':
                nu = np.zeros(self.nr_graphs)
            else:
                nu = np.zeros((self.nr_graphs, len(groups)))


            for i in range(self.nr_graphs):
                lwr, upr = self.get_X_index(i,self.obs_per_graph,self.l, self.X_type)
                x_tmp = X[lwr:upr]

                nu_tmp = {}
                if liktype == 't':
                    kurt = np.mean(kurtosis(x_tmp, bias=False))
                    nu[i] = 6.0 / kurt + 4.0
                    nu[i] = np.min((np.max((nu[i],3)), 100))
                else:
                    for j in np.unique(groups):
                        kurt = np.mean(kurtosis(x_tmp[:, groups == j], bias=False))
                        nu_tmp[j] = 6.0 / kurt + 4.0
                        nu_tmp[j] = np.min((np.max((nu_tmp[j],3)), 100))
                    nu[i] = np.array([nu_tmp[j] for j in groups])
        else:
            nu = np.ones(self.nr_graphs)


        return nu
            






