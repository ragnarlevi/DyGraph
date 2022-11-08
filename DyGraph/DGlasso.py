
import inspect
import numpy as np
import warnings




class DGlasso():


    def __init__(self, obs_per_graph, max_iter, lamda, kappa, tol = 1e-6) -> None:

        """
        Parameters
        ------------------
        obs_per_graph: int,
            Observations used to construct each each matrix, can be 1 or larger


        max_iter: int,
            Maximum number of iterations
        
        lambda: float,
            regularization parameters used for z0 l1 off diagonal 

        kappa: float,
            regularization parameters used for z1 and z2 temporal penalties

        tol: float,
            Convergence tolerance.
        
        
        """
        assert obs_per_graph >= 0, "block size has to be bigger than on1"

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.obs_per_graph = int(obs_per_graph)

        self.rho = float(obs_per_graph+1)



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
            for i in range(0, X.shape[0], self.obs_per_graph):
                self.S.append( np.cov(X[i:(i+self.obs_per_graph)].T)) 
        else:
            raise ValueError(f"No method for S called {method}")
    
    def u_update(self):
        self.u0 = self.u0 + self.theta - self.z0
        self.u1[:(self.nr_graphs-1)] = self.u1[:(self.nr_graphs-1)] + self.theta[:(self.nr_graphs-1)]-self.z1[:(self.nr_graphs-1)]
        self.u2[1:] = self.u2[1:] + self.theta[1:] - self.z2[1:]


    def theta_update(self,i):
        """
        Theta update

        Parameters
        ----------------
        i: int,
            index of being updates
        """
        if i == 0 or i == self.nr_graphs-1:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/2.0
        else:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/3.0
        AT = A.T
        eta = self.obs_per_graph/self.rho
        M =  0.5*eta*(A+AT) - self.S[i]
        D, Q = np.linalg.eig(M)
        diag_m = np.diag(D+np.sqrt(D**2 + 4/eta))
        self.theta[i] = np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T))


    def z12_update(self, E, i):

        """
        Parameters
        --------------------
        E: np.array,
            affine transformation of Z

        i: int,
            index of being updates

        """
        summ = 0.5*(self.theta[i]+self.theta[i-1]+self.u1[i-1]+self.u2[i])
        self.z1[i-1] = summ - 0.5*E
        self.z2[i] = summ + 0.5*E



    def fit(self, X, temporal_penalty) -> None:
        """
        Fit a Dynamic Glasso

        Parameters
        --------------------

        X : array like,
            Data matrix with shape (n_observations, n_features)
        temporal_penalty: str,
            Which penality use for consecutive graphs: element-wise
        
        """
        self.iteration = 0
        self.nr_graphs = int(X.shape[0]/self.obs_per_graph)
        assert self.nr_graphs >1, "X.shape[0]/obs_per_graph has to be at least 2 "
        thetas_pre = np.zeros((self.nr_graphs, X.shape[1],X.shape[1]))

        # Calculate
        self.calc_S(X, "empirical")

        

        self.u0 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        self.z0 = np.ones((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        self.theta = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))


        if X.shape[0] % self.obs_per_graph:
            warnings.warn("Observations per graph estimation not divisiable by total number of observations. Last observations not used.")
        while self.iteration < self.max_iter:

            # update theta
            for i in range(self.nr_graphs):
                self.theta_update(i)
            
            # update z0
            for i in range(self.nr_graphs):
                self.z0[i] = self.soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda)
                np.fill_diagonal(self.z0[i], np.diag(self.theta[i]+self.u0[i]))
            
            # update z1 and z3
            for i in range(1,self.nr_graphs):
                if temporal_penalty == "element-wise":
                    E = self.soft_threshold_odd(self.theta[i]-self.theta[i-1]+self.u2[i]-self.u1[i-1], self.kappa)
                    self.z12_update(E,i)
                else:
                    raise ValueError(f"{temporal_penalty} not a defined penalty function")

            # update u
            self.u_update()

            # check convergence
            fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_pre[i]
                fro_norm += np.linalg.norm(dif)
            if fro_norm < self.tol:
                break

            thetas_pre = self.theta.copy()
            self.iteration+=1
                

    
    def soft_threshold_odd(self,  A, lamda):

        """
        Off-diagonal lasso penalty

        Parameters
        ------------------
        A: np.array,
        
        lamda: float,
            regularization
        """
        opt_m = (A-lamda)*(A>=lamda) + (A+lamda)*(A<=-lamda)
        

        return opt_m


            









        