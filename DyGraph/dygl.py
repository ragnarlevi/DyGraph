
import inspect
import numpy as np
import warnings
from scipy import optimize
import time







class dygl():


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
                x_tmp = X[i:(i+self.obs_per_graph)]
                if x_tmp.shape[0] == 1:
                    self.S.append(np.outer(x_tmp,x_tmp))
                else:
                    self.S.append( np.cov(x_tmp.T))
        else:
            raise ValueError(f"No method for S called {method}")
    
    def u_update(self):
        self.u0 = self.u0 + self.theta - self.z0
        self.u1[:(self.nr_graphs-1)] = self.u1[:(self.nr_graphs-1)] + self.theta[:(self.nr_graphs-1)]-self.z1[:(self.nr_graphs-1)]
        self.u2[1:] = self.u2[1:] + self.theta[1:] - self.z2[1:]


    def Gaussian_update(self, S, A, n_t):

        """
        Update according to gaussian likelihood
        """
        AT = A.T
        eta = n_t/self.rho/3.0
        M =  0.5*(A+AT)/eta - S
        D, Q = np.linalg.eig(M)
        diag_m = np.diag(D+np.sqrt(D**2 + 4.0/eta))
        return np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T))

    def get_A(self, i):
        if i == 0 or i == self.nr_graphs-1:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/2.0
        else:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/3.0
        return A


    def inner_em(self,X, i,lik_type = "t", nr_itr = 5, tol = 1e-5):
        """
        EM of theta
        """
        if lik_type == "t":

            d = X.shape[1]
            A = self.get_A(i)
            iteration = 0
            theta_pre = self.theta[i].copy()
            while iteration < nr_itr:
                # E-step
                x = X[i*self.obs_per_graph:((i+1)*self.obs_per_graph)]
                M = np.einsum('nj,jk,nk->n', x, theta_pre, x)  # Mahalanobis distance
                tau = (self.nu + d)/(self.nu  + M)
                S = np.einsum('nj,n,nk->jk', x,tau, x)/float(x.shape[0])
                # M-step
                theta_new = self.Gaussian_update(S,A, float(x.shape[0]))

                fro = np.linalg.norm(theta_new - theta_pre)
                if fro < tol:
                    theta_pre = theta_new.copy()
                    break
                else:
                    theta_pre = theta_new.copy()
                iteration+=1

            if iteration == nr_itr:
                warnings.warn("EM algorithm did not converge. Try to increase number of iterations")
            # print(f"{iteration} {fro}")
        else:
            raise ValueError(f"likelihood {lik_type} not known")


        return theta_new


    def theta_update(self,i, lik_type = "gaussian", X = None, **kwargs):
        """
        Theta update

        Parameters
        ----------------
        i: int,
            index of being updates
        lik_type: str,
            type of likelihood: gaussian, t
        """
        if lik_type == "gaussian":
            A = self.get_A(i)
            self.theta[i] = self.Gaussian_update(self.S[i],A, self.obs_per_graph_used[i])
        elif lik_type == "t":
            A = self.get_A(i)
            self.theta[i] = self.inner_em(X,i, nr_itr= kwargs.get("nr_em_itr",5))

        
        else:
            raise ValueError(f"likelihood {lik_type} not known")



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



    def fit(self, X, temporal_penalty, lik_type = "gaussian", time_index = None, **kwargs) -> None:
        """
        Fit a Dynamic Glasso

        Parameters
        --------------------

        X : array like,
            Data matrix with shape (n_observations, n_features)
        temporal_penalty: str,
            Which penality use for consecutive graphs: element-wise
        lik_type: strm
            Type of likelihood
        time_index: list, None
            Example a list of dates with the same length as X. Used to keep track of time of estimation
        kwargs:
            Arguments to likelihood or temporal penaltiy
        
        """

        if lik_type == "t":
            self.nu = kwargs.get("nu")

        if lik_type == "gaussian":
            self.calc_S(X, kwargs.get("S_method", "empirical"))

        # find obs_per_graph
        self.obs_per_graph_used = []
        for i in range(0, X.shape[0], self.obs_per_graph):
            x_tmp = X[i:(i+self.obs_per_graph)]
            self.obs_per_graph_used.append(x_tmp.shape[0])


        self.nr_graphs = len(range(0, X.shape[0], self.obs_per_graph))
        self.iteration = 0
        assert self.nr_graphs >1, "X.shape[0]/obs_per_graph has to be above 1 "
       

        self.u0 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        self.z0 = np.ones((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        self.theta = np.array([np.identity(X.shape[1]) for _ in range(self.nr_graphs) ])
        thetas_pre = self.theta.copy()

        if time_index is not None:
            self.graph_time = [time_index[k] for k in range(0, self.nr_graphs*self.obs_per_graph, self.obs_per_graph)]
            assert len(self.graph_time) == self.nr_graphs


        if X.shape[0] % self.obs_per_graph:
            warnings.warn("Observations per graph estimation not divisiable by total number of observations. Last observations not used.")
        while self.iteration < self.max_iter:

            # update theta
            st = time.time()
            for i in range(self.nr_graphs):
                self.theta_update(i, lik_type=lik_type, X = X, **kwargs)
            print(time.time()-st)

            
            # update z0
            for i in range(self.nr_graphs):
                self.z0[i] = self.soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda)
                np.fill_diagonal(self.z0[i], np.diag(self.theta[i]+self.u0[i]))
            
            # update z1 and z3
            for i in range(1,self.nr_graphs):
                A = self.theta[i]-self.theta[i-1]+self.u2[i]-self.u1[i-1]
                if temporal_penalty == "element-wise":
                    E = self.soft_threshold_odd(A, 2*self.kappa/self.rho)
                    self.z12_update(E,i) 
                elif temporal_penalty == "global-reconstruction":
                    E = self.global_reconstruction(A, 2*self.kappa/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "ridge":
                    E = self.ridge_penalty(A,2*self.kappa/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "block-wise-reconstruction":
                    E = self.block_wise_reconstruction(A,2*self.kappa/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "perturbed-node":
                    Y1,Y2 = self.perturbed_node(self.theta[i], self.theta[i-1], self.u2[i], self.u1[i-1], tol = kwargs.get('tol', 1e-4), n_iter =  kwargs.get('n_iter', 1000))
                    self.z1[i-1] = Y1
                    self.z2[i] = Y2
                else:
                    raise ValueError(f"{temporal_penalty} not a defined penalty function")

            # update u
            self.u_update()

            # check convergence
            self.fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_pre[i]
                self.fro_norm += np.linalg.norm(dif)
            if self.fro_norm < self.tol:
                break

            thetas_pre = self.theta.copy()
            self.iteration+=1

            if self.iteration == self.max_iter:
                warnings.warn("Max iterations reached.")
                

    
    def soft_threshold_odd(self,  A, lamda):

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


    def global_reconstruction(self, A, eta):
        """
        l2 group fused lasso

        Parameters
        ------------------
        A: np.array,
        
        lamda: float,
            regularization
        """
        
        # LOOP OVER COLUMNS
        E = np.zeros(shape=A.shape)
        for i in range(A.shape[1]):
            norm_val = np.sqrt(np.inner(A[:,i], A[:,i]))
            if norm_val <= eta:
                continue

            E[:,i] = (1.0 - eta/norm_val)*A[:,i]*(norm_val>eta)

        return E



    def ridge_penalty(self, A, eta):
        """
        Ridge/Laplacian penalty

        Parameters
        ------------------
        A: np.array,
        
        lamda: float,
            regularization
        """

        return A/(1.0+2.0*eta)


    def block_wise_reconstruction(self,A,  eta ):
        """
        Block-wise reconstruction: l_\infty norm

        Parameters
        ------------------
        A: np.array,
        
        lamda: float,
            regularization
        """
        def f(x, v):
            # Solve for hidden threshold

            return  np.sum([np.max([np.abs(v[i]) - x, 0]) for i in range(len(v))]) -1

        # LOOP OVER COLUMNS
        E = np.zeros(shape=A.shape)
        for i in range(A.shape[1]):
            if np.sum(np.abs(E[:,i])) <= eta:
                continue

            l_opt = optimize.bisect(f = f, a = 0, b = np.sum(np.abs(E[:,i]/eta)), args = (E[:,i]/eta,))

            E[:,i] = E[:,i] - eta*self.soft_threshold_odd(E[:,i]/eta, l_opt)

        return E


    def perturbed_node(self, theta_i, theta_i_1, U_i, U_i_1, tol = 1e-4, n_iter = 1000):
        """
        Block-wise reconstruction: l_\infty norm

        Parameters
        ------------------
        A: np.array,
        
        lamda: float,
            regularization
        """

        p = theta_i.shape
        Y1 = np.ones(shape = p )
        Y2 = np.ones(shape = p)
        V = np.ones(shape = p)
        W = np.ones(shape = p)
        U_tilde_1 = np.zeros(shape = p)
        U_tilde_2 = np.zeros(shape = p)

        count_it = 0
        while count_it < n_iter:

            A = (Y1-Y2-W-U_tilde_1 +W.T-U_tilde_2.T)/2
            V = self.global_reconstruction(A,self.kappa/(2.0*self.rho))

            I = np.identity(p[0])
            C = np.hstack((I,-I,I))
            D = V+U_tilde_1

            tmp_vec = np.vstack((V.T+U_tilde_2.T,  theta_i_1 + U_i_1, theta_i + U_i))

            out = np.dot(np.linalg.inv(np.dot(C.T,C)+2*np.identity(C.shape[1])), 2*tmp_vec - np.dot(C.T,D))

            W = out[:p[0]].copy()
            Y_1_pre =Y1.copy() 
            Y1 = out[p[0]:(2*p[0])].copy()
            Y2 = out[(2*p[0]):].copy()

            U_tilde_1 = U_tilde_1 + V + W - Y1 + Y2
            U_tilde_2 = U_tilde_2 + V - W.T

            dif = Y1 -Y_1_pre
            if n_iter >0:
                fro_norm = np.linalg.norm(dif)
                if  fro_norm < tol:
                    break

            count_it += 1


        if count_it == n_iter:
            print(f"ADMM for perturbed node did not converge w.r.t to tolerance. Last difference was {fro_norm}")

        return Y1, Y2

            









        