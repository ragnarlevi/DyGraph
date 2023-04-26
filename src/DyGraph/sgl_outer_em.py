

# Dynamic graph estimation in parallel

from decimal import Decimal
import numpy as np
import warnings
import tqdm
from multiprocessing.pool import Pool
from DyGraph.dygl_utils import soft_threshold_odd, t_em, group_em, skew_group_em, Gaussian_update_outer_em
from DyGraph.RootDygl import RootDygl



def update_gamma_static(G1, G2, theta):
    return np.dot(np.linalg.inv(np.multiply(theta, G2)), G1 )


class sgl_outer_em(RootDygl):


    def __init__(self, max_iter, lamda, tol = 1e-6) -> None:

        """
        Parameters
        ------------------
        obs_per_graph: int,
            Observations used to construct each each matrix, can be 1 or larger


        max_iter: int,
            Maximum number of iterations
        
        lambda: float,
            regularization strength used for z0 l1 off diagonal 

        kappa: float,
            regularization strength used for z1 and z2 temporal penalties

        kappa: float,
            regularization strength used for z3 and z4 gamma temporal penalties

        tol: float,
            Convergence tolerance.
        l: int
            If X_type = rolling-window. l is the rolling window jumpt size
        X_type: str
            disjoint or rolling-window.
        
        
        """
        RootDygl.__init__(self, 1, max_iter, lamda, 0, 0 , tol, None, 'disjoint' ) 



    def get_A(self):
        return self.z0[0] - self.u0[0]
    


    def u_update(self):
        self.u0 = self.u0 + self.theta - self.z0


    def fit(self, X, nr_workers = 1,theta_init = None, lik_type= "gaussian", verbose = True, max_admm_iter = 1, **kwargs):

        self.n = X.shape[0]
        self.nr_graphs = 1
        self.obs_per_graph = self.n
        self.calc_S(X, kwargs.get("S_method", "empirical"))
        self.nu = kwargs.get("nu", None)
        self.groups = kwargs.get("groups", None)
        self.max_admm_iter = max_admm_iter

        if type(self.lamda) is float:
            self.lamda = self.lamda*np.ones((X.shape[1], X.shape[1]))
            np.fill_diagonal(self.lamda,0)


        if  np.isin(lik_type, ('skew-group-t', 'group-t')) and  kwargs.get("groups", None) is None:
            raise ValueError("groups has to be given for skew-group-t and group-t")



        if (kwargs.get("nu", None) is None):
            self.nu = self.calc_nu(X,lik_type, kwargs.get("groups", None))

        if verbose:
            pbar1 = tqdm.tqdm(total = self.max_iter)

        # find obs_per_graph
        self.obs_per_graph_used = [float(X.shape[0])]
        self.rho = float(X.shape[0])

    

        self.F_error = []
        self.iteration = 0

        d = X.shape[1]

        self.u0 = np.zeros((self.nr_graphs, d, d))


        if theta_init is None:
            self.theta = np.array([np.identity(X.shape[1]) for _ in range(self.nr_graphs) ])
            self.z0 = np.ones((self.nr_graphs, d, d))
        else:
            self.theta = theta_init.copy()
            self.z0 = theta_init.copy()

        self.gamma = np.array([np.zeros(X.shape[1]) for _ in range(self.nr_graphs) ])
        
        thetas_pre = self.theta.copy()

        if nr_workers >1:
            pool = Pool(nr_workers)
        else:
            pool = None
        
        if self.nr_graphs< nr_workers:
            nr_workers = self.nr_graphs


        while self.iteration < self.max_iter:

            # Perform E-step

            if lik_type == 't':
                self.S[0] = t_em(X, self.nu[0], self.theta[0].copy())
            elif lik_type == 'group-t':
                self.S[0] = group_em(X, self.nu[0], self.theta[0].copy(), self.groups, kwargs.get("nr_quad", 5), pool)
            elif lik_type == 'skew-group-t':
                self.S[0], G1, G2 = skew_group_em(X, self.nu[0], self.theta[0].copy(), self.gamma[0], self.groups, kwargs.get("nr_quad", 5), pool)

            # if verbose:
            #     pbar2 = tqdm.tqdm(total = self.max_admm_iter, position = 2)
            admm_itr = 0
            while admm_itr < self.max_admm_iter:
                eta = self.obs_per_graph_used[0]/self.rho/2.0
                self.theta[0],_ = Gaussian_update_outer_em(0, self.S[0], self.get_A(),  eta  )
                if lik_type == 'skew-group-t':
                    self.gamma[0] = update_gamma_static(G1, G2, self.theta[0] )


                # update dual in parallel
                # update z0
                self.z0[0] = soft_threshold_odd(self.theta[0]+self.u0[0], self.lamda/self.rho)
                # np.fill_diagonal(self.z0[0], np.diag(self.theta[0]+self.u0[0]))

                # update u
                self.u_update()
                admm_itr += 1
                # if verbose:
                #     pbar2.update()


            # check convergence
            self.fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_pre[i]
                self.fro_norm += np.linalg.norm(dif)/np.linalg.norm(thetas_pre[i])

            if verbose:   
                pbar1.set_description(f"Error {Decimal(self.fro_norm):.2E}")
                pbar1.update()

            if self.fro_norm < self.tol:
                break

            thetas_pre = self.theta.copy()
            self.iteration+=1

        if self.iteration == self.max_iter:
            warnings.warn("Max iterations reached.")

        if verbose:
            pbar1.close()
            # pbar2.close()





