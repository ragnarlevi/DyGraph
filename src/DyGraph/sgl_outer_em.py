

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


    def __init__(self, X,  max_iter, lamda, obs_per_graph = None, S=None, lik_type = 'gaussian', tol = 1e-6, groups = None) -> None:

        """
        Parameters
        ------------------
        X: array of size n, d
            data with n observations and d features.
        max_iter: int,
            Maximum number of iterations
        
        lambda: float,
            regularization strength used for z0 l1 off diagonal 
        
        lik_type: str,
            Likelihood

        tol: float,
            Convergence tolerance.
        groups: numpy array of size d
            Grouping for EM
        
        
        """
        
        if X is None:
            obs_per_graph = obs_per_graph
        else:
            obs_per_graph = X.shape[0]

        RootDygl.__init__(self, X, obs_per_graph, max_iter, lamda, 0, S, 0, lik_type , tol, groups) 
        self.obs_per_graph = obs_per_graph

    def get_A(self):
        return self.z0[0] - self.u0[0]
    

    def u_update(self):
        self.u0 = self.u0 + self.theta - self.z0


    def fit(self, nr_workers = 1,theta_init = None, verbose = True, nr_admm_iter = 1, **kwargs):

        self.nr_graphs = 1
        self.calc_S(kwargs.get("S_method", "empirical"))
        self.nr_admm_iter = nr_admm_iter

        if kwargs.get("nu", None) is None:
            self.nu = self.calc_nu(self.lik_type)
        else:
            self.nu = kwargs.get("nu")

        if verbose:
            pbar1 = tqdm.tqdm(total = self.max_iter)

        # find obs_per_graph
        self.obs_per_graph_used = [float(self.obs_per_graph)]
  
        self.F_error = []
        self.iteration = 0



        self.u0 = np.zeros((self.nr_graphs, self.d, self.d))


        if theta_init is None:
            self.theta = np.array([np.identity(self.d) for _ in range(self.nr_graphs) ])
            self.z0 = np.ones((self.nr_graphs, self.d, self.d))
        else:
            self.theta = theta_init.copy()
            self.z0 = theta_init.copy()

        self.gamma = np.array([np.zeros(self.d) for _ in range(self.nr_graphs) ])
        
        thetas_pre = self.theta.copy()

        if nr_workers >1:
            pool = Pool(nr_workers)
        else:
            pool = None
        
        if self.nr_graphs< nr_workers:
            nr_workers = self.nr_graphs


        while self.iteration < self.max_iter:

            # Perform E-step

            if self.lik_type == 't':
                self.S[0] = t_em(self.X, self.nu[0], self.theta[0].copy())
            elif self.lik_type == 'group-t':
                self.S[0] = group_em(self.X, self.nu[0], self.theta[0].copy(), self.groups, kwargs.get("nr_quad", 5), pool)
            elif self.lik_type == 'skew-group-t':
                self.S[0], G1, G2 = skew_group_em(self.X, self.nu[0], self.theta[0].copy(), self.gamma[0], self.groups, kwargs.get("nr_quad", 5), pool)

            # if verbose:
            #     pbar2 = tqdm.tqdm(total = self.max_admm_iter, position = 2)
            admm_itr = 0
            while admm_itr < self.nr_admm_iter:
                eta = self.obs_per_graph_used[0]/self.rho/2.0
                self.theta[0],_ = Gaussian_update_outer_em(0, self.S[0], self.get_A(),  eta  )
                if self.lik_type == 'skew-group-t':
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

            self.F_error.append(self.fro_norm)

            if verbose:   
                pbar1.set_description(f"Error {Decimal(self.fro_norm):.2E}")
                pbar1.update()

            if self.fro_norm < self.tol:
                break

            thetas_pre = self.theta.copy()
            self.iteration+=1

        if self.iteration == self.max_iter:
            warnings.warn("Max iterations reached.")

        # terminate pool 
        if pool is not None:
            pool.terminate()

        if verbose:
            pbar1.close()
            # pbar2.close()





