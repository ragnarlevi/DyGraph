

# Dynamic graph estimation in parallel

import inspect
from decimal import Decimal
import numpy as np
import warnings
import scipy
import tqdm
from multiprocessing.pool import Pool
from DyGraph.dygl_utils import theta_update, soft_threshold_odd, global_reconstruction, ridge_penalty, block_wise_reconstruction, perturbed_node
from DyGraph.RootDygl import RootDygl


class dygl_inner_em(RootDygl):


    def __init__(self, X, obs_per_graph, max_iter, lamda, kappa, S= None, kappa_gamma = 0, lik_type = 'gaussian', tol = 1e-6, groups = None) -> None:

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

        kappa: float,
            regularization strength used for z1 and z2 gamma temporal penalties

        kappa_gamma: float,
            regularization strength used for z3 and z4 gamma temporal penalties
        
        lik_type: str,
            Likelihood

        tol: float,
            Convergence tolerance.
        groups: numpy array of size d
            Grouping for EM
        
        
        """
        
        RootDygl.__init__(self, X, obs_per_graph, max_iter, lamda, kappa, S, kappa_gamma, lik_type , tol,  groups) 



    def get_A(self, i):
        if i == 0 or i == self.nr_graphs-1:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/2.0
        else:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/3.0
        return A
    
    def get_A_gamma(self,i):
        if i == 0 or i == self.nr_graphs-1:
            A = self.z3[i] + self.z4[i] - self.u3[i] - self.u4[i]
        else:
            A = (self.z3[i] + self.z4[i] - self.u3[i] - self.u4[i])/2.0
        return  A




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

    def u_update(self):
        self.u0 = self.u0 + self.theta - self.z0
        self.u1[:(self.nr_graphs-1)] = self.u1[:(self.nr_graphs-1)] + self.theta[:(self.nr_graphs-1)]-self.z1[:(self.nr_graphs-1)]
        self.u2[1:] = self.u2[1:] + self.theta[1:] - self.z2[1:]
        
        self.u3[:(self.nr_graphs-1)] = self.u3[:(self.nr_graphs-1)] + self.gamma[:(self.nr_graphs-1)]-self.z3[:(self.nr_graphs-1)]
        self.u4[1:] = self.u4[1:] + self.gamma[1:] - self.z4[1:]




    def fit(self, temporal_penalty,theta_init = None,  nr_workers = 1, verbose = True,  **kwargs):

        self.get_nr_graphs()
        self.calc_S(kwargs.get("S_method", "empirical"))

        if kwargs.get("nu", None) is None:
            self.nu = self.calc_nu(self.lik_type)
        else:
            self.nu = kwargs.get("nu")


        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)

        # find obs_per_graph
        self.obs_per_graph_used = []
        for i in range(0, self.nr_graphs):
            if self.X is not None:
                x_tmp = self.return_X(i)
                self.obs_per_graph_used.append(x_tmp.shape[0])
            else:
                self.obs_per_graph_used.append(self.obs_per_graph)

    

        self.F_error = []
        self.iteration = 0
        assert self.nr_graphs >1, "X.shape[0]/obs_per_graph has to be above 1"

        self.u0 = np.zeros((self.nr_graphs, self.d, self.d))
        self.u1 = np.zeros((self.nr_graphs, self.d, self.d))
        self.u2 = np.zeros((self.nr_graphs, self.d, self.d))
        self.u3 = np.zeros((self.nr_graphs, self.d))
        self.u4 = np.zeros((self.nr_graphs, self.d))

        self.z3 = np.zeros((self.nr_graphs, self.d))
        self.z4 = np.zeros((self.nr_graphs, self.d))

        if theta_init is None:
            self.theta = np.array([np.identity(self.d) for _ in range(self.nr_graphs) ])
            self.z0 = np.ones((self.nr_graphs, self.d, self.d))
            self.z1 = np.zeros((self.nr_graphs, self.d,self.d))
            self.z2 = np.zeros((self.nr_graphs, self.d, self.d))
        else:
            self.theta = theta_init.copy()
            self.z0 = theta_init.copy()
            self.z1 = theta_init.copy()
            self.z1[-1] = np.zeros((self.d,self.d))
            self.z2 = theta_init.copy()
            self.z2[0] = np.zeros((self.d,self.d))



        self.gamma = np.array([np.zeros(self.d) for _ in range(self.nr_graphs) ])
        thetas_pre = self.theta.copy()

        if nr_workers >1:
            pool = Pool(nr_workers)
        else:
            pool = None


        if not hasattr(self.kappa, "__len__"):
            self.kappa = np.array([self.kappa for _ in range(self.nr_graphs)])
        if not hasattr(self.lamda, "__len__"):
            self.lamda = np.array([self.lamda for _ in range(self.nr_graphs)])  
        elif  (len(self.lamda.shape) == 1) and (len(self.lamda) == self.nr_graphs):
            self.lamda = self.lamda
        elif (self.lamda.shape[0] == self.lamda.shape[1]) and len(self.lamda.shape) == 2:
            self.lamda = np.array([self.lamda for _ in range(self.nr_graphs)])  
        if not hasattr(self.kappa_gamma, "__len__"):
            self.kappa_gamma = np.array([self.kappa_gamma for _ in range(self.nr_graphs)])


        while self.iteration < self.max_iter:

            if self.nr_graphs< nr_workers:
                nr_workers = self.nr_graphs
            
            # update theta in parallel
            if nr_workers >1:
                results = pool.starmap(theta_update,((i,
                                                        self.get_A(i), 
                                                        self.S[i], 
                                                        self.obs_per_graph_used[i],
                                                        self.rho,
                                                        self.rho_gamma,
                                                        self.nr_graphs,
                                                        self.get_A_gamma(i),
                                                        self.groups,
                                                        self.lik_type,
                                                        self.return_X(i),
                                                        kwargs.get("nr_em_itr", 1),
                                                        self.theta[i],
                                                        self.gamma[i],
                                                        self.nu[i],
                                                        kwargs.get("em_tol", 1e-4),
                                                        kwargs.get("nr_quad", 5),
                                                        kwargs.get("pool", None)  ) for i in range(self.nr_graphs)))
                for result in results:
                    self.theta[result[2]] = result[0]
                    self.gamma[result[2]] = result[1]

            else:
                for i in range(self.nr_graphs):
                    
                    self.theta[i], self.gamma[i], _ = theta_update(i,
                                                        self.get_A(i), 
                                                        self.S[i], 
                                                        self.obs_per_graph_used[i],
                                                        self.rho, 
                                                        self.rho_gamma,
                                                        self.nr_graphs,
                                                        self.get_A_gamma(i),
                                                        self.groups,
                                                        self.lik_type,
                                                        self.return_X(i),
                                                        kwargs.get("nr_em_itr", 5),
                                                        self.theta[i],
                                                        self.gamma[i],
                                                        self.nu[i],
                                                        kwargs.get("em_tol", 1e-4),
                                                        kwargs.get("nr_quad", 5),
                                                        kwargs.get("pool", None)  )


            # update dual in parallel
            # update z0
            for i in range(self.nr_graphs):
                self.z0[i] = soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda[i]/self.rho)

            # update z1, z2, z3, z4
            for i in range(1,self.nr_graphs):
                A = self.theta[i]-self.theta[i-1]+self.u2[i]-self.u1[i-1]
                if temporal_penalty == "element-wise":
                    E = soft_threshold_odd(A, 2*self.kappa[i-1]/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "global-reconstruction":
                    E = global_reconstruction(A, 2*self.kappa[i-1]/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "ridge":
                    E = ridge_penalty(A,2*self.kappa[i-1]/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "block-wise-reconstruction":
                    E = block_wise_reconstruction(A,2*self.kappa[i-1]/self.rho, kwargs.get('bwr_xtol', 1e-5))
                    self.z12_update(E,i)
                elif temporal_penalty == "perturbed-node":
                    Y1,Y2 = perturbed_node(self.theta[i], self.theta[i-1], self.u2[i], self.u1[i-1], self.kappa[i-1], self.rho, tol = kwargs.get('p_node_tol', 1e-5), max_iter =  kwargs.get('p_node_max_iter', 5000))
                    self.z1[i-1] = Y1
                    self.z2[i] = Y2
                else:
                    raise ValueError(f"{temporal_penalty} not a defined penalty function")
                
                if self.lik_type == 'skew-group-t':
                    A_gamma = self.gamma[i]-self.gamma[i-1]+self.u4[i]-self.u3[i-1]
                    E = soft_threshold_odd(A_gamma, 2*self.kappa_gamma[i-1]/self.rho_gamma)
                    summ = 0.5*(self.gamma[i]+self.gamma[i-1]+self.u3[i-1]+self.u4[i])
                    self.z3[i-1] = summ - 0.5*E
                    self.z4[i] = summ + 0.5*E

            # update u
            self.u_update()

            # check convergence
            self.fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_pre[i]
                self.fro_norm += np.linalg.norm(dif)/np.linalg.norm(thetas_pre[i])
            self.F_error.append(self.fro_norm)

            if verbose:   
                pbar.set_description(f"Error {Decimal(self.fro_norm):.2E}")
                pbar.update()

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
            pbar.close()






