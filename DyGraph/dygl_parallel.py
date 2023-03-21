

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

class dygl(RootDygl):


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
        RootDygl.__init__(self, obs_per_graph, max_iter, lamda, kappa, kappa_gamma , tol, l, X_type ) 



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




    def fit(self, X,temporal_penalty,theta_init = None, lik_type= "gaussian",  nr_workers = 1, verbose = True, True_prec = None, **kwargs):

        self.n = X.shape[0]
        self.get_nr_graphs()
        self.calc_S(X, kwargs.get("S_method", "empirical"))
        self.nu = kwargs.get("nu", None)
        self.groups = kwargs.get("groups", None)


        if  np.isin(lik_type, ('skew-group-t', 'group-t')) and  kwargs.get("groups", None) is None:
            raise ValueError("groups has to be given for skew-group-t and group-t")



        if (kwargs.get("nu", None) is None):
            self.nu = self.calc_nu(X,lik_type, kwargs.get("groups", None))

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)

        # find obs_per_graph
        self.obs_per_graph_used = []
        for i in range(0, self.nr_graphs):
            x_tmp = self.return_X(i, X)
            self.obs_per_graph_used.append(x_tmp.shape[0])

    

        self.F_error = []
        self.iteration = 0
        assert self.nr_graphs >1, "X.shape[0]/obs_per_graph has to be above 1"
       
        d = X.shape[1]

        self.u0 = np.zeros((self.nr_graphs, d, d))
        self.u1 = np.zeros((self.nr_graphs, d,d))
        self.u2 = np.zeros((self.nr_graphs, d, d))
        self.u3 = np.zeros((self.nr_graphs, d))
        self.u4 = np.zeros((self.nr_graphs, d))

        self.z3 = np.zeros((self.nr_graphs, d))
        self.z4 = np.zeros((self.nr_graphs, d))

        if theta_init is None:
            self.theta = np.array([np.identity(X.shape[1]) for _ in range(self.nr_graphs) ])
            self.z0 = np.ones((self.nr_graphs, d, d))
            self.z1 = np.zeros((self.nr_graphs, d,d))
            self.z2 = np.zeros((self.nr_graphs, d, d))
        else:
            self.theta = theta_init.copy()
            self.z0 = theta_init.copy()
            self.z1 = theta_init.copy()
            self.z1[-1] = np.zeros((d,d))
            self.z2 = theta_init.copy()
            self.z2[0] = np.zeros((d,d))



        self.gamma = np.array([np.zeros(X.shape[1]) for _ in range(self.nr_graphs) ])
        thetas_pre = self.theta.copy()

        pool = Pool(nr_workers)


        if not hasattr(self.kappa, "__len__"):
            self.kappa = np.array([self.kappa for _ in range(self.nr_graphs)])
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
                                                        lik_type,
                                                        self.return_X(i, X),
                                                        kwargs.get("nr_em_itr", 1),
                                                        self.theta[i].copy(),
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
                                                        lik_type,
                                                        self.return_X(i, X),
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
                self.z0[i] = soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda/self.rho)
                np.fill_diagonal(self.z0[i], np.diag(self.theta[i]+self.u0[i]))

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
                    E = block_wise_reconstruction(A,2*self.kappa[i-1]/self.rho)
                    self.z12_update(E,i)
                elif temporal_penalty == "perturbed-node":
                    Y1,Y2 = perturbed_node(self.theta[i], self.theta[i-1], self.u2[i], self.u1[i-1], self.kappa[i-1], self.rho, tol = kwargs.get('p_node_tol', 1e-5), max_iter =  kwargs.get('p_node_max_iter', 5000))
                    self.z1[i-1] = Y1
                    self.z2[i] = Y2
                else:
                    raise ValueError(f"{temporal_penalty} not a defined penalty function")
                
                if lik_type == 'skew-group-t':
                    A_gamma = self.gamma[i]-self.gamma[i-1]+self.u4[i]-self.u3[i-1]
                    E = soft_threshold_odd(A_gamma, 2*self.kappa_gamma[i-1]/self.rho_gamma)
                    summ = 0.5*(self.gamma[i]+self.gamma[i-1]+self.u3[i-1]+self.u4[i])
                    self.z3[i-1] = summ - 0.5*E
                    self.z4[i] = summ + 0.5*E

            

            # update u
            self.u_update()

            if True_prec is not None:
                F_score = np.mean([scipy.linalg.norm(True_prec[k]-self.theta[k], ord = 'fro')/scipy.linalg.norm(True_prec[k], ord = 'fro') for k in range(len(self.theta))])
                self.F_error.append(F_score)

            # check convergence
            self.fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_pre[i]
                self.fro_norm += np.linalg.norm(dif)/np.linalg.norm(thetas_pre[i])
            if self.fro_norm < self.tol:
                break

            if verbose:   
                if True_prec is not None:              
                    pbar.set_description(f"Error {Decimal(self.fro_norm):.2E}, F {Decimal(self.F_error[-1]):.3E}")
                else:
                    pbar.set_description(f"Error {Decimal(self.fro_norm):.2E}")
                pbar.update()

            thetas_pre = self.theta.copy()
            self.iteration+=1

        if self.iteration == self.max_iter:
            warnings.warn("Max iterations reached.")

        if verbose:
            pbar.close()






