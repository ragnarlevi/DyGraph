

# Dynamic graph estimation in parallel

import inspect
from decimal import Decimal
import numpy as np
import warnings
import tqdm
from multiprocessing.pool import Pool
from DyGraph.dygl_utils import theta_update, soft_threshold_odd, global_reconstruction, ridge_penalty, block_wise_reconstruction, perturbed_node, t_em, group_em, skew_group_em, Gaussian_update_outer_em
from DyGraph.RootDygl import RootDygl



def update_gamma(i, G1, G2, theta, rho_gamma, A_gamma):
    d = theta.shape[0]
    A = np.dot(np.linalg.inv(np.multiply(theta, G2)+rho_gamma*np.identity(d)), G1 +rho_gamma*A_gamma)

    return A, i

class dygl_outer_em(RootDygl):


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

    def correct_eta(self,i):
        if i == self.nr_graphs-1 or i == 0:
            eta = self.obs_per_graph_used[i]/self.rho/2.0
        else:
            eta = self.obs_per_graph_used[i]/self.rho/3.0
        return eta


    def fit(self, X,temporal_penalty,theta_init = None, lik_type= "gaussian",  nr_workers = 1, verbose = True, max_admm_iter = 1, **kwargs):

        self.n = X.shape[0]
        self.get_nr_graphs()
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

        if nr_workers >1:
            pool = Pool(nr_workers)
        else:
            pool = None
        
        if self.nr_graphs< nr_workers:
            nr_workers = self.nr_graphs


        if not hasattr(self.kappa, "__len__"):
            self.kappa = np.array([self.kappa for _ in range(self.nr_graphs)])
        if not hasattr(self.kappa_gamma, "__len__"):
            self.kappa_gamma = np.array([self.kappa_gamma for _ in range(self.nr_graphs)])

    
        while self.iteration < self.max_iter:

            # Perform E-step
            if lik_type == 't':
                for g_i in range(self.nr_graphs):
                    self.S[g_i] = t_em(self.return_X(g_i, X), self.nu[g_i], self.theta[g_i].copy())
            elif lik_type == 'group-t':
                for g_i in range(self.nr_graphs):
                    self.S[g_i] = group_em(self.return_X(g_i, X), self.nu[g_i], self.theta[g_i].copy(), self.groups, kwargs.get("nr_quad", 5), pool)
            elif lik_type == 'skew-group-t':
                G1 = [None] * self.nr_graphs
                G2 = [None] * self.nr_graphs
                for g_i in range(self.nr_graphs):
                    self.S[g_i], G1[g_i], G2[g_i] = skew_group_em(self.return_X(g_i, X), self.nu[g_i], self.theta[g_i].copy(), self.gamma[g_i], self.groups, kwargs.get("nr_quad", 5), pool)

            admm_itr = 0
            # if verbose:
            #     pbar2 = tqdm.tqdm(total = self.max_admm_iter, position=2)
            while admm_itr < self.max_admm_iter:
                
            
                # update theta in parallel
                if nr_workers >1:
                    results = pool.starmap(Gaussian_update_outer_em,((i, self.S[i], self.get_A(i),  self.correct_eta(i)  ) for i in range(self.nr_graphs)))
                    for result in results:
                        self.theta[result[1]] = result[0]
                else:
                    for i in range(self.nr_graphs):
                        self.theta[i],_ = Gaussian_update_outer_em(i, self.S[i], self.get_A(i),  self.correct_eta(i)  )

                # if skew update gamma
                if lik_type == 'skew-group-t':
                    if nr_workers >1:
                        # i, G1, G2, theta, rho_gamma, A_gamma
                        results = pool.starmap(update_gamma,((i, G1[i], G2[i], self.theta[i], self.rho_gamma, self.get_A_gamma(i) ) for i in range(self.nr_graphs)))
                        for result in results:
                            self.gamma[result[1]] = result[0]
                    else:
                        for i in range(self.nr_graphs):
                            self.gamma[i], _ = update_gamma(i, G1[i], G2[i], self.theta[i], self.rho_gamma, self.get_A_gamma(i) )


                # update dual in parallel
                # update z0
                for i in range(self.nr_graphs):
                    self.z0[i] = soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda/self.rho)


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
                    
                    if lik_type == 'skew-group-t':
                        A_gamma = self.gamma[i]-self.gamma[i-1]+self.u4[i]-self.u3[i-1]
                        E = soft_threshold_odd(A_gamma, 2*self.kappa_gamma[i-1]/self.rho_gamma)
                        summ = 0.5*(self.gamma[i]+self.gamma[i-1]+self.u3[i-1]+self.u4[i])
                        self.z3[i-1] = summ - 0.5*E
                        self.z4[i] = summ + 0.5*E

            
                # update u
                self.u_update()
                # pbar2.update()
                admm_itr+=1


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





