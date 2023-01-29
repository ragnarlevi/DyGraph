





# Dynamic graph estimation in parallel

import inspect


from decimal import Decimal
import numpy as np
import warnings
from scipy import optimize
import time
import tqdm

from multiprocessing.pool import Pool






def Gaussian_update(S, A,  eta):

    """
    Update according to gaussian likelihood
    """
    AT = A.T
    M =  0.5*(A+AT)/eta - S
    D, Q = np.linalg.eig(M)
    diag_m = np.diag(D+np.sqrt(D**2 + 4.0/eta))
    return np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T))


def t_dist_update(i, A, v, x, nr_graphs, n_t, rho):

    if i == nr_graphs-1 or i == 0:
        eta = n_t/rho/2.0
    else:
        eta = n_t/rho/3.0
    S = np.einsum('nj,n,nk->jk', x, v, x)/float(x.shape[0])
    AT = A.T
    M =  0.5*(A+AT)/eta - S
    D, Q = np.linalg.eig(M)
    diag_m = np.diag(D+np.sqrt(D**2 + 4.0/eta))
    return np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T)),i


class dygl_outer_em():


    def __init__(self, obs_per_graph, nr_admm_its, nr_em_its, lamda, kappa, tol_em = 1e-6, tol_admm = 1e-6) -> None:

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


    def get_A(self, i):
        if i == 0 or i == self.nr_graphs-1:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/2.0
        else:
            A = (self.z0[i] + self.z1[i] + self.z2[i] - self.u0[i] - self.u1[i] - self.u2[i])/3.0
        return A

    def get_A_z(self, i):
       return self.theta[i]-self.theta[i-1]+self.u2[i]-self.u1[i-1]





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


    def fit(self, X,temporal_penalty,  nr_workers = 1, verbose = True, time_index = None, **kwargs):

        if verbose:
            pbar = tqdm.tqdm(total = self.nr_em_its)

        # find obs_per_graph
        self.obs_per_graph_used = []
        for i in range(0, X.shape[0], self.obs_per_graph):
            x_tmp = X[i:(i+self.obs_per_graph)]
            self.obs_per_graph_used.append(x_tmp.shape[0])

        d = X.shape[1]

        self.nr_graphs = len(range(0, X.shape[0], self.obs_per_graph))
        self.iteration = 0
        assert self.nr_graphs >1, "X.shape[0]/obs_per_graph has to be above 1"
       

        pool = Pool(nr_workers)
        if self.nr_graphs< nr_workers:
            nr_workers = self.nr_graphs
        if time_index is not None:
            self.graph_time = [time_index[k] for k in range(0, self.nr_graphs*self.obs_per_graph, self.obs_per_graph)]
            assert len(self.graph_time) == self.nr_graphs

        if X.shape[0] % self.obs_per_graph:
            warnings.warn("Observations per graph estimation not divisiable by total number of observations. Last observations not used.")

        self.em_it = 0

        def init_theta():
            A = np.random.uniform(size = (X.shape[1], X.shape[1]))
            return np.dot(A.T,A)/np.max(A)
        
        thetas_before_em = np.array([ init_theta() for _ in range(self.nr_graphs) ])

        self.u0 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.u2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        self.z0 = np.ones((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z1 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))
        self.z2 = np.zeros((self.nr_graphs, X.shape[1], X.shape[1]))

        while self.em_it  < self.nr_em_its:

            # Calculate EM-update
            M = np.concatenate([np.einsum('nj,jk,nk->n', X[self.obs_per_graph*k:(k+1)*self.obs_per_graph], thetas_before_em[k], X[self.obs_per_graph*k:(k+1)*self.obs_per_graph]) for k in range(self.nr_graphs)])  # Mahalanobis distance
            self.v = (kwargs.get('nu') + d)/(kwargs.get('nu')  + M)

            self.theta = thetas_before_em.copy()
            thetas_admm = self.theta.copy()

            self.iteration = 0
            while self.iteration < self.nr_admm_its:
                
                # update theta in parallel
                
                if nr_workers >1:
                    results = pool.starmap(t_dist_update,((i,
                                                            self.get_A(i), 
                                                            self.v[self.obs_per_graph*i:(i+1)*self.obs_per_graph], 
                                                            X[self.obs_per_graph*i:(i+1)*self.obs_per_graph],
                                                            self.nr_graphs,
                                                            self.obs_per_graph_used[i],
                                                            self.rho
                                                            ) for i in range(self.nr_graphs)))
                    for result in results:
                        self.theta[result[1]] = result[0]
                else:
                    for i in range(self.nr_graphs):
                        self.theta[i],_ = t_dist_update(i,
                                                            self.get_A(i), 
                                                            self.v[self.obs_per_graph*i:(i+1)*self.obs_per_graph], 
                                                            X[self.obs_per_graph*i:(i+1)*self.obs_per_graph],
                                                            self.nr_graphs,
                                                            self.obs_per_graph_used[i],
                                                            self.rho)


                # update dual in parallel
                # update z0
                for i in range(self.nr_graphs):
                    self.z0[i] = self.soft_threshold_odd(self.theta[i]+self.u0[i], self.lamda/self.rho)
                    np.fill_diagonal(self.z0[i], np.diag(self.theta[i]+self.u0[i]))

                # update z1, z2
                for i in range(1,self.nr_graphs):
                    A = self.theta[i]-self.theta[i-1]+self.u2[i]-self.u1[i-1]
                    if temporal_penalty == "element-wise":
                        E = self.soft_threshold_odd(A, 2*self.kappa/self.rho)
                        self.z12_update(E,i)
                    elif temporal_penalty == "global-reconstruction":
                        E = self.global_reconstruction(A, 2*self.kappa/self.rho)
                        self.z12_update(E,i)
                    elif temporal_penalty == "ridge":
                        E = self.ridge_penalty(A, 2*self.kappa/self.rho)
                        self.z12_update(E,i)
                    elif temporal_penalty == "block-wise-reconstruction":
                        E = self.block_wise_reconstruction(A,2*self.kappa/self.rho)
                        self.z12_update(E,i)
                    elif temporal_penalty == "perturbed-node":
                        Y1,Y2 = self.perturbed_node(self.theta[i], self.theta[i-1], self.u2[i], self.u1[i-1], tol = kwargs.get('p_node_tol', 1e-4), max_iter =  kwargs.get('p_node_max_iter', 1000))
                        self.z1[i-1] = Y1
                        self.z2[i] = Y2
                    else:
                        raise ValueError(f"{temporal_penalty} not a defined penalty function")

                # update u
                self.u_update()

                # check convergence ADMM
                self.fro_norm_admm = 0.0
                for i in range(self.nr_graphs):
                    dif = self.theta[i] - thetas_admm[i]
                    self.fro_norm_admm += np.linalg.norm(dif)
                if self.fro_norm_admm < self.tol_admm:
                    break

                thetas_admm = self.theta.copy()
                self.iteration+=1
            # if self.iteration == self.nr_admm_its:
            #     warnings.warn(f'Max admm iterations reached. Tolerance is {Decimal(self.fro_norm_admm):.2E}')

            # check convergence EM
            self.fro_norm = 0.0
            for i in range(self.nr_graphs):
                dif = self.theta[i] - thetas_before_em[i]
                self.fro_norm += np.linalg.norm(dif)
            if self.fro_norm < self.tol_em:
                break

            thetas_before_em = self.theta
            self.em_it += 1


            if verbose:
                pbar.set_description(f"Outer Error {Decimal(self.fro_norm):.2E}, Inner error {Decimal(self.fro_norm_admm):.2E}")
                pbar.update()


            
        if self.em_it == self.nr_em_its:
            warnings.warn(f'Max EM iterations reached. Tolerance is {Decimal(self.fro_norm):.2E}')
        if verbose:
            pbar.close()



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
            if np.sum(np.abs(A[:,i])) <= eta:
                continue

            l_opt = optimize.bisect(f = f, a = 0, b = np.sum(np.abs(A[:,i]/eta)), args = (A[:,i]/eta,))

            E[:,i] = A[:,i] - eta*self.soft_threshold_odd(A[:,i]/eta, l_opt)

        return E



    def perturbed_node(self, theta_i, theta_i_1, U_i, U_i_1, tol = 1e-4, max_iter = 1000):
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
        while count_it < max_iter:

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
            if max_iter >0:
                fro_norm = np.linalg.norm(dif)
                if  fro_norm < tol:
                    break

            count_it += 1


        if count_it == max_iter:
            print(f"ADMM for perturbed node reached maximum iterations. Last difference was {fro_norm}")
            

        return Y1, Y2




