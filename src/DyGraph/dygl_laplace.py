
import numpy as np
from DyGraph.dygl_utils import t_em, soft_threshold_odd, global_reconstruction, ridge_penalty, block_wise_reconstruction, perturbed_node, update_w, update_theta_laplace, A_op, A_inv_op, L_inv_op, L_op, L_star, D_star
from scipy.stats import kurtosis
import warnings
from multiprocessing.pool import Pool
import tqdm
from decimal import Decimal
from DyGraph.sgl_laplace import sgl_laplace

class dygl_laplace(sgl_laplace):


    def __init__(self, X, k, obs_per_graph, kappa,   liktype = 'gaussian', tol = 1e-5, max_iter = 100 ) -> None:
        """
        X: data array n times d where n is number of obervations and d is number od nodes
        k: degree of each node, either number or vector
        liktype: gaussian or t 
        tol: relative tolerance
        max_iter: max iterations
        """


        sgl_laplace.__init__(self, X, k, liktype, tol, max_iter )

        self.obs_per_graph = obs_per_graph
        self.kappa = kappa
        self.get_nr_graphs()



    def fit(self, temporal_penalty, nr_workers = 1, verbose = True,  **kwargs):

        # Make sure k is in the right form
        self.transform_degree_param()

        if kwargs.get("nu", None) is None:
            self.nu = self.calc_nu(self.liktype)
        else:
            self.nu = kwargs.get("nu")


        self.S = np.zeros((self.nr_graphs, self.d, self.d))
        S_inv_single = np.zeros((self.nr_graphs, self.d, self.d))
        for t in range(self.nr_graphs):
            self.S[t] = np.cov(self.return_X(t).T)
            S_inv_single[t] = np.linalg.pinv(self.S[t])

        nr_params = int(self.d*(self.d-1)/2)

        # number of nodes
        # w-initialization
        self.w = np.zeros((self.nr_graphs, nr_params))
        self.W1 = np.zeros((self.nr_graphs, self.d, self.d))
        self.W2 = np.zeros((self.nr_graphs, self.d, self.d))
        for t in range(self.nr_graphs):
            self.w[t] = L_inv_op(S_inv_single[t])
            self.w[t][self.w[t] < 0] = 0

            A0 = A_op(self.w[t])
            self.w[t] = A_inv_op(A0)
            self.W1[t] = L_op(self.w[t])
            self.W2[t] = L_op(self.w[t])
        self.W1[-1] = np.zeros((self.d, self.d))
        self.W2[0] = np.zeros((self.d, self.d))



        J = np.ones((self.d, self.d)) / self.d

        # Theta-initilization
        self.Lw = np.zeros((self.nr_graphs, self.d, self.d))
        self.theta = np.zeros((self.nr_graphs, self.d, self.d))

        for t in range(self.nr_graphs):
            self.Lw[t] = L_op(self.w[t])
            self.theta[t] = self.Lw[t].copy()
        
        Y = np.zeros((self.nr_graphs, self.d, self.d))
        Y1 = np.zeros((self.nr_graphs, self.d, self.d))
        Y2 = np.zeros((self.nr_graphs, self.d, self.d))
        y = np.zeros((self.nr_graphs, self.d))

        # ADMM constants
        self.mu = 2
        self.tau = 2
        self.rho = self.obs_per_graph
        self.kappa = self.rho*self.kappa

        Lw_pre = self.Lw.copy()
        theta_pre = self.theta.copy()
        LstarS = np.zeros((self.nr_graphs, nr_params))


        if nr_workers >1:
            pool = Pool(nr_workers)
        else:
            pool = None

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)

        for i in range(self.max_iter): 

            # E-step, if applicable
            for t in range(self.nr_graphs):
                if self.liktype =='t':
                    self.S[t] = t_em(self.return_X(t), self.nu[-1], self.theta[0].copy())
            
                LstarS[t] = L_star(self.S[t])

            # update w
            if nr_workers == 1:
                for t in range(self.nr_graphs):
                    self.Lw[t], self.w[t], _ = update_w(self.Lw[t], self.w[t], self.rho, self.k[t], self.theta[t], LstarS[t], Y[t], y[t], t)
            else:
                results = pool.starmap(update_w,((self.Lw[t], self.w[t], self.rho, self.k[t], self.theta[t], LstarS[t], Y[t], y[t], t ) for t in range(self.nr_graphs)))
                for result in results:
                    self.Lw[result[2]] = result[0]
                    self.w[result[2]] = result[1]

            # update Theta
            if nr_workers == 1:
                for t in range(self.nr_graphs):
                    self.theta[t], _ = update_theta_laplace(self.rho, J, self.Lw[t], Y[t], self.W1[t],self.W2[t], t, self.nr_graphs, True )
            else:
                results = pool.starmap(update_theta_laplace,((self.rho, J, self.Lw[t], Y[t], self.W1[t], self.W2[t], t, self.nr_graphs, True ) for t in range(self.nr_graphs)))
                for result in results:
                    self.theta[result[1]] = result[0]
            

            # Update temporal
            for t in range(1,self.nr_graphs):
                A = self.theta[t] - self.theta[t-1] + Y2[t]/self.rho - Y1[t-1]/self.rho 
                summ = 0.5*(self.theta[t] + self.theta[t-1] + Y2[t]/self.rho + Y1[t-1]/self.rho )
                if temporal_penalty == "element-wise":
                    E = soft_threshold_odd(A, 2*self.kappa/self.rho)
                    self.W1[t-1] = summ - 0.5*E
                    self.W2[t] = summ + 0.5*E
                elif temporal_penalty == "global-reconstruction":
                    E = global_reconstruction(A, 2*self.kappa/self.rho)
                    self.W1[t-1] = summ - 0.5*E
                    self.W2[t] = summ + 0.5*E
                elif temporal_penalty == "ridge":
                    E = ridge_penalty(A, 2*self.kappa/self.rho)
                    self.W1[t-1] = summ - 0.5*E
                    self.W2[t] = summ + 0.5*E
                elif temporal_penalty == "block-wise-reconstruction":
                    E = block_wise_reconstruction(A,2*self.kappa/self.rho, kwargs.get('bwr_xtol', 1e-5))
                    self.W1[t-1] = summ - 0.5*E
                    self.W2[t] = summ + 0.5*E
                elif temporal_penalty == "perturbed-node":
                    Y1,Y2 = perturbed_node(self.theta[i], self.theta[i-1], self.W2[i], self.W1[i-1], self.kappa, self.rho, tol = kwargs.get('p_node_tol', 1e-5), max_iter =  kwargs.get('p_node_max_iter', 5000))
                    self.W1[t-1] = summ - 0.5*E
                    self.W2[t] = summ + 0.5*E
                else:
                    raise ValueError(f"{temporal_penalty} not a defined penalty function")

    
            # update Y
            for t in range(self.nr_graphs):
                R1 = self.theta[t] - self.Lw[t]
                Y[t] = Y[t] + self.rho * R1
                # update y
                R2 = np.diag(self.Lw[t]) - self.k[t]
                y[t] = y[t] + self.rho * R2
                # update Y1,Y2
                if t != self.nr_graphs-1:
                    #R3 = Thetai[t] - (np.diag(k[t])-W1[t]) 
                    R3 = self.theta[t] - self.W1[t] 
                    Y1[t] = Y1[t] + self.rho * R3
                if t != 0:
                    #R4 = Thetai[t] - (np.diag(k[t])-W2[t]) 
                    R4 = self.theta[t]-self.W2[t]
                    Y2[t] = Y2[t] + self.rho * R4

           

            self.iteration = i
            self.relnorm = np.sum([np.linalg.norm(self.theta[t] - theta_pre[t], ord = "fro") / np.linalg.norm(theta_pre[t], ord = "fro") for t in range(self.nr_graphs)])

            if verbose:   
                pbar.set_description(f"Error {Decimal(self.relnorm):.2E}")
                pbar.update()

            has_converged = (self.relnorm < self.tol) & (i > 1)
            if (has_converged):
                break

            #w_pre = self.w.copy()
            Lw_pre = self.Lw.copy()
            theta_pre = self.theta.copy()

        # terminate pool 
        if pool is not None:
            pool.terminate()

        if verbose:
            pbar.close()







