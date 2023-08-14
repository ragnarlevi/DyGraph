
import numpy as np
from DyGraph.dygl_utils import t_em, update_w, A_op, A_inv_op, L_inv_op, L_op, L_star, D_star, update_theta_laplace
from scipy.stats import kurtosis
import warnings







class sgl_laplace():


    def __init__(self, X, k, liktype = 'gaussian', tol = 1e-5, max_iter = 100 ) -> None:
        """
        X: data array n times d where n is number of obervations and d is number od nodes
        k: degree of each node, either number or vector
        liktype: gaussian or t 
        tol: relative tolerance
        max_iter: max iterations
        """

        self.X = X
        self.k = k
        self.liktype = liktype
        self.tol = tol
        self.max_iter = max_iter

        self.d = X.shape[1]
        self.n = X.shape[0]
        if ~hasattr(self, "obs_per_graph"):
            self.obs_per_graph = self.n


        self.get_nr_graphs()
            
    

            
    def calc_nu(self,liktype):

        if liktype != 'gaussian':

            if liktype == 't':
                nu = np.zeros(self.nr_graphs)
            else:
                nu = np.zeros((self.nr_graphs, self.d))


            for i in range(self.nr_graphs):
                x_tmp = self.return_X(i)

                nu_tmp = {}
                if liktype == 't':
                    kurt = np.mean(kurtosis(x_tmp, bias=False))
                    nu[i] = 6.0 / kurt + 4.0
                    nu[i] = np.min((np.max((nu[i],3)), 100))
                else:
                    for j in np.unique(self.groups):
                        kurt = np.mean(kurtosis(x_tmp[:, self.groups == j], bias=False))
                        nu_tmp[j] = 6.0 / kurt + 4.0
                        nu_tmp[j] = np.min((np.max((nu_tmp[j],4)), 100))
                    nu[i] = np.array([nu_tmp[j] for j in self.groups])
        else:
            nu = np.ones(self.nr_graphs)


        return nu


    def get_nr_graphs(self):
        """
        Calculate number of graphs
        """

        if self.n % self.obs_per_graph:
            warnings.warn("Observations per graph estimation not divisiable by total number of observations")
        self.nr_graphs = int(np.ceil(self.n/self.obs_per_graph))




    def fit(self, **kwargs):


        
        if not hasattr(self.k, "__len__"):
            self.k = np.array([self.k for _ in range(self.nr_graphs)])
        # elif  (len(self.k.shape) == 1):
            
        #     if (len(self.k) == self.nr_graphs):
        #         self.k = self.k
        #     elif (len(self.k) == self.d):
        #         self.k = np.array([self.k for _ in range(self.nr_graphs)])
        #     else:
        #         raise ValueError("shape of degree should be float, or equal to graph number, equal to graph node or  (nr_graphs, nr_nodes)")
        # elif (len(self.k.shape) == 2):
        #     if (self.k.shape[0]== self.nr_graphs) and (self.k.shape[1]== self.d):
        #         self.k = self.k
        #     else:
        #         raise ValueError("shape of degree should be float, or equal to graph number, equal to graph node or  (nr_graphs, nr_nodes)")

        if kwargs.get("nu", None) is None:
            self.nu = self.calc_nu(self.liktype)
        else:
            self.nu = kwargs.get("nu")


        self.S = np.zeros((1, self.d, self.d))
        S_inv_single = np.zeros((1, self.d, self.d))
        self.S[0] = np.corrcoef(self.X.T)
        S_inv_single[0] = np.linalg.pinv(self.S[0])

        nr_params = int(self.d*(self.d-1)/2)

        # number of nodes
        # w-initialization
        self.w = np.zeros((1, nr_params))
        self.w[0] = L_inv_op(S_inv_single[0])
        self.w[0][self.w[0] < 0] = 0

        A0 = A_op(self.w[0])
        A0 = A0/np.sum(A0, axis = 1)[:,None]
        self.w[0] = A_inv_op(A0)

        J = np.ones((self.d, self.d)) / self.d

        # Theta-initilization
        self.Lw = np.zeros((1, self.d, self.d))
        self.theta = np.zeros((1, self.d, self.d))

        self.Lw[0] = L_op(self.w[0])
        self.theta[0] = self.Lw[0].copy()
        Y = np.zeros((1,self.d, self.d))
        y = np.zeros((1, self.d))

        # ADMM constants
        self.mu = 2
        self.tau = 2
        self.rho = 1

        Lw_pre = self.Lw.copy()
        theta_pre = self.theta.copy()

        for i in range(self.max_iter): 

            # E-step
            if self.liktype =='t':
                self.S[0] = t_em(self.X, self.nu[-1], self.theta[0].copy())
           
            LstarS = L_star(self.S[0])


            # update w
            self.Lw[0], self.w[0], _ = update_w(self.Lw[0], self.w[0], self.rho, self.k[0], self.theta[0], LstarS, Y[0], y[0], 0)

            # update Theta
            self.theta[0], _ = update_theta_laplace(self.rho, J, self.Lw[0], Y[0], 0,0, 0, 1, False )

            
            # update Y
            R1 = self.theta[0] - self.Lw[0]
            Y[0] = Y[0] + self.rho * R1
            # update y
            R2 = np.diag(self.Lw[0]) - self.k[0]
            y[0] = y[0] + self.rho * R2
            # update rho
            self.s = self.rho * np.linalg.norm(L_star(theta_pre[0] - self.theta[0]))
            self.r = np.linalg.norm(R1, ord = "fro")
            if (self.r > self.mu * self.s):    
                self.rho = self.rho * self.tau
            elif (self.s > self.mu * self.r):

                self.rho = self.rho / self.tau

            self.iteration = i
            self.relnorm = np.linalg.norm(self.Lw[0] - Lw_pre[0], ord = "fro") / np.linalg.norm(Lw_pre[0], ord = "fro")

            has_converged = (self.relnorm < self.tol) & (i > 1)
            if (has_converged):
                break

            #w_pre = self.w.copy()
            Lw_pre = self.Lw.copy()
            theta_pre = self.theta.copy()



    def get_X_index(self, i):
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

        lwr = self.obs_per_graph*i
        upr = self.obs_per_graph*(i+1)

        
        return lwr, upr
    
    def return_X(self, i):
        lwr, upr = self.get_X_index(i)
        return self.X[lwr:upr]
    

    
    


    





