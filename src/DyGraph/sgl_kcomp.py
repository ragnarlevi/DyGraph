
import numpy as np
from DyGraph.dygl_utils import t_em, update_w, A_op, A_inv_op, L_inv_op, L_op, L_star, D_star, update_theta_laplace
from scipy.stats import kurtosis
import warnings
import tqdm
from decimal import Decimal
from sklearn.preprocessing import StandardScaler


def compute_student_weight(w, LstarSq, p, nu):
  return (p + nu) / (np.sum(w * LstarSq) + nu)




class sgl_kcomp():


    def __init__(self, X, k, liktype = 'gaussian', tol = 1e-5, max_iter = 100 ) -> None:
        """
        X: data array n times d where n is number of obervations and d is number od nodes
        k: degree of each node, either number or vector
        liktype: gaussian or t 
        tol: relative tolerance
        max_iter: max iterations
        """

        scaler = StandardScaler()

        self.X = scaler.fit_transform(X)
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




    def fit(self, nr_comp, eta = 1e-9, verbose = True, update_rho = False, update_eta = True, **kwargs):
        
        # Make sure k is in the right form
        self.transform_degree_param()

        if kwargs.get("nu", None) is None:
            self.nu = self.calc_nu(self.liktype)
        else:
            self.nu = kwargs.get("nu")

        LstarSq = []
        for i in range(self.n):
            LstarSq.append(L_star(np.outer(self.X[i],self.X[i]))/self.n )



        self.S = np.zeros((1, self.d, self.d))
        S_inv_single = np.zeros((1, self.d, self.d))
        self.S[0] = np.cov(self.X.T)
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


        # Theta-initilization
        self.Lw = np.zeros((1, self.d, self.d))
        self.theta = np.zeros((1, self.d, self.d))

        self.Lw[0] = L_op(self.w[0])
        self.theta[0] = self.Lw[0].copy()

        _, V = np.linalg.eigh(self.Lw[0])
        V = V[:,:nr_comp]



        Y = np.zeros((1,self.d, self.d))
        y = np.zeros((1, self.d))

        # ADMM constants
        self.mu = 2
        self.tau = 2
        self.rho = 1

        Lw_pre = self.Lw.copy()
        theta_pre = self.theta.copy()

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)


        for i in range(self.max_iter): 

            # E-step
            LstarSweighted = np.zeros(nr_params)
            for ij in range(self.n):
                if self.liktype =='t':
                    LstarSweighted += LstarSq[ij]*compute_student_weight(self.w[0], LstarSq[ij], self.d, self.nu[-1])# #LstarSq[ij]*(self.nu[-1]+self.d)/(np.sum(self.w[0]*LstarSq[ij])+self.nu[-1]) 
                else:
                    LstarSweighted += LstarSq[ij]
                
            # if self.liktype =='t':
            #     LstarSweighted =  L_star(t_em(self.X, self.nu[-1], self.theta[0].copy()))
            # else:
            #     LstarSweighted = L_star(self.S[0])
           
            # LstarS = L_star(self.S[0])

            # update w
            LstarLw = L_star(self.Lw[0])
            DstarDw = D_star(np.diag(self.Lw[0]))

            grad = LstarSweighted + L_star(eta*np.dot(V,V.T) -Y[0] - self.rho * self.theta[0] ) + D_star(y[0] - self.rho * self.k) + self.rho * (LstarLw + DstarDw)

            l_rate = 1 / (2*self.rho * (2*self.d - 1))
            self.w[0] = self.w[0] - l_rate * grad
            self.w[0][self.w[0] < 0] = 0
            self.Lw[0] = L_op(self.w[0])

            # update V
            _, V = np.linalg.eigh(self.Lw[0])
            V = V[:,:nr_comp]

            # update Theta
            gamma, U =  np.linalg.eigh(self.rho * self.Lw[0] - Y[0])
            gamma = gamma[::-1]
            gamma = gamma[:self.d-nr_comp]
            U = U[:,::-1]
            U = U[:,:self.d-nr_comp]
            self.theta[0] = np.dot(U,np.diag((gamma + np.sqrt(gamma**2 + 4 * self.rho)) / (2 * self.rho))).dot(U.T)

            
            # update Y
            R1 = self.theta[0] - self.Lw[0]
            Y[0] = Y[0] + self.rho * R1
            # update y
            R2 = np.diag(self.Lw[0]) - self.k[0]
            y[0] = y[0] + self.rho * R2
            # update rho
            if update_rho:
                self.s = self.rho * np.linalg.norm(L_star(theta_pre[0] - self.theta[0]))
                self.r = np.linalg.norm(R1, ord = "fro")
                if (self.r > self.mu * self.s):    
                    self.rho = self.rho * self.tau
                elif (self.s > self.mu * self.r):
                    self.rho = self.rho / self.tau

            self.iteration = i
            self.relnorm = np.linalg.norm(self.Lw[0] - Lw_pre[0], ord = "fro") / np.linalg.norm(Lw_pre[0], ord = "fro")

            if verbose:   
                pbar.set_description(f"Error {Decimal(self.relnorm):.2E}")
                pbar.update()

            if update_eta:
                eig_vals,_ = np.linalg.eigh(self.Lw[0])
                n_zero_eig_vals = np.sum(eig_vals<1e-8)
                if nr_comp < n_zero_eig_vals:
                    eta*=0.5
                elif nr_comp>n_zero_eig_vals:
                    eta*=2
            
            has_converged = (self.relnorm < self.tol) & (i > 1)
   
            if (has_converged):
                break
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
    

    def transform_degree_param(self):
                
        if not hasattr(self.k, "__len__"):
            self.k = np.array([self.k for _ in range(self.nr_graphs)])
        else:

            if len(self.k.shape) == 1:
                assert self.k.shape[0] == self.d, "If degree is a vector is should equal number of nodes"
                self.k = np.array([self.k for _ in range(self.nr_graphs)])
            else:
                assert (self.k.shape[0] == self.nr_graphs) and (self.k.shape[1] == self.d), "If degree is a matrix, the shape should be (number graphs, number of nodes)"


        

    
    


    





