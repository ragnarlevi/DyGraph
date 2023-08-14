

import numpy as np
from scipy.optimize import minimize



class CovReg():

    
    def __init__(self, Y, alpha, max_iter = 100, tol = 1e-6, do_all_iter = False) -> None:

        self.Y = Y
        self.max_iter = max_iter
        self.tol = tol

        self.d = Y.shape[1]
        self.n = Y.shape[0]

        self.alpha = self.n*alpha

        self.with_graph = False
        self.do_all_iter = do_all_iter



    @staticmethod
    def lasso_objective(param, X, Y,Psi_inv, d,r, alpha)->tuple:
        """"
        Objective and gradient to pass into the L-BFGS-B solver to solve the lasso problem.
        """
        C = np.reshape(param[:r*d]-param[r*d:],(d,r))
        obj =  np.trace(np.dot((Y- np.dot(X,C.T)).T,(Y- np.dot(X,C.T))).dot(Psi_inv)) + alpha*param.sum()
        t_0 = Y.T-np.dot(C,X.T)
        grad = (-0.5*np.dot(t_0,X)).flatten()
        return obj, np.concatenate((grad + alpha , - grad + alpha ), axis=None)
        

    def filter_x(self):
        pass

    def fit(self):
        """
        Wrapper
        """
        pass


    def one_iteration(self, Psi_pre_inv, psi, C_pre, c, Y_tilde, X1 = None, X2 = None, F_pre = None, with_mean = True):


        v = np.ones(self.n)
        m = np.ones(self.n)


        if with_mean:
            A = C_pre[:,:self.r1]
            B = C_pre[:,self.r1:]
        else:
            A = 0
            B = C_pre


        C_tmp = C_pre.flatten()
        C_vec = np.zeros(2*(self.r1+self.r2)*self.d)
        C_vec[:(self.r1+self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
        C_vec[(self.r1+self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)


        # E-step
        for i in range(self.n):
            if with_mean & (not self.with_graph):
                x_i = np.hstack((X1[i], X2[i]))
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, X1[i])).T,Psi_pre_inv).dot(C_pre).dot(x_i)
            elif (not with_mean) & (not self.with_graph):
                x_i = X2[i]
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-0).T,Psi_pre_inv).dot(C_pre).dot(x_i)
            elif with_mean & self.with_graph:
                x_i = np.hstack((F_pre[i], F_pre[i]))
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, F_pre[i])).T,Psi_pre_inv).dot(C_pre).dot(x_i)
            elif (not with_mean) & self.with_graph:
                x_i = F_pre[i]
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-0).T,Psi_pre_inv).dot(C_pre).dot(x_i)

        
        # M-step
        if with_mean & (not self.with_graph):
            X_tilde =  np.vstack((np.hstack((X1, m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*X2))))
        elif (not with_mean) & (not self.with_graph):
            X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))
        elif with_mean & self.with_graph:
            X_tilde =  np.vstack((np.hstack((F_pre, m[:,np.newaxis]*F_pre)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*F_pre))))
        elif (not with_mean) & self.with_graph:
            X_tilde =  np.vstack((m[:,np.newaxis]*F_pre,v[:,np.newaxis]*F_pre))


        if c is None:
            if self.alpha != 0:
                out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, Psi_pre_inv,self.d,self.r1+self.r2, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r1+self.r2)*self.d))
                C_vec = out.x
                C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
            else:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde) + 0.001*np.identity(self.r1+self.r2)))
        else:
            C = c

        #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
        #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)

        if psi is None:
            Psi_est = np.cov((Y_tilde-np.dot(X_tilde,C.T)).T)*(self.n-1)/self.n
        else:
            Psi_est = psi


        if with_mean:
            A = C[:,:self.r1]
            B = C[:,self.r1:]
        else:
            A = 0
            B = C


        if self.with_graph:
            if (self.F_method == 'direct') & with_mean:
                F = self.F_direct( B, A, m,v, Psi_pre_inv)
            elif (self.F_method == 'direct') & (not with_mean):
                F = self.F_direct_cov_only( B, m,v, Psi_pre_inv)
            elif (self.F_method == 'secant') & with_mean:
                F = self.F_secant_optim(B, A, m,v, Psi_pre_inv)
            elif (self.F_method == 'secant') & (not with_mean):
                F = self.F_secant_optim(B, np.zeros((self.d, self.r)), m,v, Psi_pre_inv)
            else:
                raise ValueError(f"F_method {self.F_method} not known")
        else:
            F = None


        if self.with_graph:
            tol_i = (np.linalg.norm(C-C_pre) + np.linalg.norm(F-F_pre))/(np.linalg.norm(F_pre) + np.linalg.norm(C_pre))
        else:
            tol_i = np.linalg.norm(C-C_pre)/np.linalg.norm(C_pre)


        return C, Psi_est, F, tol_i






    def fit_hoff_b_only(self, X2, psi = None, X_filter = None)-> None:
        """
        Hoff only fit covariance term
        """

        c = None

        self.r2 = X2.shape[1]
        self.r1 = 0
        
        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))

        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        
        C_pre = np.ones((self.d, self.r2))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:

            C, Psi_est, _, tol_i = self.one_iteration(np.linalg.inv(Psi_pre_inv), psi, C_pre, c, Y_tilde, X1 = None, X2 = X2, with_mean = False)
            self.tol_vec[self.iteration] = tol_i

            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()

        self.A = None
        self.B = C
        self.Psi = Psi_est

    def fit_hoff(self, X1, X2, psi = None, X_filter = None)-> None:
        """
        Hoff with lasso
        """

        c = None
        self.r1 = X1.shape[1]
        self.r2 = X2.shape[1]

        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))

        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        X_tilde =  np.vstack((np.hstack((X1, m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*X2))))
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r1+self.r2))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:
            C, Psi_est, _, tol_i = self.one_iteration(np.linalg.inv(Psi_pre_inv), psi, C_pre, c, Y_tilde, X1 = X1, X2 = X2, with_mean = True)
            self.tol_vec[self.iteration] = tol_i
            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()

        self.A = C[:,:self.r1]
        self.B = C[:,self.r1:]
        self.Psi = Psi_est


    def CCC(self, X, nr_its = 100, tol = 1e-6, mean_vec = np.zeros(R.shape[1])):
        def log_lik_cc(param, x, R, R_inv, sigma_start, mean_vec):
            d = R.shape[0]
            T = x.shape[0]
            alpha_0 = param[0]
            alpha_1 = param[1:(d+1)]
            beta_1 = param[-1]#param[-1]
            sigmas = np.zeros((T+1))
            sigmas[0] = sigma_start
            for t in range(T):
                sigmas[t+1] = alpha_0 + np.inner(alpha_1,x[t])**2 + beta_1*sigmas[t]
            
            Ms = np.einsum('nj,jk,nk->n', (x-mean_vec), R_inv, (x-mean_vec)  )

            obj = 0.5*d*np.sum(np.log(sigmas[1:T])) + 0.5*np.sum(Ms[1:]*np.reciprocal(sigmas[1:T])) # minimize this

            grad_2 = np.zeros(d)
            for t in range(1,T):
                grad_2 += d*np.reciprocal(sigmas[t])*np.inner(alpha_1, x[t-1])*x[t-1] - Ms[t]*np.reciprocal(sigmas[t])**2*np.inner(alpha_1, x[t-1])*x[t-1]

            

            grad_1 = 0.5*d*np.sum(np.reciprocal(sigmas[1:T])) - 0.5*np.sum(Ms[1:]* np.reciprocal(sigmas[1:T])**2 )  # grad w.r.t. alpha_0
            #grad_2 = d*np.sum((np.reciprocal(sigmas[1:T]) * np.dot(x[1:],alpha_1))[:, np.newaxis] * x[1:], axis = 0) - np.sum((Ms[1:]* np.reciprocal(sigmas[1:T])**2 * np.dot(x[1:],alpha_1))[:,np.newaxis]*x[1:], axis=0 )  # grad w.r.t. alpha_1
            grad_3 = 0.5*d*np.sum(np.reciprocal(sigmas[1:T])*sigmas[:T-1]) - 0.5*np.sum(Ms[1:]* np.reciprocal(sigmas[1:T])**2*sigmas[:T-1] )    # grad w.r.t. beta_1

            return obj, np.concatenate(([grad_1], grad_2, [grad_3]))
        


        R_est = np.identity(d)
        R_est_inv = np.linalg.inv(R_est)


        d = X.shape[1]
        params =  np.ones(d+2)*0.1
        params[0] = 0.2
        params[-1] = 0.5
        sigma_start = 0.1

        T = X.shpae[0]
        for it in range(nr_its):

            out = minimize(log_lik_cc, params, args = (X, R_est, R_est_inv, sigma_start, mean_vec), jac=True, method = 'L-BFGS-B', bounds = [(1e-6,1)] + [(1e-6,1)] + [(-1,1)]*(d-1) + [(1e-6,1)]) #

            old_params = params.copy()
            params = out.x
            sigmas = np.zeros((T+1))
            sigmas[0] = sigma_start
            for t in range(T):
                sigmas[t+1] = params[0] + np.inner(params[1:(d+1)],x[t])**2 + params[-1]*sigmas[t]


            R_est = np.einsum('ki,kj->ij', X*(1/sigmas[:T, np.newaxis]), X)/T
            R_est_inv = np.linalg.inv(R_est)

            print(np.linalg.norm(params-old_params))

