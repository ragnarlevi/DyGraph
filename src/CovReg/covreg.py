

import numpy as np
from scipy.optimize import minimize
import tqdm
from decimal import Decimal
from scipy.linalg import cholesky
from sklearn import linear_model


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
    def lasso_objective(param, X, Y,Psi_inv, d,r, alpha, a = None)->tuple:
        """"
        Objective and gradient to pass into the L-BFGS-B solver to solve the lasso problem.
        """
        if a is not None:
            C = np.hstack((a, np.reshape(param[:r*d]-param[r*d:],(d,r))))
        else:
            C = np.reshape(param[:r*d]-param[r*d:],(d,r))
        
        obj =  np.trace(np.dot((Y- np.dot(X,C.T)).T,(Y- np.dot(X,C.T))).dot(Psi_inv)) + alpha*param.sum()
        t_0 = Y.T-np.dot(C,X.T)
        if a is None:
            grad = (-0.5*np.dot(t_0,X)).flatten()
        else:
            grad = (-0.5*np.dot(t_0,X)).flatten()[:d*r]
        return obj, np.concatenate((grad + alpha , - grad + alpha ), axis=None)
        

    def filter_x(self):
        pass

    def fit(self):
        """
        Wrapper
        """
        pass


    def one_iteration(self, Psi_pre_inv, psi, C_pre, c, Y_tilde, a = None, X1 = None, X2 = None, F_pre = None, with_mean = True):


        v = np.zeros(self.n)
        m = np.zeros(self.n)



        if with_mean:

            A = C_pre[:,:self.r1]
            B = C_pre[:,self.r1:]
        else:
            A = 0
            B = C_pre

    
        if a is None:
            C_tmp = C_pre.flatten()
            C_vec = np.zeros(2*(self.r1+self.r2)*self.d)
            C_vec[:(self.r1+self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
            C_vec[(self.r1+self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)
        else:
            C_tmp = C_pre[:,self.r1:].flatten()
            C_vec = np.zeros(2*(self.r2)*self.d)
            C_vec[:(self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
            C_vec[(self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)


        # E-step
        for i in range(self.n):
            if with_mean & (not self.with_graph):
                v[i] = (1+np.dot(X2[i].T, B.T).dot(Psi_pre_inv).dot(B).dot(X2[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, X1[i])).T,Psi_pre_inv).dot(B).dot(X2[i])
            elif (not with_mean) & (not self.with_graph):
                v[i] = (1+np.dot(X2[i].T, B.T).dot(Psi_pre_inv).dot(B).dot(X2[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]).T,Psi_pre_inv).dot(B).dot(X2[i])
            elif with_mean & self.with_graph:
                v[i] = (1+np.dot(F_pre[i].T, B.T).dot(Psi_pre_inv).dot(B).dot(F_pre[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(B, F_pre[i])).T,Psi_pre_inv).dot(B).dot(F_pre[i])
            elif (not with_mean) & self.with_graph:
                v[i] = (1+np.dot(F_pre[i].T, B.T).dot(Psi_pre_inv).dot(B).dot(F_pre[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]).T,Psi_pre_inv).dot(B).dot(F_pre[i])

        
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
                if a is None:
                    out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, np.identity(self.d),self.d,self.r1+self.r2, self.alpha, None), method='L-BFGS-B', jac=True, bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d))
                    C_vec = out.x
                    C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
                    # clf = linear_model.MultiTaskLasso(alpha=self.alpha/self.n)
                    # clf.fit(X_tilde, Y_tilde)
                    # C = clf.coef_
                else:
                    out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, np.identity(self.d),self.d,self.r2, self.alpha, a), method='L-BFGS-B', jac=True, bounds = [(0,None)]*(2*(self.r2)*self.d))
                    C_vec = out.x
                    C = np.reshape(out.x[:(self.r2)*self.d] - out.x[(self.r2)*self.d:], (self.d,self.r2))
                    C = np.hstack((a, C))
            else:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde) + 1e-6*np.identity(self.r1+self.r2)))
                if a is not None:
                    C = np.hstack((a, C[:,self.r1:]))
        else:
            C = c

        #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
        #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)

        if psi is None:
            Psi_est = np.dot((Y_tilde-np.dot(X_tilde,C.T)).T, (Y_tilde-np.dot(X_tilde,C.T)))/self.n
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
                F = self.F_direct( B, A, m,v, np.identity(self.d))
            elif (self.F_method == 'direct') & (not with_mean):
                F = self.F_direct_cov_only( B, m,v, np.identity(self.d))
            elif (self.F_method == 'secant') & with_mean:
                F = self.F_secant_optim(B, A, m,v, np.identity(self.d))
            elif (self.F_method == 'secant') & (not with_mean):
                F = self.F_secant_optim(B, np.zeros((self.d, self.r)), m,v, np.identity(self.d))
            else:
                raise ValueError(f"F_method {self.F_method} not known")
        else:
            F = None


        if self.with_graph:
            tol_i = (np.linalg.norm(C-C_pre) + np.linalg.norm(F-F_pre))/(np.linalg.norm(F_pre) + np.linalg.norm(C_pre))
        else:
            tol_i = np.linalg.norm(C-C_pre)/np.linalg.norm(C_pre)
        
        tol_i = 1

    


        return C, Psi_est, F, tol_i






    def fit_hoff_b_only(self, X2, psi = None, X_filter = None, verbose = True)-> None:
        """
        Hoff only fit covariance term
        """

        c = None

        self.r2 = X2.shape[1]
        self.r1 = 0
        
        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))

        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre+1e-5)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)


        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        m = np.ones(self.n)# np.random.normal(loc = 0, scale=1, size = self.n)
        v = np.ones(self.n)
        X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))
        C_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde) + 1e-6*np.identity(self.r1+self.r2)))
        for _ in range(100):

            for i in range(self.n):
                v[i] = (1+np.dot(X2[i].T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(X2[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]).T,Psi_pre_inv).dot(C_pre).dot(X2[i])

            C_pre = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde) + 1e-6*np.identity(self.r1+self.r2)))

        if psi is None:
            Psi_pre = np.dot((Y_tilde-np.dot(X_tilde,C_pre.T)).T, (Y_tilde-np.dot(X_tilde,C_pre.T)))/self.n
            Psi_pre_inv = np.linalg.inv(Psi_pre+1e-6)
        else:
            Psi_pre = psi
            Psi_pre_inv = Psi_pre_inv

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)
        
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:

            C, Psi_est, _, tol_i = self.one_iteration(Psi_pre_inv, psi, C_pre, c, Y_tilde, X1 = None, X2 = X2, with_mean = False)
            self.tol_vec[self.iteration] = tol_i

            if verbose:
                pbar.set_description(f"Error {Decimal(tol_i):.2E}")
                pbar.update()

            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()
                Psi_pre_inv = np.linalg.inv(Psi_pre+1e-5)

            C_pre = C.copy()


        self.A = None
        self.B = C
        self.Psi = Psi_est
        if verbose:
            pbar.close()

    def fit_hoff(self, X1, X2, psi = None, X_filter = None, verbose = True)-> None:
        """
        Hoff with lasso
        """

        c = None
        self.r1 = X1.shape[1]
        self.r2 = X2.shape[1]

        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))

        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre+1e-3)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)


        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r1+self.r2))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:
            C, Psi_est, _, tol_i = self.one_iteration(Psi_pre_inv, psi, C_pre, c, Y_tilde, X1 = X1, X2 = X2, with_mean = True)
            self.tol_vec[self.iteration] = tol_i

            if verbose:
                pbar.set_description(f"Error {Decimal(tol_i):.2E}")
                pbar.update()
            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()
                Psi_pre_inv = np.linalg.inv(Psi_pre+1e-3)

            C_pre = C.copy()





        self.A = C[:,:self.r1]
        self.B = C[:,self.r1:]
        self.Psi = Psi_est
        if verbose:
                pbar.close()

    def marg_lik(self, X1 = None, X2 = None):

        v = np.ones(self.n)
        m = np.ones(self.n)

        psi_inv = np.linalg.inv(self.Psi+1e-6)


        if self.A is None:
            C = self.B
        else:
            C = np.hstack((self.A, self.B))
        

        # E-step
        for i in range(self.n):
            
            if self.A is not None:
                v[i] = (1+np.dot(X2[i], self.B.T).dot(psi_inv).dot(self.B).dot(X2[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(self.A, X1[i])).T,psi_inv).dot(self.B).dot(X2[i])
            else:
                v[i] = (1+np.dot(X2[i], self.B.T).dot(psi_inv).dot(self.B).dot(X2[i])) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]).T,psi_inv).dot(self.B).dot(X2[i])

        
        # M-step

        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))
        if self.A is not None:
            X_tilde =  np.vstack((np.hstack((X1, m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*X2))))
        else:
            X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))


        # L = cholesky(self.Psi)
        try:
            v, _ = np.linalg.eigh(self.Psi)
            v = v[v>1e-6]
            obj = 0.5*np.trace(np.dot((Y_tilde-np.dot(X_tilde, C.T)).T, (Y_tilde-np.dot(X_tilde, C.T))).dot(psi_inv)) + 0.5*self.n*np.sum(np.log(v)) + self.n*self.d*np.log(2*np.pi)
        except:
            obj = np.nan

        nr_params = np.sum(np.abs(C)>1e-4)
        
        return obj, nr_params



    def CCC(self, X, nr_its = 100, tol = 1e-6, mean_vec = None):

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

        if mean_vec is None:
            mean_vec = np.zeros(R_est.shape[1])


        d = X.shape[1]
        params =  np.ones(d+2)*0.1
        params[0] = 0.2
        params[-1] = 0.5
        sigma_start = 0.1

        T = X.shpae[0]
        for it in range(nr_its):

            out = minimize(log_lik_cc, params, args = (X, R_est, R_est_inv, sigma_start, mean_vec), jac=True, method = 'L-BFGS-B', D = [(1e-6,1)] + [(1e-6,1)] + [(-1,1)]*(d-1) + [(1e-6,1)]) #

            old_params = params.copy()
            params = out.x
            sigmas = np.zeros((T+1))
            sigmas[0] = sigma_start
            for t in range(T):
                sigmas[t+1] = params[0] + np.inner(params[1:(d+1)],X[t])**2 + params[-1]*sigmas[t]


            R_est = np.einsum('ki,kj->ij', X*(1/sigmas[:T, np.newaxis]), X)/T
            R_est_inv = np.linalg.inv(R_est)

            print(np.linalg.norm(params-old_params))

