

import numpy as np
from scipy.optimize import minimize
import tqdm
from decimal import Decimal
from scipy.linalg import cholesky
from sklearn import linear_model
from scipy.stats import multivariate_normal, multivariate_t
from CovReg.cole import cov_reg_given_mean

class CovReg():

    
    def __init__(self, Y, alpha, max_iter = 100, tol = 1e-6, do_all_iter = False, method = 'secant') -> None:

        self.Y = Y
        self.max_iter = max_iter
        self.tol = tol

        self.d = Y.shape[1]
        self.n = Y.shape[0]

        self.alpha = self.n*alpha

        self.with_graph = False
        self.do_all_iter = do_all_iter
        self.method = method



    @staticmethod
    def lasso_objective(param, X, Y,Psi_inv, d,r, alpha, a = None)->tuple:
        """"
        Objective and gradient to pass into the L-BFGS-B solver to solve the lasso problem.
        """
        if a is not None:
            C = np.hstack((a, np.reshape(param[:r*d]-param[r*d:],(d,r))))
        else:
            C = np.reshape(param[:r*d]-param[r*d:],(d,r))
        
        obj =  0.5*np.trace(np.dot((Y- np.dot(X,C.T)).T,(Y- np.dot(X,C.T))).dot(Psi_inv)) + alpha*param.sum()
        t_0 = np.dot(Psi_inv, Y.T-np.dot(C,X.T))
        if a is None:
            grad = (-np.dot(t_0,X)).flatten()
        else:
            grad = (-np.dot(t_0,X)).flatten()[:d*r]
        return obj, np.concatenate((grad + alpha , - grad + alpha ), axis=None)
        

    def filter_x(self):
        pass

    def fit(self):
        """
        Wrapper
        """
        pass


    def one_iteration_F(self, Psi_pre_inv, psi, C_pre, c, Y_tilde, a = None, X1 = None, X2 = None, F_pre = None, with_mean = True):


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


    


        if self.method == 'secant_psi_identity':
            psi_in_b_est =np.identity(self.d)
        elif self.method == 'secant':
            psi_in_b_est = Psi_pre_inv.copy()


        if c is None:
            if self.alpha != 0:
                if a is None:
                    out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r1+self.r2, self.alpha, None), method='L-BFGS-B', jac=True, bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d))
                    C_vec = out.x
                    C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
                    # clf = linear_model.MultiTaskLasso(alpha=self.alpha/self.n)
                    # clf.fit(X_tilde, Y_tilde)
                    # C = clf.coef_
                else:
                    out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r2, self.alpha, a), method='L-BFGS-B', jac=True, bounds = [(0,None)]*(2*(self.r2)*self.d))
                    C_vec = out.x
                    C = np.reshape(out.x[:(self.r2)*self.d] - out.x[(self.r2)*self.d:], (self.d,self.r2))
                    C = np.hstack((a, C))
            else:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde) + 1e-6*np.identity(self.r1+self.r2)))
                if a is not None:
                    C = np.hstack((a, C[:,self.r1:]))
        else:
            C = c


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
  


        return C, Psi_est, F, tol_i

    def one_iteration(self, Psi_pre_inv, psi, C_pre, c, Y_tilde, it_nr, a = None, X1 = None, X2 = None, F_pre = None, with_mean = True):


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
        if it_nr == 0:
            v = np.ones(self.n)
            m = np.random.normal(0, 1, self.n)
        else:

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


    
        if self.alpha == 0:
            try:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde))) #np.ones((self.d, self.r1+self.r2))# 
            except:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.pinv(np.dot(X_tilde.T,X_tilde)))
        elif self.method == 'secant_psi_identity':
            psi_in_b_est = np.identity(self.d)
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r1+self.r2, self.n*self.alpha, None), 
                           method='L-BFGS-B', 
                           jac=True, 
                           bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d),
                           options = {'maxiter':4000})
            if out.status != 0:
                print(out.message)
            C_vec = out.x
            C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
        elif self.method == 'secant':
            psi_in_b_est = Psi_pre_inv.copy()
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r1+self.r2, self.n*self.alpha, None), 
                           method='L-BFGS-B', 
                           jac=True, 
                           bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d),
                           options = {'maxiter':4000})
            if out.status != 0:
                print(out.message)
            C_vec = out.x
            C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
        elif self.method == 'Lasso':
            reg_lasso = linear_model.Lasso(alpha = self.alpha, fit_intercept=False, max_iter=4000).fit(X_tilde,Y_tilde)
            C = reg_lasso.coef_.copy()
        elif self.method == 'MultiTaskLasso':
            reg_lasso = linear_model.MultiTaskLasso(alpha=self.alpha, fit_intercept=False, max_iter=4000).fit(X_tilde,Y_tilde)
            C = reg_lasso.coef_.copy()
        else:
            raise ValueError("Wrong method")


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


        tol_i = np.linalg.norm(C-C_pre)/np.linalg.norm(C_pre)
  


        return C, Psi_est, None, tol_i
    

    def get_m_v(self, C_pre, Psi_pre_inv, X1, X2, tau, error, sample = 'sample'):
       
        v = np.zeros(self.n)
        m = np.zeros(self.n)
        gamma = np.zeros(self.n)

        if X1 is not None:
            A = C_pre[:,:self.r1]
            B = C_pre[:,self.r1:]
        else:
            A = 0
            B = C_pre

        if error == 'gaussian':
            tau = np.ones(self.n)

        # for i in range(self.n):
        #     if X1 is not None:
        #         v[i] = (1+np.dot(X2[i].T, B.T).dot(tau[i]*Psi_pre_inv).dot(B).dot(X2[i])) ** (-1)
        #         m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, X1[i])).T,tau[i]*Psi_pre_inv).dot(B).dot(X2[i])
        #     elif X1 is None:
        #         v[i] = (1+np.dot(X2[i].T, B.T).dot(tau[i]*Psi_pre_inv).dot(B).dot(X2[i])) ** (-1)
        #         m[i] = v[i]*np.dot((self.Y[i]).T,tau[i]*Psi_pre_inv).dot(B).dot(X2[i])
        #     if sample == 'sample':
        #         gamma[i] = np.random.normal(loc = m[i], scale = v[i]**0.5)
        #     else:
        #         gamma[i] = m[i]


        v = np.abs(1+(tau[:,np.newaxis]*X2*np.dot(X2, np.dot(B.T, Psi_pre_inv).dot(B))).sum(1)) ** (-1)
        if X1 is not None:
            errors = self.Y - np.dot(X1,A.T)
        else:
            errors = self.Y.copy()

        m =  v*(errors*np.dot(tau[:,np.newaxis]*X2, B.T).dot(Psi_pre_inv)).sum(1)

        # if sample == 'mode':
        #     v[v<0] = 1 

        if sample == 'sample':
             gamma = np.random.normal(loc = m, scale = v**0.5)
        else:
            gamma = m


        return m, v, gamma
    
    def get_tau(self,X1,X2, gamma, C_pre, Psi_est_inv, sample = 'sample'):
        
        tau_mean = np.zeros(self.n)
        tau = np.zeros(self.n)

        if X1 is not None:
            A = C_pre[:,:self.r1]
            B = C_pre[:,self.r1:]
        else:
            A = 0
            B = C_pre

        alpha_1 = float((self.nu+self.d)/2.0)
        if X1 is not None:
            tmp = self.Y - np.dot(X1, A.T) - gamma[:, np.newaxis]*np.dot(X2, B.T)
        else:
            tmp = self.Y - gamma[:, np.newaxis]*np.dot(X2, B.T)

        beta = (self.nu + (tmp * np.dot(tmp, Psi_est_inv)).sum(1))/2.0
        if sample == 'sample':
            tau = np.random.gamma(shape = alpha_1, scale = 1/beta)
        elif sample == 'mode':
                tau = (alpha_1-1)/beta
        tau_mean = alpha_1/beta


        # for i in range(self.n):
        #     alpha_1 = float((self.nu+self.d)/2.0)
        #     if X1 is not None:
        #         tmp = self.Y[i] - np.dot(A,X1[i]) - gamma[i]*np.dot(B,X2[i])
        #     else:
        #         tmp = self.Y[i] - gamma[i]*np.dot(B,X2[i])

        #     beta = (self.nu + np.dot(tmp, Psi_est_inv).dot(tmp))/2.0
        #     if sample == 'sample':
        #         tau[i] = np.random.gamma(shape = alpha_1, scale = 1/beta)
        #     if sample == 'mode':
        #         tau[i] = (alpha_1-1)/beta
        #     tau_mean[i] = np.max((alpha_1/beta,0))

        return tau_mean, tau

    def solve_B_psi(self, X_tilde, Y_tilde, Psi_pre_inv, psi, a, C_pre, method):
        # M-step
    
        if (self.alpha == 0) or (method == 'direct'):
            try:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T,X_tilde))) #np.ones((self.d, self.r1+self.r2))# 
            except:
                C = np.dot(Y_tilde.T, X_tilde).dot(np.linalg.pinv(np.dot(X_tilde.T,X_tilde)))
            
        elif method == 'secant_psi_identity':

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

            psi_in_b_est = np.identity(self.d)
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r1+self.r2, self.n*self.alpha, None), 
                           method='L-BFGS-B', 
                           jac=True, 
                           bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d),
                           options = {'maxiter':10000})
            if out.status != 0:
                print(out.message)
            C_vec = out.x
            C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
        elif method == 'secant':

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

            psi_in_b_est = Psi_pre_inv.copy()
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, psi_in_b_est, self.d,self.r1+self.r2, self.n*self.alpha, None), 
                           method='L-BFGS-B', 
                           jac=True, 
                           bounds = [(0,None)]*(2*(self.r1+self.r2)*self.d),
                           options = {'maxiter':10000})
            if out.status != 0:
                print(out.message)
            C_vec = out.x
            C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
        elif method == 'Lasso':
            reg_lasso = linear_model.Lasso(alpha = self.alpha, fit_intercept=False, max_iter=10000).fit(X_tilde,Y_tilde)
            C = reg_lasso.coef_.copy()
        elif method == 'MultiTaskLasso':
            reg_lasso = linear_model.MultiTaskLasso(alpha=self.alpha, fit_intercept=False, max_iter=10000).fit(X_tilde,Y_tilde)
            C = reg_lasso.coef_.copy()
        else:
            raise ValueError("Wrong method")

        if psi is None:
            Psi_est = np.dot((Y_tilde-np.dot(X_tilde,C.T)).T, (Y_tilde-np.dot(X_tilde,C.T)))/self.n
            try:
                Psi_est_inv = np.linalg.inv(Psi_est)
            except:
                Psi_est_inv = np.linalg.pinv(Psi_est)
        else:
            Psi_est = psi
            Psi_est_inv = Psi_pre_inv


        return C, Psi_est, Psi_est_inv

    def init_param(self, C_init, psi, Y_tilde, X1, X2):

        if X1 is not None:
            with_mean = True
        else:
            with_mean = False

        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.identity(self.d)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)


        if isinstance(C_init, str):
            if C_init == "multi_start_hoff":
            
                best_ml = np.inf

                for o in range(200): 
                    m = np.random.normal(loc = 0, scale=1, size = self.n)
                    v = np.ones(self.n)
     
                    C_pre, _, Psi_pre_inv = self.solve_B_psi(X1, X2, m,v, with_mean, Y_tilde, Psi_pre_inv, psi, None, None, 'direct')

                    for _ in range(20):

                        m,v = self.get_m_v(with_mean, C_pre, Psi_pre_inv, X1, X2)
                        C_pre, Psi_pre, Psi_pre_inv = self.solve_B_psi(X1, X2, m,v, with_mean, Y_tilde, Psi_pre_inv, psi, None, None, 'direct')

                    
                    ml = self.marg_lik(X1, X2, Psi = Psi_pre, C = C_pre)
                    if ml < best_ml:
                        best_ml = ml
                        C_best_pre = C_pre.copy()
                        Psi_best_pre = Psi_pre.copy()
                        Psi_best_pre_inv = Psi_pre_inv.copy()

                C_pre = C_best_pre.copy()
                Psi_pre = Psi_best_pre.copy()
                Psi_pre_inv = Psi_best_pre_inv.copy()

            elif C_init == "cole":
                rnd_cole = np.random.RandomState(42)#np.random.seed(0)
                m = rnd_cole.normal(loc = 0, scale=1, size = self.n)
                v = np.ones(self.n)
                X_tilde = self.get_X_tilde(X1, X2, m, v, None, 'gaussian')
                
                C_pre, Psi_pre, Psi_pre_inv = self.solve_B_psi(X_tilde, Y_tilde, Psi_pre_inv, psi, None, None, 'direct')

            elif C_init == "random":
                m = np.random.normal(loc = 0, scale=1, size = self.n)
                v = np.ones(self.n)
                X_tilde = self.get_X_tilde(X1, X2, m, v, None, 'gaussian')
                C_pre, Psi_pre, Psi_pre_inv = self.solve_B_psi(X_tilde, Y_tilde, Psi_pre_inv, psi, None, None, 'direct')
        else:
            C_pre = C_init



        return C_pre, Psi_pre, Psi_pre_inv


    def get_X_tilde(self, X1, X2, m,v, tau, error):


        if error == 'gaussian':
            tau_mix = 1.0
        else:
            tau_mix = np.sqrt(tau[:,np.newaxis])

        s = v[:,np.newaxis] ** 0.5

        if X1 is not None:
            X_tilde =  np.vstack((np.hstack((X1*tau_mix, tau_mix*m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),tau_mix*s*X2))))
        else:
            X_tilde =  np.vstack((tau_mix*m[:,np.newaxis]*X2,tau_mix*s*X2))

        
        return X_tilde
    

    def get_X_bar(self, X1, X2, tau, gamma):

        if X1 is None:
            with_mean = False
            tau_mix = np.sqrt(tau[:,np.newaxis])
        
        gamma = gamma[:,np.newaxis]


        if with_mean:
            X_bar =  np.hstack((X1*tau_mix, tau_mix*gamma*X2))
        elif not with_mean:
            X_bar =  tau_mix*gamma*X2

        
        return X_bar
    
    def get_Y_bar(self, tau):
        return np.sqrt(tau[:, np.newaxis])*self.Y
    

    def get_Y_tilde(self, Y, tau, error):
        
        if error == 'gaussian':
            tau_mix = 1.0
        else:
            tau_mix = np.sqrt(tau[:,np.newaxis])

        return  np.vstack((Y*tau_mix, np.zeros((self.n,self.d))))





    def fit_hoff(self, X2, X1 = None, psi = None, nu = None, error = 'gaussian', sample = 'sample', X_filter = None, verbose = True, C_init = 'multi_start_hoff')-> None:
        """
        Hoff only fit covariance term
        """

        self.nu = nu

        self.r2 = X2.shape[1]
        if X1 is not None:
            self.r1 = X1.shape[1]
        else:
            self.r1 = 0

        Y_tilde = self.get_Y_tilde(self.Y, None, 'gaussian')
        C_est, Psi_est, Psi_pre_inv = self.init_param(C_init, psi, Y_tilde, X1, X2)

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)

        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)

        tau = np.ones(self.n)
        tau_mean = np.ones(self.n)

        while self.iteration <self.max_iter:
            C_pre = C_est.copy()
            m,v, gamma = self.get_m_v(C_pre, Psi_pre_inv, X1, X2, tau_mean, error, sample = 'mode')
            self.m = m
            self.v = v
            self.gamma = gamma

            if error == 't':
                tau_mean, tau = self.get_tau(X1,X2, gamma, C_est, Psi_pre_inv, 'mode')


            X_tilde = self.get_X_tilde(X1, X2, m,v, tau_mean, error)
            Y_tilde = self.get_Y_tilde(self.Y, tau_mean, error)
            C_est, Psi_est, Psi_pre_inv = self.solve_B_psi(X_tilde, Y_tilde, Psi_pre_inv, psi, None, C_est, self.method)

            # if error is t we have to do another EM step
            # if error == 't':
            #     tau_mean, tau = self.get_tau(None,X2, gamma, C_est, Psi_pre_inv, sample)
            #     X_bar =  self.get_X_bar(None, X2, tau_mean, gamma)
            #     Y_bar =  self.get_Y_bar(tau_mean)

            #     C_est, Psi_est, Psi_pre_inv = self.solve_B_psi(X_bar, Y_bar, Psi_pre_inv, psi, None, C_est, self.method)
    


            tol_i = np.linalg.norm(C_est-C_pre)/np.linalg.norm(C_pre)

            self.tol_vec[self.iteration] = tol_i

            if verbose:
                pbar.set_description(f"Error {Decimal(tol_i):.2E}")
                pbar.update()

            if (tol_i<self.tol) & (not self.do_all_iter):
                break

            if (self.iteration > 5) & (np.sum(np.abs(C_est)) < 1e-8):
                break
            
            self.iteration+=1



        self.A = C_est[:, :self.r1].copy()
        self.B = C_est[:, self.r1:].copy()
        self.Psi = Psi_est
        self.Psi_inv = Psi_pre_inv

        if verbose:
            pbar.close()

    def fit_hoff_cole(self, X2, X1 = None, psi = None, nu = None, error = 'gaussian', sample = 'sample', X_filter = None, verbose = True, C_init = 'multi_start_hoff')-> None:
        """
        Hoff only fit covariance term
        """

        self.nu = nu

        self.r2 = X2.shape[1]
        if X1 is not None:
            self.r1 = X1.shape[1]
        else:
            self.r1 = 0

        A_est = np.zeros((1, self.d))
        A_est[np.abs(A_est)>0] = 0
        spline_basis_transform = np.zeros((1,self.n))
        B_est_cole, Psi_est_cole, self.iteration = cov_reg_given_mean(A_est=A_est, basis=spline_basis_transform, x=X2.T, y=self.Y.T, iterations=self.max_iter, 
                                                                      technique = self.method, alpha= self.alpha, max_iter=4000 )
        B_est_cole = B_est_cole.T



        self.A = np.zeros(shape=(self.d,0))
        self.B = B_est_cole.copy()
        self.Psi = Psi_est_cole
        try:
            self.Psi_inv = np.linalg.inv(Psi_est_cole)
        except:
            self.Psi_inv = np.linalg.pinv(Psi_est_cole)     

    def fit_hoff_b_only_second(self, X2, psi = None, nu = None, error = 'gaussian', sample = 'sample', X_filter = None, verbose = True, C_init = 'multi_start_hoff')-> None:
        """
        Hoff only fit covariance term
        """

        self.nu = nu

        self.r2 = X2.shape[1]
        self.r1 = 0
        
        Y_tilde = self.get_Y_tilde(self.Y, None, 'gaussian')
        C_est, Psi_est, Psi_pre_inv = self.init_param(C_init, psi, Y_tilde, None, X2)

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)

        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)

        tau = np.ones(self.n)
        while self.iteration <self.max_iter:
            C_pre = C_est.copy()
            m,v, gamma = self.get_m_v(C_pre, Psi_pre_inv, None, X2, tau, error, sample = 'mode')
            self.m = m
            self.v = v
            self.gamma = gamma

            #if error == 't':
            #    tau_mean, tau = self.get_tau(None,X2, gamma, C_est, Psi_pre_inv, 'mode')


            X_tilde = self.get_X_tilde(None, X2, m,v, tau, error)
            Y_tilde = self.get_Y_tilde(self.Y, tau, error)
            C_est, Psi_est, Psi_pre_inv = self.solve_B_psi(X_tilde, Y_tilde, Psi_pre_inv, psi, None, C_est, self.method)

            #if error is t we have to do another EM step
            if error == 't':
                tau_mean, tau = self.get_tau(None, X2, gamma, C_est, Psi_pre_inv, sample)
                X_bar =  self.get_X_bar(None, X2, tau_mean, gamma)
                Y_bar =  self.get_Y_bar(tau_mean)

                C_est, Psi_est, Psi_pre_inv = self.solve_B_psi(X_bar, Y_bar, Psi_pre_inv, psi, None, C_est, self.method)
    


            tol_i = np.linalg.norm(C_est-C_pre)/np.linalg.norm(C_pre)

            self.tol_vec[self.iteration] = tol_i

            if verbose:
                pbar.set_description(f"Error {Decimal(tol_i):.2E}")
                pbar.update()

            if (tol_i<self.tol) & (not self.do_all_iter):
                break

            if (self.iteration > 5) & (np.sum(np.abs(C_est)) < 1e-8):
                break
            
            self.iteration+=1



        self.A = None
        self.B = C_est
        self.Psi = Psi_est
        if verbose:
            pbar.close()

    # def fit_hoff(self, X1, X2, psi = None, X_filter = None, verbose = True, C_init = 'multi_start_hoff')-> None:
    #     """
    #     Hoff with lasso
    #     """

    #     c = None
    #     self.r1 = X1.shape[1]
    #     self.r2 = X2.shape[1]

    #     Y_tilde = self.get_Y_tilde(self.Y, None, 'gaussian')
    #     C_est, Psi_est, Psi_pre_inv = self.init_param(C_init, psi, Y_tilde, None, X2)

    #     if verbose:
    #         pbar = tqdm.tqdm(total = self.max_iter)

    #     self.iteration = 0
    #     self.tol_vec = np.zeros(self.max_iter)

    #     tau = np.ones(self.n)
    #     tau_mean = np.ones(self.n)

    #     while self.iteration <self.max_iter:
            
    #         m,v = self.get_m_v(True, C_pre, Psi_pre_inv, X1, X2)
    #         C_est, Psi_est, Psi_pre_inv = self.solve_B_psi(X1, X2, m,v, True, Y_tilde, Psi_pre_inv, psi, None, C_pre, self.method)
            
    #         tol_i = np.linalg.norm(C_est-C_pre)/np.linalg.norm(C_pre)

    #         self.tol_vec[self.iteration] = tol_i

    #         if verbose:
    #             pbar.set_description(f"Error {Decimal(tol_i):.2E}")
    #             pbar.update()

    #         if (tol_i<self.tol) & (not self.do_all_iter):
    #             break

    #         if (self.iteration > 5) & (np.sum(np.abs(C_est)) < 1e-8):
    #             break
            
    #         self.iteration+=1


    #         C_pre = C_est.copy()


    #     self.A = C_est[:, :self.r1].copy()
    #     self.B = C_est[:, self.r1:].copy()
    #     self.Psi = Psi_est

    #     if verbose:
    #         pbar.close()

    def marg_lik(self, X1 = None, X2 = None, Psi = None, C = None, error = 'gaussian'):

        v = np.ones(self.n)
        m = np.ones(self.n)

        if Psi is None:
            try:
                psi_inv = np.linalg.inv(self.Psi)
            except:
                psi_inv = np.linalg.pinv(self.Psi)
            Psi = self.Psi
        else:
            try:
                psi_inv = np.linalg.inv(Psi)
            except:
                psi_inv = np.linalg.pinv(Psi)


        if C is None:
            if self.A is None:
                C = self.B
            else:
                C = np.hstack((self.A, self.B))
                


        tau_mean = np.ones(self.n)
        m,v, gamma = self.get_m_v(C, psi_inv, X1, X2, tau_mean, error = error, sample = 'mode')


        if error == 't':
            tau_mean, _ = self.get_tau(X1,X2, gamma, C, psi_inv, 'mode')


        X_tilde = self.get_X_tilde(X1, X2, m,v, tau_mean, error)
        Y_tilde = self.get_Y_tilde(self.Y, tau_mean, error)
        
        # M-step

        # L = cholesky(self.Psi)
        try:
            v, _ = np.linalg.eigh(Psi)
            v = v[v>1e-6]
            obj = 0.5*np.trace(np.dot((Y_tilde-np.dot(X_tilde, C.T)).T, (Y_tilde-np.dot(X_tilde, C.T))).dot(psi_inv)) + 0.5*self.n*np.sum(np.log(v)) + self.n*self.d*np.log(2*np.pi)
        
            #obj = np.sum(multivariate_normal.logpdf(Y_tilde, mean = np.dot(X_tilde, C.T), cov = Psi, allow_singular=True))
        except:
            obj = np.nan

        
        return obj


    def l2(self, X1 = None, X2 = None):

        v = np.ones(self.n)
        m = np.ones(self.n)

        try:
            psi_inv = np.linalg.inv(self.Psi)
        except:
            psi_inv = np.linalg.pinv(self.Psi)

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
        obj = 0.5*np.trace(np.dot((Y_tilde-np.dot(X_tilde, C.T)).T, (Y_tilde-np.dot(X_tilde, C.T))))
        
        return obj
    
    def nr_params(self):
        if self.A is None:
            C = self.B
        else:
            C = np.hstack((self.A, self.B))
        return np.sum(np.abs(C)>1e-3)


    
    def likelihood(self, X1 = None, X2 = None, error = 'gaussian', nu = None):

        lik = 0.0

        if error == 'gaussian':

            for i in range(self.n):
                cov = np.dot(self.B, np.outer(X2[i], X2[i])).dot(self.B.T) + self.Psi
                if X1 is None:
                    mean = np.zeros(self.d)
                else:
                    mean = np.dot(self.A, X1[i])
                    
                lik += multivariate_normal(mean = mean, cov = cov, allow_singular=True).logpdf(self.Y[i])
        elif error == 't':

            for i in range(self.n):
                cov = np.dot(self.B, np.outer(X2[i], X2[i])).dot(self.B.T) + self.Psi*(nu-1.0)/nu
                if X1 is None:
                    mean = np.zeros(self.d)
                else:
                    mean = np.dot(self.A, X1[i])
                    
                lik += multivariate_t(loc= mean, shape= cov, df = nu, allow_singular=True).logpdf(self.Y[i])


        return lik



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

