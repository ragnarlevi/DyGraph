
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import networkx as nx
import yfinance as yf
import sys
sys.path.insert(0, 'C:/Users/User/Code/DyGraph')
sys.path.insert(0, 'C:/Users/User/Code/DyGraph/src')

import DyGraph as dg
import port_measures as pm
import matplotlib.pyplot as plt
import tqdm
import scipy
from scipy.optimize import minimize
from sklearn import linear_model




class CovReg():

    
    def __init__(self, Y, alpha, max_iter = 100, tol = 1e-6) -> None:

        self.Y = Y
        self.max_iter = max_iter
        self.tol = tol

        self.d = Y.shape[1]
        self.n = Y.shape[0]

        self.alpha = self.n*alpha



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
    
    @staticmethod
    def secant_objective(param, r, m, v, Y, psi_inv, B,A, K_inv, omega, H_sq_inv):

        
        n = Y.shape[0]
        d = Y.shape[1]

        M = np.diag(m)
        S = np.diag(v)


        Y_tilde = np.vstack((Y, np.zeros((n,d))))

        F = np.reshape(param,(n,r), order='F')
    
        F_tilde =  np.vstack((np.hstack((F, m[:,np.newaxis]*F)),np.hstack((np.zeros((n, r)),v[:,np.newaxis]*F))))
        C = np.hstack((A,B))

        obj = 0.5*np.trace(np.dot((Y_tilde-np.dot(F_tilde, C.T)).T, (Y_tilde-np.dot(F_tilde, C.T))).dot(psi_inv))  + 0.5*omega*np.trace(np.dot(K_inv, F).dot(H_sq_inv).dot(F.T))
        
        grad1 = -np.dot(M,Y).dot(psi_inv).dot(B) -np.dot(Y,psi_inv).dot(A) 
        grad2 = np.dot(F,A.T).dot(psi_inv).dot(A)
        grad3 = np.dot(M, F).dot(A.T).dot(psi_inv).dot(B) + np.dot(M, F).dot(B.T).dot(psi_inv).dot(A)
        grad4 = np.dot(M**2+S**2,F).dot(B.T).dot(psi_inv).dot(B)

        grad = (grad1 +grad2+grad3+grad4+ omega*np.dot(K_inv, F).dot(H_sq_inv)).flatten(order='F')

        return obj, grad
        

    def F_secant_optim(self,B, A, m,v, psi_inv):


    
        H_sq = np.dot(self.H,self.H)
        H_sq_inv = np.linalg.inv(H_sq + 0.001*np.identity(self.r2))


    
        out = minimize(self.secant_objective, np.ones(self.n*self.r), args = (self.r, m, v, self.Y, psi_inv, B,A, self.K_inv, self.omega, H_sq_inv), jac=True, method= 'L-BFGS-B')


        return np.reshape(out.x,(self.n,self.r), order='F')


    def F_ls(self, A, psi_inv):

        A_quad = np.dot(A.T, psi_inv).dot(A)
        F_ls_no_graph = np.dot(self.Y, psi_inv).dot(A).dot(np.linalg.pinv(A_quad)+0.001*np.identity(A.shape[1])) 


        return F_ls_no_graph


    

    def F_ls_reg(self, A, psi_inv):

        I_n = np.identity(self.n)


        A_quad = np.dot(A.T, psi_inv).dot(A)

        mat = np.kron(A_quad, I_n) + np.kron(self.H_sq_inv,self.K_inv)
        l, u = np.linalg.eigh(mat)
        if self.type_reg == 'Tikanov':
            inv_mat = np.dot(u, np.diag(1/(l+self.reg))).dot(u.T)
        elif self.type_reg == 'spectral':
            l_inv = 1/l
            l_inv[l<self.reg] = 0
            inv_mat = np.dot(u, np.diag(l_inv)).dot(u.T)

        F = np.dot(inv_mat, np.dot(self.Y, psi_inv).dot(A).flatten(order='F'))
        
        return np.reshape(F,(self.n,self.r), order='F')



    def F_cov_no_graph(self,  B, m,v,psi_inv):

        M = np.diag(m)
        S = np.diag(v)
        inv_em = np.linalg.inv(M**2+S**2)
        B_quad = np.dot(B.T, psi_inv).dot(B)

        F = np.dot(inv_em, M).dot(self.Y).dot(psi_inv).dot(B).dot(np.linalg.pinv(B_quad))
        return F


    def F_cov_reg(self,  B, m,v, psi_inv ):
        M = np.diag(m)
        S = np.diag(v)

        B_quad = np.dot(B.T, psi_inv).dot(B)

        # mat = np.kron(H_sq_inv+0.1*np.identity(r), K_inv+0.1*np.identity(n)) + np.kron(A_quad+0.1*np.identity(r), np.identity(n))
        mat = np.kron(self.H_sq_inv, self.K_inv) + np.kron(B_quad, M**2+S**2)
        l, u = np.linalg.eigh(mat)
        if self.type_reg == 'Tikanov':
            inv_mat = np.dot(u, np.diag(1/(l+self.reg))).dot(u.T)
        elif self.type_reg == 'spectral':
            l_inv = 1/l
            l_inv[l<self.reg] = 0
            inv_mat = np.dot(u, np.diag(l_inv)).dot(u.T)
        F_ls = np.dot( inv_mat, np.dot(M, self.Y).dot(psi_inv).dot(B).flatten(order='F'))
        F_ls = np.reshape(F_ls,(self.n,self.r), order='F')

        return F_ls


    def F_direct(self,B, A, m,v, psi_inv):

        I_n = np.identity(self.n)
        M = np.diag(m)
        S = np.diag(v)


        B_quad = np.dot(B.T, psi_inv).dot(B)
        A_quad = np.dot(A.T, psi_inv).dot(A)
        AB_quad = np.dot(A.T, psi_inv).dot(B)

        mat = np.kron(A_quad, I_n) + np.kron(AB_quad+AB_quad.T, M) + np.kron(B_quad, M**2+S**2) + np.kron(self.H_sq_inv,self.K_inv)
        l, u = np.linalg.eigh(mat)
        if self.type_reg == 'Tikanov':
            inv_mat = np.dot(u, np.diag(1/(l+self.reg))).dot(u.T)
        elif self.type_reg == 'spectral':
            l_inv = 1/l
            l_inv[l<self.reg] = 0
            inv_mat = np.dot(u, np.diag(l_inv)).dot(u.T)

        F = np.dot(inv_mat, (np.dot(self.Y, psi_inv).dot(A) + np.dot(M, self.Y).dot(psi_inv).dot(B)).flatten(order='F'))
        return np.reshape(F,(self.n,self.r), order='F')




    def filter_x(self):
        pass

    def fit(self):
        """
        Wrapper
        """
        pass


    def fit_hoff_b_only(self, X2, psi = None, X_filter = None)-> None:
        """
        Hoff only fit covariance term
        """



        self.r2 = X2.shape[1]
        
        
        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))

        if psi is None:
            Psi_pre = np.identity(self.d)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r2))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))


        C_tmp = C_pre.flatten()
        C_vec = np.zeros(2*(self.r2)*self.d)
        C_vec[:(self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
        C_vec[(self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)
        
        self.iteration = 0
        while self.iteration <self.max_iter:
            if psi is None:
                Psi_pre_inv = np.linalg.inv(Psi_pre)

            # E-step
            for i in range(self.n):
                x_i = X2[i]
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-0).T,Psi_pre_inv).dot(C_pre).dot(x_i)

            X_tilde =  np.vstack((m[:,np.newaxis]*X2,v[:,np.newaxis]*X2))
            # M-step
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, Psi_pre_inv,self.d,self.r2, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r2)*self.d))
            C_vec = out.x
            C = np.reshape(out.x[:(self.r2)*self.d] - out.x[(self.r2)*self.d:], (self.d,self.r2))
            #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
            #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)

            if psi is None:
                Psi_est = np.cov((Y_tilde-np.dot(X_tilde,C.T)).T)*(self.n-1)/self.n
            else:
                Psi_est = psi
    
            if np.linalg.norm(C-C_pre)<self.tol:
                break

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()
            self.iteration+=1


        self.A = 0
        self.B = C
        self.Psi = Psi_est

    def fit_hoff(self, X1, X2, psi = None, X_filter = None)-> None:
        """
        Hoff with lasso
        """
        self.r1 = X1.shape[1]
        self.r2 = X2.shape[1]


        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))



        if psi is None:
            Psi_pre = np.identity(self.d)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        X_tilde =  np.vstack((np.hstack((X1, m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*X2))))
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r1+self.r2))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))



        C_tmp = C_pre.flatten()
        C_vec = np.zeros(2*(self.r1+self.r2)*self.d)
        C_vec[:(self.r1+self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
        C_vec[(self.r1+self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)
        
        self.iteration = 0
        while self.iteration <self.max_iter:
            if psi is None:
                Psi_pre_inv = np.linalg.inv(Psi_pre)

            # E-step
            A = C_pre[:,:self.r1]
            for i in range(self.n):
                x_i = np.hstack((X1[i], X2[i]))
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, X1[i])).T,Psi_pre_inv).dot(C_pre).dot(x_i)

            X_tilde =  np.vstack((np.hstack((X1, m[:,np.newaxis]*X2)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*X2))))
            # M-step
            out = minimize(self.lasso_objective, C_vec, args = (X_tilde, Y_tilde, Psi_pre_inv,self.d,self.r1+self.r2, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r1+self.r2)*self.d))
            C_vec = out.x
            C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))
            #inv_m = np.linalg.inv(np.dot(X_tilde.T, X_tilde))
            #B = np.dot(Y_tilde.T,X_tilde).dot(inv_m)

            if psi is None:
                Psi_est = np.cov((Y_tilde-np.dot(X_tilde,C.T)).T)*(self.n-1)/self.n
            else:
                Psi_est = psi
    
            if np.linalg.norm(C-C_pre)<self.tol:
                break

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()
        
            #print(scipy.linalg.norm(B_pre-B_true))
            self.iteration+=1


        self.A = C[:,:self.r1]
        self.B = C[:,self.r1:]
        self.Psi = Psi_est


    def fit_ggp_a_only(self, psi = None, X_filter = None)-> None:
        """
        Graph GP only fit mean term
        """
        pass
    def fit_ggp_b_only(self, r, psi, F_start = None, X_filter = None, reg = None, reg_type = None)-> None:
        """
        Graph GP only fit covariance term
        """

         graph dæmi
        
        
        self.r  = r

        if F_start is None:
            F_pre = np.zeros((self.n, self.r))

        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))



        if psi is None:
            Psi_pre = np.identity(self.d)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))

        C_tmp = C_pre.flatten()
        C_vec = np.zeros(2*(self.r)*self.d)
        C_vec[:(self.r)*self.d] = np.abs(C_tmp) * (C_tmp>0)
        C_vec[(self.r)*self.d:] = np.abs(C_tmp) * (C_tmp<0)
        
        self.iteration = 0
        while self.iteration <self.max_iter:
            if psi is None:
                Psi_pre_inv = np.linalg.inv(Psi_pre)

            # E-step
            B = C_pre[:,self.r1:]
            for i in range(self.n):
                x_i = F_pre[i], F_pre[i]
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-0).T,Psi_pre_inv).dot(C_pre).dot(x_i)

            F_tilde =  np.vstack((m[:,np.newaxis]*F_pre, v[:,np.newaxis]*F_pre))


            # M-step
            F = self.F_cov_reg(self,  B, m,v, Psi_pre_inv )
            out = minimize(self.lasso_objective, C_vec, args = (F_tilde, Y_tilde, Psi_pre_inv,self.d,self.r, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r1+self.r2)*self.d))
            C_vec = out.x
            C = np.reshape(out.x[:(self.r)*self.d] - out.x[(self.r)*self.d:], (self.d,2*self.r))


            if psi is None:
                Psi_est = np.cov((Y_tilde-np.dot(F_tilde,C.T)).T)*(self.n-1)/self.n
            else:
                Psi_est = psi
    
            if np.linalg.norm(C-C_pre)<self.tol:
                break

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()
            F_pre = F.copy()
        
            #print(scipy.linalg.norm(B_pre-B_true))
            self.iteration+=1


        self.A = 0
        self.B = C[:,self.r:]
        self.F = F
        self.Psi = Psi_est

    def fit_ggp(self, r, psi, F_start = None, X_filter = None, reg = None, reg_type = None)-> None:
        """
        Graph GP only fit covariance term
        """

        graph dæmi

        self.r  = r

        if F_start is None:
            F_pre = np.zeros((self.n, self.r))


        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))



        if psi is None:
            Psi_pre = np.identity(self.d)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        C_pre = np.ones((self.d, self.r+self.r))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))

        C_tmp = C_pre.flatten()
        C_vec = np.zeros(2*(self.r+self.r)*self.d)
        C_vec[:(self.r+self.r)*self.d] = np.abs(C_tmp) * (C_tmp>0)
        C_vec[(self.r+self.r)*self.d:] = np.abs(C_tmp) * (C_tmp<0)
        
        self.iteration = 0
        while self.iteration <self.max_iter:
            if psi is None:
                Psi_pre_inv = np.linalg.inv(Psi_pre)

            # E-step
            A = C_pre[:,:self.r]
            B = C_pre[:,self.r:]
            for i in range(self.n):
                x_i = np.hstack((F_pre[i], F_pre[i]))
                v[i] = (1+np.dot(x_i.T, C_pre.T).dot(Psi_pre_inv).dot(C_pre).dot(x_i)) ** (-1)
                m[i] = v[i]*np.dot((self.Y[i]-np.dot(A, F_pre[i])).T,Psi_pre_inv).dot(C_pre).dot(x_i)

            F_tilde =  np.vstack((np.hstack((F_pre, m[:,np.newaxis]*F_pre)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*F_pre))))


            # M-step
            F = self.F_direct( B, A, m,v, Psi_pre_inv)
            out = minimize(self.lasso_objective, C_vec, args = (F_tilde, Y_tilde, Psi_pre_inv,self.d,self.r, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r1+self.r2)*self.d))
            C_vec = out.x
            C = np.reshape(out.x[:(2*self.r)*self.d] - out.x[(2*self.r)*self.d:], (self.d,2*self.r))


            if psi is None:
                Psi_est = np.cov((Y_tilde-np.dot(F_tilde,C.T)).T)*(self.n-1)/self.n
            else:
                Psi_est = psi
    
            if np.linalg.norm(C-C_pre)<self.tol:
                break

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()
            F_pre = F.copy()
        
            #print(scipy.linalg.norm(B_pre-B_true))
            self.iteration+=1


        self.A = C[:,:self.r]
        self.B = C[:,self.r:]
        self.F = F
        self.Psi = Psi_est




    







