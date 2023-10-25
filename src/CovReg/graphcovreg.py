




import numpy as np


import tqdm
from scipy.optimize import minimize
from decimal import Decimal


from CovReg.covreg import CovReg



class GraphCovReg(CovReg):

    
    def __init__(self, Y, alpha, K, L, omega, beta, r, max_iter = 100, tol = 1e-3) -> None:

        CovReg.__init__(self, Y, alpha, max_iter, tol) 

        self.with_graph = True
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.L = L
        self.omega = omega
        self.beta = beta
        self.r = r


        v, u = np.linalg.eigh(L)

        self.H = np.array(np.dot(u, np.diag(np.exp(-beta*v))).dot(u.T))
        self.H_sq_inv = np.linalg.inv(np.dot(self.H,self.H) + 0.001*np.identity(self.r))
  

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

        
            out = minimize(self.secant_objective, np.ones(self.n*self.r), args = (self.r, m, v, self.Y, psi_inv, B,A, self.K_inv, self.omega, self.H_sq_inv), jac=True, method= 'L-BFGS-B')


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


    def F_direct_cov_only(self,  B, m,v, psi_inv ):
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
    

    def fit_ggp(self, psi = None, F_start = None, reg = None, type_reg = None, F_method = 'direct', c = None, verbose = True, a = None)-> None:
        """
        Graph GP only fit covariance term
        """


        self.reg = reg
        self.type_reg = type_reg

        self.F_method = F_method


        self.r1 = self.r
        self.r2 = self.r


        if F_start is None:
            F_pre = np.ones((self.n, self.r))
        else:
            F_pre = F_start


        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))


        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        if c is None:
            C_pre = np.ones((self.d, self.r+self.r))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        else:
            C_pre = c

        
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)

        if verbose:
            pbar = tqdm.tqdm(total = self.max_iter)
        while self.iteration <self.max_iter:
            C, Psi_est, F, tol_i = self.one_iteration(np.linalg.inv(Psi_pre_inv), psi, C_pre, c, Y_tilde, F_pre = F_pre, a = a, with_mean = True)    
            self.tol_vec[self.iteration] = tol_i

            if verbose:
                pbar.set_description(f"Error {Decimal(tol_i):.2E}")
                pbar.update()

            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()

            C_pre = C.copy()
            F_pre = F.copy()

        self.A = C[:,:self.r]
        self.B = C[:,self.r:]
        self.F = F
        self.Psi = Psi_est

        if verbose:
            pbar.close()



    def marg_lik(self):

        v = np.ones(self.n)
        m = np.ones(self.n)

        psi_inv = np.linalg.inv(self.Psi)

        if self.B is None:

            obj = 0.5*np.trace(np.dot((self.Y-np.dot(self.F, self.A.T)).T, (self.Y-np.dot(self.F, self.A.T))).dot(psi_inv)) + 0.5*self.omega*np.trace(np.dot(self.K_inv, self.F).dot(self.H_sq_inv).dot(self.F.T))
        

        else:

            if self.A is None:
                C = self.B
            else:
                C = np.hstack((self.A, self.B))
        

            # E-step
            for i in range(self.n):
                
                if self.A is not None:
                    x_i = np.hstack((self.F[i], self.F[i]))
                    v[i] = (1+np.dot(x_i.T, C.T).dot(psi_inv).dot(C).dot(x_i)) ** (-1)
                    m[i] = v[i]*np.dot((self.Y[i]-np.dot(self.A, self.F[i])).T,psi_inv).dot(C).dot(x_i)
                else:
                    x_i = self.F[i]
                    v[i] = (1+np.dot(x_i.T, C.T).dot(psi_inv).dot(C).dot(x_i)) ** (-1)
                    m[i] = v[i]*np.dot((self.Y[i]).T,psi_inv).dot(C).dot(x_i)

        
        # M-step

            Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))
            if self.A is not None:
                X_tilde =  np.vstack((np.hstack((self.F, m[:,np.newaxis]*self.F)),np.hstack((np.zeros((self.n, self.r1)),v[:,np.newaxis]*self.F))))
            else:
                X_tilde =  np.vstack((m[:,np.newaxis]*self.F,v[:,np.newaxis]*self.F))


            obj = 0.5*np.trace(np.dot((Y_tilde-np.dot(X_tilde, C.T)).T, (Y_tilde-np.dot(X_tilde, C.T))).dot(psi_inv)) + 0.5*self.omega*np.trace(np.dot(self.K_inv, self.F).dot(self.H_sq_inv).dot(self.F.T))
        
        return obj




    def fit_ggp_b_only(self, psi = None, F_start = None,  reg = None, type_reg = None, F_method = 'direct', c = None)-> None:
        """
        Graph GP only fit covariance term
        """

        self.A = None
        self.F_method = F_method
        self.reg = reg
        self.type_reg = type_reg

        self.r1 = 0
        self.r2 = self.r

        if F_start is None:
            F_pre = np.ones((self.n, self.r))
        else:
            F_pre = F_start

        Y_tilde = np.vstack((self.Y, np.zeros((self.n,self.d))))


        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)

        v = np.ones(self.n)
        m = np.ones(self.n)
        # The solution is evry sensitive to the inital matrix, the l2 regularization for B_pre effects alpha
        
        if c is None:
            C_pre = np.ones((self.d, self.r))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        else:
            C_pre = c
        
        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:
            C, Psi_est, F, tol_i = self.one_iteration(np.linalg.inv(Psi_pre), psi, C_pre, c, Y_tilde, F_pre = F_pre, with_mean = False)
            self.tol_vec[self.iteration] = tol_i
            
            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1

            if psi is None:
                Psi_pre = Psi_est.copy()
            else:
                Psi_pre = psi

            C_pre = C.copy()
            F_pre = F.copy()



        self.A = None
        self.B = C
        self.F = F
        self.Psi = Psi_est



    def fit_ggp_a_only(self, psi = None, F_start = None, reg = None, type_reg = None, F_method = 'direct', c = None) -> None:

        self.B = None
        self.reg = reg
        self.type_reg = type_reg
        self.F_method = F_method

        self.r1 = self.r
        self.r2 = 0


        if psi is None:
            Psi_pre = np.identity(self.d)
            Psi_pre_inv = np.linalg.inv(Psi_pre)
        else:
            Psi_pre = psi
            Psi_pre_inv = np.linalg.inv(Psi_pre)



        if c is None:
            C_pre = np.ones((self.d, self.r))# np.dot(Y_tilde.T, X_tilde).dot(np.linalg.inv(np.dot(X_tilde.T, X_tilde) + alpha*np.identity(r)))
        else:
            C_pre = c


        self.iteration = 0
        self.tol_vec = np.zeros(self.max_iter)
        while self.iteration <self.max_iter:
                    # M-step
            Psi_pre_inv = np.linalg.inv(Psi_pre)
            if self.F_method == 'direct':
                F = self.F_ls_reg( C_pre, Psi_pre_inv)
            elif self.F_method == 'secant':
                F = self.F_secant_optim(np.zeros((self.d, self.r)), C_pre, np.ones(self.n),np.ones(self.n), Psi_pre_inv)

            if c is None:
                C_tmp = C_pre.flatten()
                C_vec = np.zeros(2*(self.r1+self.r2)*self.d)
                C_vec[:(self.r1+self.r2)*self.d] = np.abs(C_tmp) * (C_tmp>0)
                C_vec[(self.r1+self.r2)*self.d:] = np.abs(C_tmp) * (C_tmp<0)

                out = minimize(self.lasso_objective, C_vec, args = (F, self.Y, Psi_pre_inv,self.d,self.r1+self.r2, self.alpha), method='L-BFGS-B', jac=True, bounds = [(0.0,None)]*(2*(self.r1+self.r2)*self.d))
                C_vec = out.x
                C = np.reshape(out.x[:(self.r1+self.r2)*self.d] - out.x[(self.r1+self.r2)*self.d:], (self.d,self.r1+self.r2))

            else:
                C = c

            if psi is None:
                Psi_est = np.cov((self.Y-np.dot(F,C.T)).T)*(self.n-1)/self.n
            else:
                Psi_est = psi


            tol_i = (np.linalg.norm(C-C_pre))/(np.linalg.norm(C_pre))
            self.tol_vec[self.iteration] = tol_i
        
            if (tol_i<self.tol) & (not self.do_all_iter):
                break
            
            self.iteration+=1
            
            if psi is None:
                Psi_est = np.cov((self.Y-np.dot(F,C.T)).T)*(self.n-1)/self.n
                Psi_pre = Psi_est.copy()
            else:
                Psi_est = psi


            C_pre = C.copy()
            self.iteration+=1


        self.A = C
        self.B = None
        self.F = F
        self.Psi = Psi_est






