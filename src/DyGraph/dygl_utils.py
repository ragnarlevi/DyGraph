
# Function used 

import numpy as np
from scipy.stats import chi2
import scipy.integrate as integrate
from collections import defaultdict
from scipy import optimize



def _T(u,nu):
    """
    Function to calcualte the generalized skew-t weights

    Parameters
    --------------------
    u: float 
        An observation from the uniform distribution
    nu: list
        A list containing the degree of freedom for each weight 

    Returns
    ---------------------
     :  np.array
        Array with possible weights/hidden RV.
    """
    return np.array([chi2.ppf(u, df = nu[i])/nu[i] for i in range(len(nu))])


def gen_skew_t_integrand(u, theta,x,nu,gamma):
    """
    integrand function for generalized skew-t
    """
    T_vec = _T(u,nu)

    mean = np.dot(np.diag(np.reciprocal(T_vec)), gamma)
    a = np.sqrt(T_vec)
    prec = np.multiply(theta, np.outer(a,a))
    return np.sqrt(np.prod(T_vec))*np.exp(-0.5*np.dot(x-mean, prec).dot(x-mean))


gen_skew_t_integrand = np.vectorize(gen_skew_t_integrand,excluded = [1,2,3,4])



def generalized_skew_t( x, theta, nu, gamma = None, n = 10):
    """
    pdf of the generalized skew-T distribution

    Paremters
    -----------------
    x: np.array
        Vector of features /(1 observation)
    theta: np.array
        Precision of the normal distribution
    nu: list
        Degree of freedom for each feature in x
    gamma: np.vector or None
        The addition term of the generalized skew-T distribution. If None a vector of zeros will be used
        which corresponds to the group-T distribution

    n: int
        Number of Gaussian-Qudrature terms for the integration

    """
    d = theta.shape[0]


    if gamma is None:
        gamma = np.zeros(theta.shape[0])

    C = 1/np.sqrt((1/np.linalg.det(theta))*(2*np.pi) ** d)
    #s, logd = np.linalg.slogdet(theta)
    # C = np.exp((d/2)*(logd -  np.log(2*np.pi)))
    result = integrate.fixed_quad(gen_skew_t_integrand, 0, 1,args = (theta, x, nu, gamma),n = n)
    return C*result[0]




def D1_pair_integrand(u,theta,gamma,x,nu,i,j):
    """
    Integrand for element (i,j) in the matrix D_1 as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    theta: np.array
        precision matrix
    gamma: np.array
        vector of current gamma estimate
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    i: int 
        Row of D_1 to be calcualted
    j: int
        Column of D_1 to be calulcate

    Returns
    ---------------------
     :  float
        Integrand value
    """



    T_vec = _T(u,nu)
    # a = np.sqrt(np.reciprocal(T_vec))
    # prec = np.multiply(theta, np.outer(a,a))
    # mean = gamma*np.reciprocal(T_vec)

    # A = np.sqrt(T_vec)
    # return A[i]*A[j]*multivariate_normal.pdf(x, mean =mean, cov = cov, allow_singular = True)

    a = np.sqrt(T_vec)
    prec = np.multiply(theta, np.outer(a,a))
    A = np.sqrt(T_vec)
    mean = gamma*np.reciprocal(T_vec)
    return np.sqrt(np.prod(T_vec))*A[i]*A[j]*np.exp(-0.5*np.dot(x-mean, prec).dot(x-mean))


D1_pair_integrand = np.vectorize(D1_pair_integrand,excluded = [1,2,3,4,5,6])

def D1_EM(theta, x, nu,m, gamma, n = 5):
    """
    D1 EM matrix as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    theta: np.array
        Curren estimate of Precision matrix
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    m: list
        list containing the group membership of each feautre in x
    gamma: np.array
        Current estimate of gamma vector
    n: int
        Number of Gaussian-Qudrature terms for the integration

    Returns
    ---------------------
     : np.array
        D1 EM matrix
    """
    d = theta.shape[0]
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = theta.shape)
    denom = generalized_skew_t( x, theta, nu, gamma, n)
    
    if denom > 0:

        C = 1/np.sqrt((1/np.linalg.det(theta))*(2*np.pi) ** d)

        for i in range(D.shape[0]):
            for j in range(i, D.shape[0]):
                combintaion = ''.join(sorted(str(m[i])+str(m[j])))
                if combination_calculated[combintaion] is None:
                    D[i,j] =  C*integrate.fixed_quad(D1_pair_integrand, 0, 1 ,args = (theta,gamma, x,nu,i,j),n = n)[0]/denom
                    combination_calculated[combintaion] = D[i,j]
                else:
                    D[i,j] = combination_calculated[combintaion]


    return np.triu(D,0) + np.triu(D,1).T




def D2_pair_integrand(u,theta,gamma,x,nu,i,j):
    """
    Integrand for element (i,j) in the matrix D_2 as defined in paper. Used for generalized skew-T estimation.

    Parameters
    ----------------------
    theta: np.array
        precision matrix
    gamma: np.array
        vector of current gamma estimate
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    i: int 
        Row of D_1 to be calcualted
    j: int
        Column of D_1 to be calulcate

    Returns
    ---------------------
     :  float
        Integrand value
    """

    T_vec = _T(u,nu)
    #a = np.sqrt(np.reciprocal(T_vec))
    #cov = np.multiply(S, np.outer(a,a))
    #mean = gamma*np.reciprocal(T_vec)

    a = np.sqrt(T_vec)
    prec = np.multiply(theta, np.outer(a,a))
    mean = gamma*np.reciprocal(T_vec)
    return np.sqrt(np.prod(T_vec))*np.sqrt(T_vec[j])*(1/np.sqrt(T_vec[i]))*np.exp(-0.5*np.dot(x-mean, prec).dot(x-mean))

    #return np.sqrt(T_vec[j])*(1/np.sqrt(T_vec[i]))*multivariate_normal.pdf(x, mean =mean, cov = cov, allow_singular = True)

D2_pair_integrand = np.vectorize(D2_pair_integrand,excluded = [1,2,3,4,5,6])


def D2_EM(theta, x, nu,m, gamma, n = 5):
    """
    D2 EM matrix as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    theta: np.array
        Curren estimate of Precision matrix
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    m: list
        list containing the group membership of each feautre in x
    gamma: np.array
        Current estimate of gamma vector
    n: int
        Number of Gaussian-Qudrature terms for the integration

    Returns
    ---------------------
     : np.array
        D2 EM matrix
    """


    d = theta.shape[0]
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = theta.shape)
    denom = generalized_skew_t( x, theta, nu, gamma, n)

    if denom > 0:
        C = 1/np.sqrt((1/np.linalg.det(theta))*(2*np.pi) ** d)

        for i in range(D.shape[0]):
            for j in range(D.shape[0]):
                combintaion = ''.join(sorted(str(m[i])+str(m[j])))
                if combination_calculated[combintaion] is None:
                    D[i,j] =  C*integrate.fixed_quad(D2_pair_integrand, 0, 1 ,args = (theta, gamma, x,nu,i,j),n = n)[0]/denom
                    combination_calculated[combintaion] = D[i,j]
                else:
                    D[i,j] = combination_calculated[combintaion]

    return D




def D3_pair_integrand(u,theta,gamma,x,nu,i,j):
    """
    Integrand for element (i,j) in the matrix D_2 as defined in paper. Used for generalized skew-T estimation.

    Parameters
    ----------------------
    S: np.array
        Covariance matrix
    gamma: np.array
        vector of current gamma estimate
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    i: int 
        Row of D_1 to be calcualted
    j: int
        Column of D_1 to be calulcate

    Returns
    ---------------------
     :  float
        Integrand value
    """


    T_vec = _T(u,nu)
    #a = np.sqrt(np.reciprocal(T_vec))
    #cov = np.multiply(S, np.outer(a,a))
    #mean = gamma*np.reciprocal(T_vec)

    a = np.sqrt(T_vec)
    prec = np.multiply(theta, np.outer(a,a))
    mean = gamma*np.reciprocal(T_vec)
    return np.sqrt(np.prod(T_vec))*(1/np.sqrt(T_vec[i]))*(1/np.sqrt(T_vec[j]))*np.exp(-0.5*np.dot(x-mean, prec).dot(x-mean))


    # return (1/np.sqrt(T_vec[i]))*(1/np.sqrt(T_vec[j]))*multivariate_normal.pdf(x, mean =mean, cov = cov, allow_singular = True)

D3_pair_integrand = np.vectorize(D3_pair_integrand, excluded = [1,2,3,4,5,6])


def D3_EM(theta, x, nu,m,gamma,n = 5):
    """
    D3 EM matrix as defined in paper. Used by Group-T and generalized skew-T

    Parameters
    ----------------------
    theta: np.array
        Curren estimate of Precision matrix
    x: np.array
        Vector of features /(1 observation)
    nu: list
        List of degress of freedom for each feature in x
    m: list
        list containing the group membership of each feautre in x
    gamma: np.array
        Current estimate of gamma vector
    n: int
        Number of Gaussian-Qudrature terms for the integration

    Returns
    ---------------------
     :np.array
        D3 EM matrix
    """


    d = theta.shape[0]
    combination_calculated = defaultdict(lambda: None)
    D = np.zeros(shape = theta.shape)
    denom = generalized_skew_t( x, theta, nu, gamma, n)

    if denom > 0:
        C = 1/np.sqrt((1/np.linalg.det(theta))*(2*np.pi) ** d)

        for i in range(D.shape[0]):
            for j in range(i, D.shape[0]):
                combintaion = ''.join(sorted(str(m[i])+str(m[j])))
                if combination_calculated[combintaion] is None:
                    D[i,j] =  C*integrate.fixed_quad(D3_pair_integrand, 0, 1 ,args = (theta, gamma, x,nu,i,j),n = n)[0]/denom
                    combination_calculated[combintaion] = D[i,j]
                else:
                    D[i,j] = combination_calculated[combintaion]

    return np.triu(D,0) + np.triu(D,1).T



def Gaussian_update(S, A,  eta):
    """
    Update theta estimate according to gaussian likelihood

    Parameters
    -----------------------
    S: np.array
        Covariance matrix
    A: np.array
    eta: float
        penalty
    """
    AT = A.T
    M =  0.5*(A+AT)/eta - S
    D, Q = np.linalg.eig(M)
    diag_m = np.diag(D+np.sqrt(D**2 + 4.0/eta))
    return np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T))



def Gaussian_update_outer_em(i, S, A,  eta):
    """
    Update theta estimate according to gaussian likelihood

    Parameters
    -----------------------
    S: np.array
        Covariance matrix
    A: np.array
    eta: float
        penalty
    """
    AT = A.T
    M =  0.5*(A+AT)/eta - S
    D, Q = np.linalg.eig(M)
    diag_m = np.diag(D+np.sqrt(D**2 + 4.0/eta))
    return np.real(0.5*eta*np.dot(Q, diag_m).dot(Q.T)), i



def t_em(X, nu, theta):
    d = X.shape[1]
    theta_pre = theta


    x = X
    M = np.einsum('nj,jk,nk->n', x, theta_pre, x)  # Mahalanobis distance
    tau = (nu + d)/(nu  + M)
    S = np.einsum('nj,n,nk->jk', x, tau, x)/float(x.shape[0])
    return S


def group_em(X, nu, theta, groups, nr_quad, pool = None):

    x = X
    d = x.shape[1]
    S = np.zeros(shape = (d,d))
 
    if pool is None:
        for i in range(x.shape[0]):
            S += group_t_EM_iteration(x[i],theta, nu, groups ,nr_quad)/x.shape[0]
    else:
        results = pool.starmap(group_t_EM_iteration,((x[i],theta, nu, groups ,nr_quad) for i in range(x.shape[0])))
        for result in results:
            S += result/x.shape[0]


    return S


def skew_group_em(X, nu, theta, gamma, groups, nr_quad, pool = None):

    d = X.shape[1]
    # E-step
    x = X

    S = np.zeros(shape = (d,d))
    G1 = np.zeros(shape = d)
    G2 = np.zeros(shape = (d,d))
    if pool is None:
        for i in range(x.shape[0]):
            S_TMP, G1_TMP,G2_TMP = skew_group_t_EM_iteration(x[i],theta, gamma, nu, groups ,nr_quad)
            S += S_TMP/x.shape[0]
            G1 += G1_TMP
            G2 += G2_TMP
    else:
        results = pool.starmap(skew_group_t_EM_iteration,((x[i],theta, gamma, nu, groups ,nr_quad) for i in range(x.shape[0])))
        for result in results:
            S += result[0]/x.shape[0]
            G1 += result[1]
            G2 += result[2]

    return S, G1,G2

def inner_em(X,A, theta, nu, eta, gamma, A_gamma = None, rho_gamma = 0, groups = None, lik_type = "t", nr_itr = 5, tol = 1e-5, nr_quad = 5, pool = None):
    """
    EM of estimation of theta

    Parameters
    --------------
    X: np.array
        Data matrix, Observations used to estimate theta_t
    A: np.array
        The array used in the theta update step. See paper
    theta: np.array
        Previous estimate of theta, used in EM-step
    nu: list
        List of degree of freedom for each feature in X
    eta: float 
        Penalization parameter as defined in paper
    gamma: np.array
        Previous estimate of theta, used in EM-step for skew-t
    A_gamma: np.array
        The array used in the gamma update step.
    rho_gamma:float,
        dual variable for gamma, only used for skew-T
    groups: list
        List defining the group memebership of each feature in X
    like_type: str
        Likelihood type: 't', 'group-t', 'skew-group-t'
    nr_itr: int
        Number of EM iterations. EM optimization terminates when either nrumber of EM iterations or error tolerance is reached.
    tol: float
        Tolerance error for EM iteration. EM optimization terminates when either nrumber of EM iterations or error tolerance is reached.
    nr_quad: int
        Number of Gaussian quaderatures for EM-step estimation. Used for group-t and skew-group-t.
    pool: multiprocessing.pool.Pool
        parallelization for the static case for the group t or skew t EM


    Returns
    --------------
    theta_new, gamma_new: tuple
        The theta and gamma estimates after the inner EM optimization step. Gamma is zero for t and group-t likelihoods.
    """


    if lik_type == "t":

        d = X.shape[1]
        iteration = 0
        theta_pre = theta
        while iteration < nr_itr:
            # E-step
            x = X
            M = np.einsum('nj,jk,nk->n', x, theta_pre, x)  # Mahalanobis distance
            tau = (nu + d)/(nu  + M)
            S = np.einsum('nj,n,nk->jk', x, tau, x)/float(x.shape[0])
            # M-step
            theta_new= Gaussian_update(S, A, eta)

            fro = np.linalg.norm(theta_new - theta_pre)
            if fro < tol:
                theta_pre = theta_new.copy()
                break
            else:
                theta_pre = theta_new.copy()
            iteration+=1

        gamma_new = gamma

    elif lik_type == "group-t":
        
        d = X.shape[1]
        iteration = 0
        theta_pre = theta

        while iteration < nr_itr:
            # E-step
            x = X
            S = np.zeros(shape = (d,d))
            if pool is None:
                for i in range(x.shape[0]):
                    S += group_t_EM_iteration(x[i],theta_pre, nu, groups ,nr_quad)/x.shape[0]
            else:
                results = pool.starmap(group_t_EM_iteration,((x[i],theta_pre, nu, groups ,nr_quad) for i in range(x.shape[0])))
                for result in results:
                    S += result/x.shape[0]
            # M-step
            theta_new= Gaussian_update(S, A, eta)

            fro = np.linalg.norm(theta_new - theta_pre)
            if fro < tol:
                theta_pre = theta_new.copy()
                break
            iteration+=1

        gamma_new = gamma

    elif lik_type == 'skew-group-t':

        d = X.shape[1]
        iteration = 0
        theta_pre = theta
        gamma_pre = gamma

        while iteration < nr_itr:
            # E-step
            x = X
            S = np.zeros(shape = (d,d))
            G1 = np.zeros(shape = d)
            G2 = np.zeros(shape = (d,d))
            if pool is None:
                for i in range(x.shape[0]):
                    S_TMP, G1_TMP,G2_TMP = skew_group_t_EM_iteration(x[i],theta_pre, gamma_pre, nu, groups ,nr_quad)
                    S += S_TMP/x.shape[0]
                    G1 += G1_TMP
                    G2 += G2_TMP
            else:
                results = pool.starmap(skew_group_t_EM_iteration,((x[i],theta_pre, gamma_pre, nu, groups ,nr_quad) for i in range(x.shape[0])))
                for result in results:
                    S += result[0]/x.shape[0]
                    G1 += result[1]
                    G2 += result[2]
            # M-step
            theta_new = Gaussian_update(S, A, eta)
            gamma_new = np.dot(np.linalg.inv(np.multiply(theta_new, G2)+rho_gamma*np.identity(d)), G1 +rho_gamma*A_gamma)
            fro = np.linalg.norm(theta_new - theta_pre)
            if fro < tol:
                theta_pre = theta_new.copy()
                break
            iteration+=1

    else:
        raise ValueError(f"likelihood {lik_type} not known")


    return theta_new, gamma_new




def group_t_EM_iteration(x_i,theta_pre, nu, groups ,nr_quad):
    

    d = theta_pre.shape[1]
    return np.multiply(np.outer(x_i,x_i), D1_EM(theta_pre,x_i,nu,groups, np.zeros(d), nr_quad))




def skew_group_t_EM_iteration(x_i,theta_pre, gamma_pre, nu, groups,nr_quad):


    D_1 = D1_EM(theta_pre,x_i,nu,groups, gamma_pre, nr_quad)
    D_2 = D2_EM(theta_pre,x_i,nu,groups, gamma_pre, nr_quad)
    D_3 = D3_EM(theta_pre,x_i,nu,groups, gamma_pre, nr_quad)


    S =  (np.multiply(np.outer(x_i,x_i),D_1 )-
            np.multiply(np.outer(gamma_pre,x_i),D_2 )-
            np.multiply(np.outer(x_i,gamma_pre),D_2.T )+
            np.multiply(np.outer(gamma_pre,gamma_pre),D_3 ) )
    
    G1 = np.multiply(theta_pre, D_2).dot(x_i)
    G2 = D_3

    return S, G1, G2
 




def theta_update(i, A, S , n_t, rho, rho_gamma, nr_graphs, A_gamma = None, groups = None, lik_type = "gaussian", X = None, nr_em_itr = 5, theta_init = None, gamma_init = None, nu = None, em_tol = 1e-3, 
    nr_quad = 5, pool = None):
    """
    Theta update. A wrapper used for inner em

    Parameters
    ----------------
    i: int,
        index of being updates
    A: np.array
        Matrix used in Guassian update of theta
    S: np.array
        Covariance matrix of time t. Used for Gaussian case.
    n_t: int
        Number of observations used in the paramters estimation at time t
    rho: float
        Penalization
    rho_gamma: float
        Penalization for gamma, only used for skew-t
    nr_graphs: int
        Number of graphs to be estimated
    A_gamma: np.array
        The array used in the gamma update step.
    groups: list
        List defining the group memebership of each feature in X
    like_type: str
        Likelihood type: 't', 'group-t', 'skew-group-t'
    X: np.array
        Data matrix, Observations used to estimate theta_t
    nr_em_itr: int
        Number of EM iterations. EM optimization terminates when either nrumber of EM iterations or error tolerance is reached.
    theta_init: np.array
        initial estimate of theta, only used in EM iteration.
    gamma_init: np.array
        initial estimate of gamma, only used in EM iteration.
    nu: float or list of floats
        List of degree of freedom for each feature in X. If student-t this should be a single float.
    em_tol: float
        Tolerance of EM optimization. EM optimization terminates when either nrumber of EM iterations or error tolerance is reached.
    nr_quad: int
        Number of Gaussian quaderatures for EM-step estimation. Used for group-t and skew-group-t.
    pool: multiprocessing.pool.Pool
        parallelization for the static case for the group t or skew t EMM


    """
    
    if i == nr_graphs-1 or i == 0:
        eta = n_t/rho/2.0
    else:
        eta = n_t/rho/3.0

    if lik_type == "gaussian":
        theta = Gaussian_update(S, A, eta)
        gamma = None
    elif lik_type == "t":
        theta, gamma = inner_em( X,A, theta_init, nu, eta, gamma_init, groups = None, lik_type = "t", nr_itr = nr_em_itr, tol = em_tol)
    elif lik_type == "group-t":
        # print("group-t update")
        theta, gamma = inner_em(X,A, theta_init, nu, eta, gamma_init, groups = groups, lik_type = "group-t", nr_itr = nr_em_itr, tol = em_tol, nr_quad = nr_quad, pool=pool)
    elif lik_type == "skew-group-t":
        # print("group-t update")
        if i == 0 or i == nr_graphs-1:
            rho_gamma = rho_gamma
        else:
            rho_gamma = 2.0*rho_gamma
        theta, gamma = inner_em(X,A, theta_init, nu, eta, gamma_init, A_gamma = A_gamma, rho_gamma = rho_gamma, groups = groups, lik_type = "skew-group-t", nr_itr = nr_em_itr, tol = em_tol, nr_quad = nr_quad)
    else:
        raise ValueError(f"likelihood {lik_type} not known")

    return theta, gamma, i



def soft_threshold_odd( A, lamda):

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


def global_reconstruction(A, eta):
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



def ridge_penalty(A, eta):
    """
    Ridge/Laplacian penalty

    Parameters
    ------------------
    A: np.array,
    
    lamda: float,
        regularization
    """

    return A/(1.0+2.0*eta)


def block_wise_reconstruction(A,  eta , xtol=1e-5):
    """
    Block-wise reconstruction: l_infty norm

    Parameters
    ------------------
    A: np.array,
    
    eta: float,
        regularization
    """
    def f(x, v):
        # Solve for hidden threshold

        return  np.sum([np.max([np.abs(v[i]) - x, 0]) for i in range(len(v))]) -1

    # LOOP OVER COLUMNS
    if eta > 0.0:
        E = np.zeros(shape=A.shape)
        for i in range(A.shape[1]):
            if np.sum(np.abs(A[:,i])) <= eta:
                continue

            l_opt = optimize.bisect(f = f, a = 0, b = np.sum(np.abs(A[:,i]/eta)), args = (A[:,i]/eta,), xtol=xtol)

            E[:,i] = A[:,i] - eta*soft_threshold_odd(A[:,i]/eta, l_opt)
    else:
        E = A.copy()

    return E



def perturbed_node(theta_i, theta_i_1, U_i, U_i_1, kappa, rho, tol = 1e-8, max_iter = 1000):
    """
    Parameters
    perturbed_node. ADMM solver
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
        V = global_reconstruction(A,kappa/(2.0*rho))

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


    return Y1, Y2




def update_w(Lw, w, rho, k, theta, LstarS, Y,y, i):
    d = theta.shape[0]
    LstarLw = L_star(Lw)
    DstarDw = D_star(np.diag(Lw))
    grad = LstarS - L_star(Y + rho * theta) + D_star(y - rho * k) + rho * (LstarLw + DstarDw)
    eta = 1 / (2*rho * (2*d - 1))
    w = w - eta * grad
    w [w  < 0] = 0
    Lw = L_op(w)

    return Lw, w, i

def update_theta_laplace(rho, J, Lw, Y, W1, W2, t, T, dynamic ):
    # update Theta
    if dynamic:
        if t == 0:
            rho = rho*2
            At = (Lw+W1)/2.0 
        elif t == T-1:
            At = (Lw+W2)/2.0
            rho = rho*2
        else:
            At = (Lw+W1+W2)/3.0
            rho = rho*3
    else:
        At = Lw
    
    At = (At+At.T)/2.0
    At = At+J

    gamma, V =  np.linalg.eigh(rho * At - Y)
    theta = np.dot(V,np.diag((gamma + np.sqrt(gamma**2 + 4 * rho)) / (2 * rho))).dot(V.T)- J
    return theta, t


def A_op(w):
    d = int((1+np.sqrt(1+8*len(w)))/2)
    A = np.zeros((d,d))
    A[np.triu_indices(d,1)] = w
    return  A+A.T

def A_inv_op( A):
    d = A.shape[0]
    return A[np.triu_indices(d,1)]

def L_inv_op( L):
    d = L.shape[0]
    return -L[np.triu_indices(d,1)]

def L_op( w):
    A = A_op(w)
    return np.diag(np.sum(A, axis = 0)) - A

def L_star( L):
    d = L.shape[0]
    w = np.zeros(int(d*(d-1)/2))
    for i in range(1,d+1):
        for j in range(1,i):
            s = int(i-j + (j-1)*(2*d-j)/2)
            w[s-1] = L[i-1,i-1]-L[i-1,j-1]-L[j-1,i-1]+L[j-1,j-1]

    return w

def D_star( a):
    d = len(a)
    p = np.zeros(int(d*(d-1)/2))
    for i in range(1,d+1):
        for j in range(1,i):
            s = int(i-j + (j-1)*(2*d-j)/2)
            p[s-1] = a[i-1]+a[j-1]

    return p