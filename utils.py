
import numpy as np
import sklearn
import networkx as nx

abs_tol = 1e-2

def calc_one_zero_error(T,Estimate, ratio = True):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    error = np.sum(~(np.sign(T[np.triu_indices(T.shape[0], k = 1)]) == np.sign(Estimate[np.triu_indices(Estimate.shape[0], k = 1)])))
    if ratio:
        error = error/float(d*(d-1)/2)
    return error

def calc_f1(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    y_true = 1-np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = 1-np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.f1_score(y_true,y_pred)


def calc_precision(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    y_true = 1-np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = 1-np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.precision_score(y_true,y_pred)

def calc_recall(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    y_true = 1-np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = 1-np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.recall_score(y_true,y_pred)

def calc_density(prec):
    tmp = prec.copy()
    np.fill_diagonal(tmp,0)
    G = nx.from_numpy_array(tmp)
    # G = nx.fast_gnp_random_graph(300,0.3)
    return nx.density(G)

def calc_roc_auc(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    y_true = 1-np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = 1-np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.roc_auc_score(y_true,y_pred)

def calc_balanced_accuaray(T,Estimate):
    d = T.shape[0]
    T[np.abs(T)<abs_tol] = 0
    Estimate[np.abs(Estimate)<abs_tol] = 0
    y_true = 1-np.abs(np.sign(T[np.triu_indices(d, 1)]))
    y_pred = 1-np.abs(np.sign(Estimate[np.triu_indices(d, 1)]))
    return sklearn.metrics.balanced_accuracy_score(y_true,y_pred)
