
![Tests](https://github.com/ragnarlevi/DyGraph/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ragnarlevi/DyGraph/badge.svg?branch=main)](https://coveralls.io/github/ragnarlevi/DyGraph?branch=main)

# DyGraph

A package for dynamic graph estimation. 

<code>pip install DyGraph </code>




```python
from sklearn.datasets import make_sparse_spd_matrix
import DyGraph as dg
import numpy as np
from scipy.stats import multivariate_t as mvt

```
Generate some data.

```python


d = 5  # number of nodes
A = make_sparse_spd_matrix(d, alpha=0.6)
X = mvt.rvs(loc = np.zeros(d),df = 4, shape = np.linalg.inv(A), size=200)


max_iter = 100
obs_per_graph = 50
alpha = 0.05
kappa = 0.1
kappa_gamma = 0.1
tol = 1e-4

```
Gaussian

```python
dg_opt = dg.dygl_inner_em(X,  obs_per_graph = obs_per_graph, max_iter = max_iter, lamda = alpha,  kappa = kappa, tol = tol, lik_type='gaussian')
dg_opt.fit(temporal_penalty = 'element-wise')

```

access the graphs via:

```python
dg_opt.theta

```



t, inner and outer. Can give degrees of freedom, or estimate

```python
# inner
dg_opt_t_inner = dg.dygl_inner_em(X = X, obs_per_graph = obs_per_graph,  max_iter = max_iter, lamda = alpha, kappa = kappa, tol = tol, lik_type='t')
dg_opt_t_inner.fit(temporal_penalty = 'element-wise')
# outer
dg_opt_t_outer = dg.dygl_outer_em(X = X, obs_per_graph = obs_per_graph,  max_iter = max_iter, lamda = alpha,  kappa = kappa, tol = tol, lik_type='t')
dg_opt_t_outer.fit(temporal_penalty = 'element-wise', nu = [4]*4)  # Note one nu/DoF for each graph.

```

Group t

```python
# outer
dg_opt_gt_outer = dg.dygl_outer_em(X = X, obs_per_graph = obs_per_graph,  max_iter = max_iter, lamda = alpha,  kappa = kappa, tol = tol, lik_type='group-t')
dg_opt_gt_outer.fit(temporal_penalty = 'element-wise', nu = [[4] * d]*4, groups = [0]*d)  # Note one nu/DoF for each graph and feature/group, all features in same group

```


skew group t

```python
# outer
dg_opt_sgt_outer = dg.dygl_outer_em(X = X, obs_per_graph = obs_per_graph,  max_iter = max_iter, lamda = alpha,  kappa = kappa, kappa_gamma = kappa_gamma, tol = tol, lik_type='skew-group-t')
dg_opt_sgt_outer.fit(temporal_penalty = 'element-wise', nu = None, groups = [0]*d)  # nus estimate, all features in same group

```