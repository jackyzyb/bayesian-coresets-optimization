"""
This file contains examples that use A-IHT I and A-IHT II algorithms from ./accelerated_iht.py

The optimization objective is
    argmin_w ||y - Aw||^2    s.t.    ||w||_0 <= K    and    w >= 0    (optional: and sum(w) = L)
        where   y is of shape (M, 1),
                A is of shape (M, N),
                w is of shape (N, 1),
                K is a positive integer,
                L is a positive number.

Associated paper:
Bayesian Coresets: Revisiting the Nonconvex Optimization Perspective (https://arxiv.org/abs/2007.00715).
Jacky Y. Zhang, Rajiv Khanna, Anastasios Kyrillidis, and Oluwasanmi Koyejo. (AISTATS 2021)
"""
import numpy as np
import torch
from accelerated_iht import *

# settings
M = 400
N = 1000
K = 100
L = None  # None or a positive number
np.random.seed(233)
A = np.random.rand(M, N) + 0.5
true_supp = np.random.permutation(N)[:K]
true_w = np.zeros([N, 1])
true_w[true_supp] = np.random.rand(K, 1)
y = A.dot(true_w)

# projection
w = np.array([1, -1, -0.5, 0]).reshape([-1, 1])
k_proj = 2
w_projected, supp = l2_projection_numpy(w, k_proj, L=2)
print('original w: {}. Sparsity: {}'.format(w, k_proj))
print('projected w: {}'.format(w_projected))
print('support: {}'.format(supp))
w_projected, supp = l2_projection_numpy(w, k_proj, L=2, K_sparse_supp=[0, 1], already_K_sparse=True)
print('projected w: {}'.format(w_projected))
print('support: {}'.format(supp))


# A-IHT I by numpy
print('\nusing A-IHT I by numpy...')
w, supp = a_iht_i(y, A, K, L=L)
print('A-IHT I (numpy) finds solution with objective value {}; sum(w) is {}\n'.format(iht_obj(y, A, w), w.sum()))

# A-IHT II by numpy
print('using A-IHT II by numpy...')
w, supp = a_iht_ii(y, A, K, L=L)
print('A-IHT II (numpy) finds solution with objective value {}; sum(w) is {}\n'.format(iht_obj(y, A, w), w.sum()))

# A-IHT II by torch
A = torch.tensor(A)
y = torch.tensor(y)
print('using A-IHT II by torch...')
w, supp = a_iht_ii_torch(y, A, K, L=L)
obj_value = torch.norm(y - A.mm(w))
print('A-IHT II (torch) finds solution with objective value {}; sum(w) is {}\n'.format(obj_value, w.sum()))





