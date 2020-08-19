import numpy as np
import torch
from accelerated_iht import iht_obj, a_iht_i, a_iht_ii, a_iht_ii_torch

# settings
M = 300
N = 1000
K = 100
np.random.seed(233)
A = np.random.rand(M, N) - 0.5
true_supp = np.random.permutation(N)[:K]
true_x = np.zeros([N, 1])
true_x[true_supp] = np.random.rand(K, 1)
y = A.dot(true_x)

# A-IHT I by numpy
print('\nusing A-IHT I by numpy...')
x, supp = a_iht_i(y, A, K)
print('A-IHT I (numpy) finds solution with objective value {}\n'.format(iht_obj(y, A, x)))

# A-IHT II by numpy
print('using A-IHT II by numpy...')
x, supp = a_iht_ii(y, A, K)
print('A-IHT II (numpy) finds solution with objective value {}\n'.format(iht_obj(y, A, x)))

# A-IHT II by torch
A = torch.tensor(A)
y = torch.tensor(y)
print('using A-IHT II by torch...')
x, supp = a_iht_ii_torch(y, A, K)
obj_value = torch.norm(y - A.mm(x))
print('A-IHT II (torch) finds solution with objective value {}\n'.format(obj_value))





