"""
This file contains A-IHT I and A-IHT II algorithms, as presented in the paper
Bayesian Coresets: Revisiting the Nonconvex Optimization Perspective (https://arxiv.org/abs/2007.00715).
Jacky Y. Zhang, Rajiv Khanna, Anastasios Kyrillidis, and Oluwasanmi Koyejo. (AISTATS 2021)

Both numpy version and pytorch version are offered, where the torch version can be run on GPU for acceleration.
6 functions are included:
iht_obj(y, A, w):                                                   calculate the objective value
l2_projection_torch(w, K, L=None, already_K_sparse=False, K_sparse_supp=None)       l2 projection implemented by numpy
l2_projection_torch(w, K, L=None, already_K_sparse=False, K_sparse_supp=None)       l2 projection implemented by torch
a_iht_i(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):         A-IHT I implemented by numpy
a_iht_ii(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):        A-IHT II implemented by numpy
a_iht_ii_torch(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):  A-IHT II implemented by torch

The optimization objective is
    argmin_w ||y - Aw||^2    s.t.    ||w||_0 <= K    and    w >= 0    (optional: and sum(w) = L)
        where   y is of shape (M, 1),
                A is of shape (M, N),
                w is of shape (N, 1),
                K is a positive integer,
                L is a positive number.
"""

import numpy as np
import torch


def iht_obj(y, A, w):
    """
    Calculate the quadratic objective value given w
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param w: numpy.ndarray of shape (N, 1)
    :return: float objective value
    """
    return np.linalg.norm(y - A.dot(w), ord=2)


def l2_projection_numpy(w, K, L=None, already_K_sparse=False, K_sparse_supp=None):
    """
    If L is None, project w to the K-sparsity constrained and non-negative region;
    if L is not None, project w the K-sparsity constrained, non-negative and the sum(w) = L region;
    the projection is optimal in l2 distance.
    For the projection with L, see the paper
    Sparse projections onto the simplex (https://arxiv.org/pdf/1206.1529.pdf).
    Anastasios Kyrillidis, Stephen Becker, Volkan Cevher, Christoph Koch (ICML 2013)
    :param w: numpy.ndarray of shape (N, 1)
    :param K: int (sparsity constraint), positive
    :param L: float, positive
    :param already_K_sparse: bool. If the input w has been already K-sparse, put 'True' to for a faster projection
    :param K_sparse_supp: list. If the input w has been already K-sparse, put its support here
    :return: w: numpy.ndarray of shape (N, 1). A new vector that is the projected w
             selected_support: list of integer indexes (the support of the w).
    """
    N = w.shape[0]
    if L is None:
        if already_K_sparse:
            w_projected = w.copy()
            w_projected[w_projected < 0] = 0
            return w_projected, K_sparse_supp
        else:
            index_w = np.flip(np.argsort(np.squeeze(w)))
            index_w = np.squeeze(index_w).tolist()
            w_projected = np.zeros([N, 1])
            selected_support = index_w[0:K]
            w_projected[selected_support] = w[index_w[0:K]]  # projection
            w_projected[w_projected < 0] = 0  # truncate negative entries
            return w_projected, selected_support
    else:
        if already_K_sparse:
            w_selected = w[K_sparse_supp]
        else:
            index_w = np.flip(np.argsort(np.squeeze(w)))
            index_w = np.squeeze(index_w).tolist()
            K_sparse_supp = index_w[:K]
            w_selected = w[K_sparse_supp]

        w_projected = np.zeros([N, 1])
        accumulate = 0
        rho = 0
        for j in range(K):
            w_j = w_selected[j].item()
            accumulate += w_j
            if w_j > (accumulate - L) / (j + 1):
                rho = j
            else:
                break
        tau = (w_selected[:(rho + 1)].sum() - L) / (rho + 1)
        w_selected = w_selected - tau
        w_selected[w_selected < 0] = 0
        w_projected[K_sparse_supp] = w_selected
        return w_projected, K_sparse_supp


def l2_projection_torch(w, K, L=None, already_K_sparse=False, K_sparse_supp=None):
    """
    If L is None, project w to the K-sparsity constrained and non-negative region;
    if L is not None, project w the K-sparsity constrained, non-negative and the sum(w) = L region;
    the projection is optimal in l2 distance.
    For the projection with L, see the paper
    Sparse projections onto the simplex (https://arxiv.org/pdf/1206.1529.pdf).
    Anastasios Kyrillidis, Stephen Becker, Volkan Cevher, Christoph Koch (ICML 2013)
    :param w: torch.tensor of shape (N, 1)
    :param K: int (sparsity constraint), positive
    :param L: float, positive
    :param already_K_sparse: bool. If the input w has been already K-sparse, put 'True' to for a faster projection
    :param K_sparse_supp: list. If the input w has been already K-sparse, put its support here
    :return: w: torch.tensor of shape (N, 1). A new vector that is the projected w
             selected_support: list of integer indexes (the support of the w).
    """
    device = w.device  # should be on the same device as A
    dtype = w.dtype  # should be the same dtype as A
    N = w.shape[0]
    if L is None:
        if already_K_sparse:
            w_projected = w.clone()
            w_projected[w_projected < 0] = 0
            return w_projected, K_sparse_supp
        else:
            index_w = torch.argsort(w.squeeze())
            index_w = index_w.flip(0)
            index_w = index_w.squeeze().tolist()
            w_projected = torch.zeros([N, 1], dtype=dtype, device=device)
            selected_support = index_w[:K]
            w_projected[selected_support] = w[selected_support]  # projection
            w_projected[w_projected < 0] = 0  # truncate negative entries
            return w_projected, selected_support
    else:
        if already_K_sparse:
            w_selected = w[K_sparse_supp]
        else:
            index_w = torch.argsort(w.squeeze())
            index_w = index_w.flip(0)
            index_w = index_w.squeeze().tolist()
            K_sparse_supp = index_w[:K]
            w_selected = w[K_sparse_supp]

        w_projected = torch.zeros([N, 1], dtype=dtype, device=device)
        accumulate = 0
        rho = 0
        for j in range(K):
            w_j = w_selected[j].item()
            accumulate += w_j
            if w_j > (accumulate - L) / (j + 1):
                rho = j
            else:
                break
        tau = (w_selected[:(rho + 1)].sum() - L) / (rho + 1)
        w_selected = w_selected - tau
        w_selected[w_selected < 0] = 0
        w_projected[K_sparse_supp] = w_selected
        return w_projected, K_sparse_supp


def a_iht_i(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):
    """
    A-IHT I implemented by numpy
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: w: numpy.ndarray of shape (N, 1)
             supp: list of integer indexes (the support of the w)
    """
    (M, N) = A.shape
    if len(y.shape) != 2:
        raise ValueError('y should have shape (M, 1)')

    # Initialize transpose of measurement matrix
    A_t = A.T

    # Initialization
    w_cur = np.zeros([N, 1])
    y_cur = np.zeros([N, 1])
    # x_cur = np.random.random([N, 1])
    # y_cur = np.random.random([N, 1])

    A_w_cur = np.zeros([M, 1])
    Y_i = []

    # auxiliary variables
    complementary_Yi = np.ones([N, 1])
    i = 1

    while i <= max_iter_num:
        w_prev = w_cur
        if i == 1:
            res = y
            der = A_t.dot(res)  # compute gradient
        else:
            res = y - A_w_cur - tau * A_diff
            der = A_t.dot(res)  # compute gradient
        A_w_prev = A_w_cur
        complementary_Yi[Y_i] = 0
        ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
        complementary_Yi[Y_i] = 1
        S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()  # identify active subspace
        ider = der[S_i]
        Pder = A[:, S_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        b = y_cur + mu_bar * der  # gradient descent
        w_cur, X_i = l2_projection_numpy(b, K, L=L)

        A_w_cur = A[:, X_i].dot(w_cur[X_i])
        res = y - A_w_cur

        if i == 1:
            A_diff = A_w_cur
        else:
            A_diff = A_w_cur - A_w_prev

        temp = A_diff.T.dot(A_diff)
        if temp > 0:
            tau = res.T.dot(A_diff) / temp
        else:
            tau = res.T.dot(A_diff) / 1e-6

        y_cur = w_cur + tau * (w_cur - w_prev)
        Y_i = np.nonzero(y_cur)[0].tolist()

        # print out objective function value during optimization of IHT
        if verbose and i % 50 == 1:
            print('at iteration {}, the objective value is: {}'.format(i, iht_obj(y, A, w_cur)))

        # stop criterion
        if i > 1 and (np.linalg.norm(w_cur - w_prev) < tol * np.linalg.norm(w_cur)):
            break
        i = i + 1

    # finished
    w = w_cur
    supp = np.nonzero(w_cur)[0].tolist()  # support of the output solution
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                              iht_obj(y, A, w_cur)))
    return w, supp


def a_iht_ii(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):
    """
    A-IHT II implemented by numpy
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: w: numpy.ndarray of shape (N, 1)
             supp: list of integer indexes (the support of the w)
    """
    (M, N) = A.shape
    if len(y.shape) != 2:
        raise ValueError('y should have shape (M, 1)')
    # Initialize transpose of measurement matrix
    A_t = A.T

    # Initialize to zero vector
    w_cur = np.zeros([N, 1])
    y_cur = np.zeros([N, 1])
    # w_cur = np.random.random([N, 1])
    # y_cur = np.random.random([N, 1])

    A_w_cur = np.zeros([M, 1])
    Y_i = []

    # auxiliary variables
    complementary_Yi = np.ones([N, 1])
    i = 1

    while i <= max_iter_num:
        w_prev = w_cur
        if i == 1:
            res = y
            der = A_t.dot(res)  # compute gradient
        else:
            res = y - A_w_cur - tau * A_diff
            der = A_t.dot(res)  # compute gradient

        A_w_prev = A_w_cur
        complementary_Yi[Y_i] = 0
        ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
        complementary_Yi[Y_i] = 1
        S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()  # identify active subspace
        ider = der[S_i]
        Pder = A[:, S_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        b = y_cur + mu_bar * der  # gradient descent
        w_cur, X_i = l2_projection_numpy(b, K, L=L)

        A_w_cur = A[:, X_i].dot(w_cur[X_i])
        res = y - A_w_cur
        der = A_t.dot(res)  # compute gradient
        ider = der[X_i]
        Pder = A[:, X_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        w_cur[X_i] = w_cur[X_i] + mu_bar * ider  # debias
        w_cur, _ = l2_projection_numpy(w_cur, K, already_K_sparse=True, K_sparse_supp=X_i, L=L)

        A_w_cur = A[:, X_i].dot(w_cur[X_i])
        res = y - A_w_cur

        if i == 1:
            A_diff = A_w_cur
        else:
            A_diff = A_w_cur - A_w_prev

        temp = A_diff.T.dot(A_diff)
        if temp > 0:
            tau = res.T.dot(A_diff) / temp
        else:
            tau = res.T.dot(A_diff) / 1e-6

        y_cur = w_cur + tau * (w_cur - w_prev)
        Y_i = np.nonzero(y_cur)[0].tolist()

        # print out objective function value during optimization of IHT
        if verbose and i % 50 == 1:
            print('at iteration {}, the objective value is: {}'.format(i, iht_obj(y, A, w_cur)))

        # stop criterion
        if (i > 1) and (np.linalg.norm(w_cur - w_prev) < tol * np.linalg.norm(w_cur)):
            break
        i = i + 1

    # finished
    w = w_cur
    supp = np.nonzero(w_cur)[0].tolist()  # support of the output solution
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                              iht_obj(y, A, w_cur)))
    return w, supp


def a_iht_ii_torch(y, A, K, tol=1e-5, max_iter_num=300, verbose=True, L=None):
    """
    A-IHT II implemented by pytorch
    :param y: torch.tensor of shape (M, 1)
    :param A: torch.tensor of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: w: torch.tensor of shape (N, 1)
             supp: list of integer indexes (the support of the w)
    """
    device = y.device  # should be on the same device as A
    dtype = y.dtype  # should be the same dtype as A
    if verbose:
        print('running A-IHT II on {}'.format(device))
    (M, N) = A.shape
    # Initialize transpose of measurement matrix
    A_t = A.T
    # Initialize to zero vector
    w_cur = torch.zeros([N, 1], dtype=dtype, device=device)
    y_cur = torch.zeros([N, 1], dtype=dtype, device=device)

    A_w_cur = torch.zeros([M, 1], dtype=dtype, device=device)
    Y_i = []

    # auxiliary variables
    complementary_Yi = torch.ones([N, 1], dtype=dtype, device=device)
    i = 1

    for i in range(1, max_iter_num + 1):
        w_prev = w_cur
        if i == 1:
            res = y
            der = A_t.mm(res)
        else:
            res = y - A_w_cur - tau * A_diff
            der = A_t.mm(res)

        A_w_prev = A_w_cur
        complementary_Yi[Y_i] = 0
        ind_der = torch.argsort(torch.abs((der * complementary_Yi).squeeze()))
        ind_der = ind_der.flip(0)
        complementary_Yi[Y_i] = 1
        S_i = Y_i + (ind_der[:K]).squeeze().tolist()
        ider = der[S_i]
        Pder = A[:, S_i].mm(ider)
        mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
        b = y_cur + mu_bar * der
        w_cur, X_i = l2_projection_torch(b, K, L=L)

        A_w_cur = A[:, X_i].mm(w_cur[X_i])
        res = y - A_w_cur
        der = A_t.mm(res)
        ider = der[X_i]
        Pder = A[:, X_i].mm(ider)
        mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
        w_cur[X_i] = w_cur[X_i] + mu_bar * ider
        w_cur, _ = l2_projection_torch(w_cur, K, already_K_sparse=True, K_sparse_supp=X_i, L=L)

        A_w_cur = A[:, X_i].mm(w_cur[X_i])
        res = y - A_w_cur
        if i == 1:
            A_diff = A_w_cur
        else:
            A_diff = A_w_cur - A_w_prev
        temp = A_diff.T.mm(A_diff)
        if temp > 0:
            tau = res.T.mm(A_diff) / temp
        else:
            tau = res.T.mm(A_diff) / 1e-6
        y_cur = w_cur + tau * (w_cur - w_prev)
        Y_i = y_cur.squeeze().nonzero().squeeze().tolist()
        if isinstance(Y_i, int):
            Y_i = [Y_i]

        # print out objective function value during optimization of IHT
        if verbose and i % 50 == 1:
            obj_value = torch.norm(y - A.mm(w_cur))
            print('at iteration {}, the objective value is: {}'.format(i, obj_value))

        # stop criterion
        if (i > 1) and (torch.norm(w_cur - w_prev) < tol * torch.norm(w_cur)):
            break

    w = w_cur
    supp = w_cur.squeeze().nonzero().squeeze().tolist()  # support of the output solution
    if isinstance(supp, int):
        supp = [supp]
    obj_value = torch.norm(y - A.mm(w_cur))
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp), obj_value))
    return w, supp
