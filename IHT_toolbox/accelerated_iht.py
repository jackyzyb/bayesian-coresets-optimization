"""
This file contains A-IHT I and A-IHT II algorithms, as presented in preprint
Bayesian Coresets: An Optimization Perspective (https://arxiv.org/abs/2007.00715).

Both numpy version and pytorch version are offered, where the torch version can be run on GPU for acceleration.
4 functions are included:
iht_obj(y, A, x):                                                   calculate the objective value
a_iht_i(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):         A-IHT I implemented by numpy
a_iht_ii(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):        A-IHT II implemented by numpy
a_iht_ii_torch(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):  A-IHT II implemented by torch

The optimization objective is
    argmin_x ||y - Ax||^2    s.t.    |x|_0 <= K    and     x >= 0
        where   y is of shape (M, 1),
                A is of shape (M, N),
                K is an positive integer
"""

import numpy as np
import torch


def iht_obj(y, A, x):
    """
    calculate the quadratic objective value given x
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param x: numpy.ndarray of shape (N, 1)
    :return: float objective value
    """
    return np.linalg.norm(y - A.dot(x), ord=2)


def a_iht_i(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):
    """
    A-IHT I implemented by numpy
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: x: numpy.ndarray of shape (N, 1)
             supp: list of integer indexes (the support of the x)
    """
    (M, N) = A.shape
    if len(y.shape) != 2:
        raise ValueError('y should have shape (M, 1)')

    # Initialize transpose of measurement matrix
    A_t = A.T

    # Initialization
    x_cur = np.zeros([N, 1])
    y_cur = np.zeros([N, 1])
    # x_cur = np.random.random([N, 1])
    # y_cur = np.random.random([N, 1])

    A_x_cur = np.zeros([M, 1])
    Y_i = []

    # auxiliary variables
    complementary_Yi = np.ones([N, 1])
    i = 1

    while (i <= max_iter_num):
        x_prev = x_cur
        if (i == 1):
            res = y
            der = A_t.dot(res)  # compute gradient
        else:
            res = y - A_x_cur - tau * A_diff
            der = A_t.dot(res)  # compute gradient
        A_x_prev = A_x_cur
        complementary_Yi[Y_i] = 0
        ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
        complementary_Yi[Y_i] = 1
        S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()  # identify active subspace
        ider = der[S_i]
        Pder = A[:, S_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        b = y_cur + mu_bar * der  # gradient descent
        ind_b = np.flip(np.argsort(np.squeeze(b)))
        ind_b = np.squeeze(ind_b).tolist()
        x_cur = np.zeros([N, 1])
        S_i_temp = ind_b[0:K]
        x_cur[S_i_temp] = b[ind_b[0:K]]  # projection
        X_i = S_i_temp
        x_cur[x_cur < 0] = 0  # truncate negative entries

        A_x_cur = A[:, X_i].dot(x_cur[X_i])
        res = y - A_x_cur

        if (i == 1):
            A_diff = A_x_cur
        else:
            A_diff = A_x_cur - A_x_prev

        temp = A_diff.T.dot(A_diff)
        if (temp > 0):
            tau = res.T.dot(A_diff) / temp
        else:
            tau = res.T.dot(A_diff) / 1e-6

        y_cur = x_cur + tau * (x_cur - x_prev)
        Y_i = np.nonzero(y_cur)[0].tolist()

        # print out objective function value during optimization of IHT
        if verbose and i % 50 == 1:
            print('at iteration {}, the objective value is: {}'.format(i, iht_obj(y, A, x_cur)))

        # stop criterion
        if i > 1 and (np.linalg.norm(x_cur - x_prev) < tol * np.linalg.norm(x_cur)):
            break
        i = i + 1

    # finished
    x = x_cur
    supp = np.nonzero(x_cur)[0].tolist()    # support of the output solution
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                              iht_obj(y, A, x_cur)))
    return x, supp


def a_iht_ii(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):
    """
    A-IHT I implemented by numpy
    :param y: numpy.ndarray of shape (M, 1)
    :param A: numpy.ndarray of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: x: numpy.ndarray of shape (N, 1)
             supp: list of integer indexes (the support of the x)
    """
    (M, N) = A.shape
    if len(y.shape) != 2:
        raise ValueError('y should have shape (M, 1)')
    # Initialize transpose of measurement matrix
    A_t = A.T

    # Initialize to zero vector
    x_cur = np.zeros([N, 1])
    y_cur = np.zeros([N, 1])
    # x_cur = np.random.random([N, 1])
    # y_cur = np.random.random([N, 1])

    A_x_cur = np.zeros([M, 1])
    Y_i = []

    # auxiliary variables
    complementary_Yi = np.ones([N, 1])
    i = 1

    while (i <= max_iter_num):
        x_prev = x_cur
        if (i == 1):
            res = y
            der = A_t.dot(res)  # compute gradient
        else:
            res = y - A_x_cur - tau * A_diff
            der = A_t.dot(res)  # compute gradient

        A_x_prev = A_x_cur
        complementary_Yi[Y_i] = 0
        ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
        complementary_Yi[Y_i] = 1
        S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()  # identify active subspace
        ider = der[S_i]
        Pder = A[:, S_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        b = y_cur + mu_bar * der  # gradient descent
        ind_b = np.flip(np.argsort(np.squeeze(b)))
        ind_b = np.squeeze(ind_b).tolist()
        x_cur = np.zeros([N, 1])
        S_i_temp = ind_b[0:K]
        x_cur[S_i_temp] = b[ind_b[0:K]]  # projection
        x_cur[x_cur < 0] = 0  # hard threshold negative entries
        X_i = S_i_temp
        A_x_cur = A[:, X_i].dot(x_cur[X_i])
        res = y - A_x_cur
        der = A_t.dot(res)  # compute gradient
        ider = der[X_i]
        Pder = A[:, X_i].dot(ider)
        mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
        x_cur[X_i] = x_cur[X_i] + mu_bar * ider  # debias
        x_cur[x_cur < 0] = 0  # hard threshold negative entries

        A_x_cur = A[:, X_i].dot(x_cur[X_i])
        res = y - A_x_cur

        if (i == 1):
            A_diff = A_x_cur
        else:
            A_diff = A_x_cur - A_x_prev

        temp = A_diff.T.dot(A_diff)
        if (temp > 0):
            tau = res.T.dot(A_diff) / temp
        else:
            tau = res.T.dot(A_diff) / 1e-6

        y_cur = x_cur + tau * (x_cur - x_prev)
        Y_i = np.nonzero(y_cur)[0].tolist()

        # print out objective function value during optimization of IHT
        if verbose and i % 50 == 1:
            print('at iteration {}, the objective value is: {}'.format(i, iht_obj(y, A, x_cur)))

        # stop criterion
        if (i > 1) and (np.linalg.norm(x_cur - x_prev) < tol * np.linalg.norm(x_cur)):
            break
        i = i + 1

    # finished
    x = x_cur
    supp = np.nonzero(x_cur)[0].tolist()  # support of the output solution
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                            iht_obj(y, A, x_cur)))
    return x, supp


def a_iht_ii_torch(y, A, K, tol=1e-5, max_iter_num=300, verbose=True):
    """
    A-IHT II implemented by pytorch
    :param y: torch.tensor of shape (M, 1)
    :param A: torch.tensor of shape (M, N)
    :param K: int (sparsity constraint)
    :param tol: float (tolerance of the ending criterion)
    :param max_iter_num: int (maximum iteration number)
    :param verbose: boolean (controls intermediate text output)
    :return: x: torch.tensor of shape (N, 1)
             supp: list of integer indexes (the support of the x)
    """
    device = y.device   # should be on the same device as A
    dtype = y.dtype     # should be the same dtype as A
    if verbose:
        print('running A-IHT II on {}'.format(device))
    (M, N) = A.shape
    # Initialize transpose of measurement matrix
    A_t = A.T
    # Initialize to zero vector
    x_cur = torch.zeros([N, 1], dtype=dtype, device=device)
    y_cur = torch.zeros([N, 1], dtype=dtype, device=device)

    A_x_cur = torch.zeros([M, 1], dtype=dtype, device=device)
    Y_i = []

    # auxiliary variables
    complementary_Yi = torch.ones([N, 1], dtype=dtype, device=device)
    i = 1

    for i in range(1, max_iter_num + 1):
        x_prev = x_cur
        if (i == 1):
            res = y
            der = A_t.mm(res)
        else:
            res = y - A_x_cur - tau * A_diff
            der = A_t.mm(res)

        A_x_prev = A_x_cur
        complementary_Yi[Y_i] = 0
        ind_der = torch.argsort(torch.abs((der * complementary_Yi).squeeze()))
        complementary_Yi[Y_i] = 1
        S_i = Y_i + (ind_der[-K:]).squeeze().tolist()
        ider = der[S_i]
        Pder = A[:, S_i].mm(ider)
        mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
        b = y_cur + mu_bar * der
        ind_b = torch.argsort(b.squeeze())
        ind_b = ind_b.squeeze().tolist()
        x_cur = torch.zeros([N, 1], dtype=dtype, device=device)
        S_i_temp = ind_b[-K:]
        x_cur[S_i_temp] = b[ind_b[-K:]]
        x_cur[x_cur < 0] = 0    # hard threshold negative entries

        X_i = S_i_temp
        A_x_cur = A[:, X_i].mm(x_cur[X_i])
        res = y - A_x_cur
        der = A_t.mm(res)
        ider = der[X_i]
        Pder = A[:, X_i].mm(ider)
        mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
        x_cur[X_i] = x_cur[X_i] + mu_bar * ider
        x_cur[x_cur < 0] = 0    # hard threshold negative entries

        A_x_cur = A[:, X_i].mm(x_cur[X_i])
        res = y - A_x_cur
        if (i == 1):
            A_diff = A_x_cur
        else:
            A_diff = A_x_cur - A_x_prev
        temp = A_diff.T.mm(A_diff)
        if (temp > 0):
            tau = res.T.mm(A_diff) / temp
        else:
            tau = res.T.mm(A_diff) / 1e-6
        y_cur = x_cur + tau * (x_cur - x_prev)
        Y_i = y_cur.squeeze().nonzero().squeeze().tolist()

        # print out objective function value during optimization of IHT
        if (verbose and i % 50 == 1):
            obj_value = torch.norm(y - A.mm(x_cur))
            print('at iteration {}, the objective value is: {}'.format(i, obj_value))

        # stop criterion
        if (i > 1) and (torch.norm(x_cur - x_prev) < tol * torch.norm(x_cur)):
            break

    x = x_cur
    supp = x_cur.squeeze().nonzero().squeeze().tolist()     # support of the output solution
    obj_value = torch.norm(y - A.mm(x_cur))
    print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp), obj_value))
    return x, supp


