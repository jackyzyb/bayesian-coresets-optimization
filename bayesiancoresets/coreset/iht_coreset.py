import numpy as np
from .coreset import Coreset

"""
This file contains the two approaches, i.e., Automated Accelerated IHT and Automated Accelerated IHT II, 
proposed in Bayesian Coresets: An Optimization Perspective.
The two approaches are presented in IHTCoreset._iht() and IHTCoreset._iht_ii(), respectively. 
"""

class FiniteTangentSpace():
    def __init__(self, tangent_space_factory, d):
        vecs = tangent_space_factory()
        d = vecs.shape[1]  # log: no
        if len(vecs.shape) != 2:
            raise ValueError('._set_vecs(): vecs must be a 2d array, otherwise the expected behaviour is ambiguous')
        if vecs.shape[1] != d:
            raise ValueError('._set_vecs(): vecs must have the correct dimension')
        self.vecs = vecs
        print('ves shape:')
        print(vecs.shape)
        self.vsum = vecs.sum(axis=0)
        self.vsum_norm = np.sqrt((self.vsum ** 2).sum())
        self.vnorms = np.sqrt((self.vecs ** 2).sum(axis=1))
        self.vnorms_sum = self.vnorms.sum()

    def sum(self):
        return self.vsum

    def sum_w(self, w, idcs):
        return w.dot(self.vecs[idcs, :])

    def sum_w_norm(self, w, idcs):
        return np.sqrt(((w.dot(self.vecs[idcs, :])) ** 2).sum())

    def num_vectors(self):
        return self.vecs.shape[0]

    def norms(self):
        return self.vnorms

    def norms_sum(self):
        return self.vnorms_sum

    def sum_norm(self):
        return self.vsum_norm


class IHTCoreset(Coreset):
  """
  Same as other 'hilbert' methods, this class takes in a tangent space for random projection to finite space.
  """
    def __init__(self, tangent_space_factory, d, iht_mode='IHT', **kw):
        super().__init__(**kw)
        self.reached_numeric_limit = False
        self.iht_mode = iht_mode
        self.T = FiniteTangentSpace(tangent_space_factory, d)
        self.dim = self.T.vecs.shape[0]
        self.full_wts = np.zeros(self.dim)
        self.full_wts_scaled = np.zeros(self.dim)  # np.random.rand(self.dim)
        self.supp = []
        self.learning_rate = 1e-6
        self.scale = self.T.norms_sum() / self.T.norms()
        self.convergence_error = 0.0001
        self.MAX_ITER_NUM = 6000
        self.iter_iht = 0
        if np.any(self.T.norms() == 0):
            raise ValueError('.__init__(): tangent space must not have any 0 vectors')

    def _objective(self):
        v = self.full_wts_scaled * self.scale - 1
        return v.T.dot(self.T.matrixK.dot(v))

    def _objective_w(self, w):
        Phi = self.T.vecs.T
        y = self.T.vsum.reshape([-1, 1])
        return np.linalg.norm(y - Phi.dot(w), ord=2)

    # accelerated iht
    def _iht(self, K):
        # parameters setting, k is sparsity
        Phi = self.T.vecs.T
        y = self.T.vsum.reshape([-1, 1])
        # np.save('Phi.npy', Phi)
        # np.save('y.npy', y)
        PrintOutResult = True

        (M, N) = Phi.shape
        tol = 1e-5
        ALPSiters = 300

        # Initialize transpose of measurement matrix
        Phi_t = Phi.T

        # Initialize to zero vector
        x_cur = np.zeros([N, 1])
        y_cur = np.zeros([N, 1])
        # x_cur = np.random.random([N, 1])
        # y_cur = np.random.random([N, 1])

        Phi_x_cur = np.zeros([M, 1])
        Y_i = []

        # auxiliary variables
        complementary_Yi = np.ones([N, 1])
        i = 1
        obj_list = []

        while (i <= ALPSiters):
            x_prev = x_cur
            if (i == 1):
                res = y
                der = Phi_t.dot(res)
            else:
                res = y - Phi_x_cur - tau * Phi_diff
                der = Phi_t.dot(res)
            Phi_x_prev = Phi_x_cur
            complementary_Yi[Y_i] = 0
            ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
            complementary_Yi[Y_i] = 1
            S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()
            ider = der[S_i]
            Pder = Phi[:, S_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2
            b = y_cur + mu_bar * der
            ind_b = np.flip(np.argsort(np.absolute(np.squeeze(b))))
            ind_b = np.squeeze(ind_b).tolist()
            x_cur = np.zeros([N, 1])
            S_i_temp = ind_b[0:K]
            x_cur[S_i_temp] = b[ind_b[0:K]]
            X_i = S_i_temp

            # truncate negative entries
            x_cur[x_cur < 0] = 0

            Phi_x_cur = Phi[:, X_i].dot(x_cur[X_i])
            res = y - Phi_x_cur

            if (i == 1):
                Phi_diff = Phi_x_cur
            else:
                Phi_diff = Phi_x_cur - Phi_x_prev

            temp = Phi_diff.T.dot(Phi_diff)
            if (temp > 0):
                tau = res.T.dot(Phi_diff) / temp
            else:
                tau = res.T.dot(Phi_diff) / 1e-6

            y_cur = x_cur + tau * (x_cur - x_prev)
            Y_i = np.nonzero(y_cur)[0].tolist()

            # print out objective function value during optimization of IHT
            if (i % 50 == -1):
                print('after iteration {}:'.format(i))
                print('objective value: {}'.format(self._objective_w(x_cur)))
                print('  ')

            # for experiment 1 to record convergence
            # if K == 200:
            #  obj_list.append(self._objective_w(x_cur))

            # stop criterion
            if i > 1 and (np.linalg.norm(x_cur - x_prev) < tol * np.linalg.norm(x_cur)):
                break
            i = i + 1

        if PrintOutResult:
            print('sparsity level: {}'.format(K))
            print('objective value: {}'.format(self._objective_w(x_cur)))
            # print('support: {}'.format(np.nonzero(x_cur)[0].tolist()))
            # print('weight: {}'.format(x_cur[np.nonzero(x_cur)[0].tolist()]))
            print('  ')
        self.supp = np.nonzero(x_cur)[0].tolist()
        self._overwrite(np.squeeze(x_cur)[self.supp], self.supp)

        # for experiment 1 to record convergence
        # if K == 200:
        #  np.save('iht-convergence.npy', np.array(obj_list))

    # accelerated IHT II
    def _iht_ii(self, K):
        # parameters setting, k is sparsity
        Phi = self.T.vecs.T
        y = self.T.vsum.reshape([-1, 1])
        # np.save('Phi.npy', Phi)
        # np.save('y.npy', y)
        PrintOutResult = True

        (M, N) = Phi.shape
        tol = 1e-5
        ALPSiters = 300

        # Initialize transpose of measurement matrix
        Phi_t = Phi.T

        # Initialize to zero vector
        x_cur = np.zeros([N, 1])
        y_cur = np.zeros([N, 1])
        # x_cur = np.random.random([N, 1])
        # y_cur = np.random.random([N, 1])

        Phi_x_cur = np.zeros([M, 1])
        Y_i = []

        # auxiliary variables
        complementary_Yi = np.ones([N, 1])
        i = 1
        obj_list = []

        while (i <= ALPSiters):
            x_prev = x_cur
            if (i == 1):
                res = y
                der = Phi_t.dot(res)
            else:
                res = y - Phi_x_cur - tau * Phi_diff
                der = Phi_t.dot(res)

            Phi_x_prev = Phi_x_cur
            complementary_Yi[Y_i] = 0
            ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
            complementary_Yi[Y_i] = 1
            S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()
            ider = der[S_i]
            Pder = Phi[:, S_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2
            b = y_cur[S_i] + mu_bar * ider
            ind_b = np.flip(np.argsort(np.absolute(np.squeeze(b))))
            ind_b = np.squeeze(ind_b).tolist()
            x_cur = np.zeros([N, 1])
            S_i_temp = [S_i[j] for j in ind_b[0:K]]
            x_cur[S_i_temp] = b[ind_b[0:K]]
            X_i = S_i_temp
            Phi_x_cur = Phi[:, X_i].dot(x_cur[X_i])
            res = y - Phi_x_cur
            der = Phi_t.dot(res)
            ider = der[X_i]
            Pder = Phi[:, X_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2
            x_cur[X_i] = x_cur[X_i] + mu_bar * ider

            # hard threshold negative entries
            x_cur[x_cur < 0] = 0

            Phi_x_cur = Phi[:, X_i].dot(x_cur[X_i])
            res = y - Phi_x_cur

            if (i == 1):
                Phi_diff = Phi_x_cur
            else:
                Phi_diff = Phi_x_cur - Phi_x_prev

            temp = Phi_diff.T.dot(Phi_diff)
            if (temp > 0):
                tau = res.T.dot(Phi_diff) / temp
            else:
                tau = res.T.dot(Phi_diff) / 1e-6

            y_cur = x_cur + tau * (x_cur - x_prev)
            Y_i = np.nonzero(y_cur)[0].tolist()

            # print out objective function value during optimization of IHT
            if (i % 50 == -1):
                print('after iteration {}:'.format(i))
                print('objective value: {}'.format(self._objective_w(x_cur)))
                print('  ')

            # for experiment 1 to record convergence
            # if K == 200:
            #  obj_list.append(self._objective_w(x_cur))

            # stop criterion
            if (i > 1) and (np.linalg.norm(x_cur - x_prev) < tol * np.linalg.norm(x_cur)):
                break
            i = i + 1

        if PrintOutResult:
            print('after iteration {}:'.format(i))
            print('objective value: {}'.format(self._objective_w(x_cur)))
            print('  ')
        self.supp = np.nonzero(x_cur)[0].tolist()
        self._overwrite(np.squeeze(x_cur)[self.supp], self.supp)

        # for experiment 1 to record convergence
        # if K == 200:
        #  print('saving convergence at M=200')
        #  np.savez('obj.npz', Phi=Phi, y=y)
        #  np.save('iht-2-convergence.npy', np.array(obj_list))

    def reset(self):
        self.snnls.reset()
        super().reset()

    def _build(self, itrs, sz):
        if self.iht_mode == 'IHT':
            self._iht(sz)
        elif self.iht_mode == 'IHT-2':
            self._iht_ii(sz)
        else:
            raise ValueError('IHT mode error: should be IHT or IHT-2')
        # w = self.snnls.weights()
        # self._overwrite(w[w>0], np.where(w>0)[0])

    def _optimize(self):
        self.snnls.optimize()
        w = self.snnls.weights()
        self._overwrite(w[w > 0], np.where(w > 0)[0])

    def error(self):
        return self.snnls.error()
