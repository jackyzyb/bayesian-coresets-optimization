from .log import set_verbosity  # , set_repeat
from .opt import nn_opt

TOL = 1e-12


def set_tolerance(tol):
    global TOL
    TOL = tol
