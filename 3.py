import numpy as np
import numpy.linalg as ln
import scipy as sp


def wolfe_line_search(func, grad, xk, pk):
    return sp.optimize.line_search(func, grad, xk, pk)[0]


def dfp_call(f, grad, x0, epsi=0.001):
    # initial values
    k = 0
    gfk = grad(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0

    while ln.norm(gfk) > epsi:
        # pk - direction
        pk = -np.dot(Hk, gfk)

        line_search = sp.optimize.line_search(f, grad, xk, pk)
        alpha_k = line_search[0]

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = grad(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1

        k += 1

        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    return xk, f(xk), k


def f(x):
    res = 0
    for i in range(1, len(x) // 2 + 1):
        res += x[2 * i - 1 - 1] ** 2 + 10 * (x[2 * i - 1 - 1] ** 2 + x[2 * i - 1] ** 2 - 1) ** 2
    return res


def grad(x):
    vec = [0] * len(x)
    for i in range(1, len(x) // 2 + 1):
        vec[2*i - 1 - 1] += 2 * x[2 * i - 1 - 1]
        vec[2*i - 1 - 1] += 10 * (x[2 * i - 1 - 1] ** 2 + x[2 * i - 1] ** 2 - 1) * 2 * x[2 * i - 1 - 1]
        vec[2 * i - 1] += 10 * (x[2 * i - 1 - 1] ** 2 + x[2 * i - 1] ** 2 - 1) * 2 * x[2 * i - 1]
    return np.array(vec)


xk, fk, k = dfp_call(f, grad, np.array([1.1, 0.1] * 5))
print("xk: ", xk)
print("f(xk): ", fk)
print("k: ", k)

# Prints:
"""
xk:  [-4.12934185e-05 -9.99996787e-01 -3.91705056e-05 -9.99997689e-01
 -5.56451382e-05 -9.99996247e-01 -5.00014764e-05 -9.99995493e-01
  1.38094833e-04 -1.00000634e+00]
f(xk):  3.1522109415938316e-08
k:  18
"""
