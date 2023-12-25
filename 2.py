from math import sin, cos
import numpy as np
import numpy.linalg as ln


def goldstein_line_search(func, grad, x_k, d, max_alpha=1, epsi=0.001, t=2):
    phi_0 = func(x_k)
    dphi_0 = np.dot(grad(x_k), d)
    a = 0
    b = max_alpha
    k = 0
    np.random.seed(69)
    alpha = np.random.rand() * max_alpha
    max_iter = 100
    while k < max_iter:
        phi = func(x_k + d * alpha)
        if phi_0 + epsi * alpha * dphi_0 >= phi:
            if phi_0 + (1 - epsi) * alpha * dphi_0 <= phi:
                break
            else:
                a = alpha
                if b >= max_alpha:
                    alpha = t * alpha
                    k += 1
                    continue
        else:
            b = alpha
        alpha = 0.5*(a + b)
        k += 1
    return alpha


def polak_ribiere(func, grad, x0, epsilon=0.01):
    k = 0
    pk = -grad(x0)
    xk = x0
    while np.linalg.norm(grad(xk)) > epsilon:
        alpha = goldstein_line_search(func, grad, xk, pk)
        xk1 = xk + alpha * pk

        beta = np.dot(grad(xk1).transpose(), (grad(xk1) - grad(xk))) / ln.norm(grad(xk)) ** 2
        pk = -grad(xk1) + beta * pk

        xk = xk1
        k += 1
    return xk, func(xk), k


def f(x):
    res = 0
    for i in range(len(x) - 1):
        res += (x[i] ** 2 - sin(x[i])) ** 2
    tmp = 0
    for i in range(len(x)):
        tmp += x[i] ** 2
    res += (tmp - 100) ** 2
    return res


def grad(x):
    vec = [0] * len(x)
    for i in range(len(x) - 1):
        vec[i] += (x[i] ** 2 - sin(x[i])) * (2 * x[i] - cos(x[i]))
    tmp = 0
    for i in range(len(x)):
        tmp += x[i] ** 2
    for i in range(len(x)):
        vec[i] += 2 * (tmp - 100) * 2 * x[i]
    return np.array(vec)


xk, fk, k = polak_ribiere(f, grad, np.array([2] * 10))
print("xk: ", xk)
print("f(xk): ", fk)
print("k: ", k)

# Prints:
"""
xk:  [0.87669595 0.87669595 0.87669595 0.87669595 0.87669595 0.87669595
 0.87669595 0.87669595 0.87669595 9.64794668]
f(xk):  6.650478102499786e-08
k:  320
"""