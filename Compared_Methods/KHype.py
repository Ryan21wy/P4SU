"""
KHype Python implementation
---------------------------

This implementation is adapted from the MATLAB code provided by Jie Chen et al.
Original MATLAB code and examples are available at:
    http://www.cedric-richard.fr/Matlab/chen2013nonlinear.zip

Chen, J., Richard, C., & Bermond, A. (2013). Nonlinear Hyperspectral Unmixing
using Kernel Methods. *IEEE Transactions on Geoscience and Remote Sensing*, 51(6),
3685–3695. doi:10.1109/TGRS.2012.2223208

Accessed: May 28, 2025.
"""

import numpy as np

from cvxopt import matrix, solvers

from joblib import Parallel, delayed

# Python implementation of KHype algorithm
# Inputs:
# r:      (L, N) data matrix (pixels/spectra)
# M:      (L, P) endmember matrix
# mu:      regularization constant


def KHype(r, M, mu, kernel='gaussian', par=2, n_jobs=-1):
    L, N = r.shape
    _, P = M.shape

    if kernel == 'gaussian':
        Q = np.eye(P) / (par ** 2)
        MQM = M @ Q @ M.T
        dMQM = np.diag(MQM).reshape(-1, 1)
        D1 = dMQM @ np.ones((1, L))
        D2 = np.ones((L, 1)) @ dMQM.T
        KM = np.exp(-0.5 * (D1 + D2 - 2 * MQM))
    else:
        # default: polynomial kernel
        KM = (1 + (1.0 / P ** 2) * (M - 0.5) @ (M - 0.5).T) ** 2

    # Precompute terms
    M1 = M @ np.ones((P, 1))
    a_est = np.zeros((P, N))
    MM = M @ M.T

    K_block = MM + KM + mu * np.eye(L)

    top = np.hstack([K_block, M, -M1])
    mid = np.hstack([M.T, np.eye(P), -np.ones((P, 1))])
    bot = np.hstack([-M1.T, -np.ones((1, P)), np.array([[P]])])
    H = np.vstack([top, mid, bot])
    # Symmetrize and regularize
    H = (H + H.T) / 2
    H += 0.0 * np.eye(L + P + 1)

    H_cvx = matrix(H)

    beta_khype = np.zeros((L, N))

    def solve_pixel(n):
        solvers.options['show_progress'] = False

        y = r[:, n].reshape(-1, 1)

        # f = -[y; zeros(P,1); -1]
        f_vec = np.vstack([-y, np.zeros((P, 1)), [[1.0]]])
        f_cvx = matrix(f_vec)

        A = np.hstack([np.zeros((P, L)), np.eye(P), np.zeros((P, 1))])
        A_cvx = matrix(-A)
        b_cvx = matrix(np.zeros((P, 1)))

        sol = solvers.qp(H_cvx, f_cvx, A_cvx, b_cvx)
        z = np.array(sol['x'])

        beta = z[:L]
        gam = z[L:L + P]
        lam = z[-1]

        h = M.T @ beta + gam - lam
        return h.flatten()

    results = Parallel(n_jobs=n_jobs)(delayed(solve_pixel)(n) for n in range(N))
    a_est = np.column_stack(results)
    return a_est


def SKHype(r, M, mu, kernel='gaussian', par=2.0, max_iter=15, tol=1e-4, n_jobs=-1):
    """
    r: (L, N) data matrix
    M: (L, R) endmember matrix (R endmembers)
    mu: regularization constant
    kernel: Gaussian/polynomial kernel
    par: Gaussian kernel bandwidth parameter
    max_iter: maximum inner iterations for mixing weights
    tol: convergence tolerance on mixing weights
    n_jobs: number of parallel jobs
    Returns: a_est (R x N)
    """
    L, N = r.shape
    _, R = M.shape

    MM = M @ M.T

    if kernel == 'gaussian':
        Q = np.eye(R) / (par ** 2)
        MQM = M @ Q @ M.T
        dMQM = np.diag(MQM).reshape(-1, 1)
        D1 = dMQM @ np.ones((1, L))
        D2 = np.ones((L, 1)) @ dMQM.T
        KM = np.exp(-0.5 * (D1 + D2 - 2 * MQM))
    else:
        # default: polynomial kernel
        KM = (1 + (1.0 / R ** 2) * (M - 0.5) @ (M - 0.5).T) ** 2

    # for n in tqdm(range(N)):
    def solve_pixel(n):
        y = r[:, n].reshape(-1, 1)
        # initialize mixing weights
        d = np.array([0.5, 0.5])
        dp = np.zeros_like(d)

        # constant parts
        f_base = -np.vstack([y, np.zeros((R, 1))])  # length L+R
        A = np.hstack([np.zeros((R, L)), np.eye(R)])
        A_cvx = matrix(-A)
        b_cvx = matrix(np.zeros((R, 1)))
        H = np.zeros((L + R, L + R))

        for iter in range(max_iter):
            # build current kernel combination
            K = d[0] * MM + d[1] * KM
            Kt = K + mu * np.eye(L)
            # assemble H
            H[:L, :L] = Kt
            H[:L, L:] = d[0] * M
            H[L:, :L] = d[0] * M.T
            H[L:, L:] = d[0] * np.eye(R)

            # top = np.hstack([Kt, d[0] * M])
            # bot = np.hstack([d[0] * M.T, d[0] * np.eye(R)])
            # H = np.vstack([top, bot])

            H = (H + H.T) / 2 + 0e-5 * np.eye(L + R)
            H_cvx = matrix(H)

            # solve QP: minimize 1/2 z'Hz + f^T z
            solvers.options['show_progress'] = False
            sol = solvers.qp(H_cvx, matrix(f_base), A_cvx, b_cvx)
            z0 = np.array(sol['x']).flatten()
            z = z0[:L]
            gam = z0[L:]

            # compute ht = (M'.beta + gam) * d[0]
            ht = (M.T @ z + gam) * d[0]

            # update mixing weights
            num = (d[1]**2) * (z.T @ (KM @ z))
            den = (ht.T @ ht)
            d_new0 = 1.0 / (1 + np.sqrt(num / den))
            d_new = np.array([d_new0, 1 - d_new0])

            # check convergence
            if iter > 0 and np.linalg.norm(d_new - dp) < tol:
                break
            dp = d.copy()
            d = d_new
        return ht

    # Parallel solve for all pixels
    results = Parallel(n_jobs=n_jobs)(delayed(solve_pixel)(n) for n in range(N))
    a_est = np.column_stack(results)
    return a_est


def SKHype_L1(r, M, mu, kernel='gaussian', par=2.0, lambda_reg=0., max_iter=15, tol=1e-4, n_jobs=-1):
    """
    r: (L, N) data matrix
    M: (L, R) endmember matrix (R endmembers)
    mu: regularization constant
    kernel: Gaussian/polynomial kernel
    par: Gaussian kernel bandwidth parameter
    lambda_reg: l1-norm regularization constant
    max_iter: maximum inner iterations for mixing weights
    tol: convergence tolerance on mixing weights
    n_jobs: number of parallel jobs
    Returns: a_est (R x N)
    """
    L, N = r.shape
    _, R = M.shape

    MM = M @ M.T

    if kernel == 'gaussian':
        Q = np.eye(R) / (par ** 2)
        MQM = M @ Q @ M.T
        dMQM = np.diag(MQM).reshape(-1, 1)
        D1 = dMQM @ np.ones((1, L))
        D2 = np.ones((L, 1)) @ dMQM.T
        KM = np.exp(-0.5 * (D1 + D2 - 2 * MQM))
    else:
        # default: polynomial kernel
        KM = (1 + (1.0 / R ** 2) * (M - 0.5) @ (M - 0.5).T) ** 2

    # results = []
    # for n in tqdm(range(N)):
    def solve_pixel(n):
        y = r[:, n].reshape(-1, 1)
        # initialize mixing weights
        d = np.array([0.5, 0.5])
        dp = np.zeros_like(d)

        # constant parts
        A1 = np.hstack([np.zeros((R, L)), np.eye(R)])  # γ ≥ 0
        A2 = np.hstack([M.T, np.eye(R)])  # M^T z + γ ≥ 0
        A = np.vstack([A1, A2])
        A_cvx = matrix(-A)
        b_cvx = matrix(np.zeros((2 * R, 1)))
        H = np.zeros((L + R, L + R))

        for iter in range(max_iter):
            # build current kernel combination
            K = d[0] * MM + d[1] * KM
            Kt = K + mu * np.eye(L)
            # assemble H
            H[:L, :L] = Kt
            H[:L, L:] = d[0] * M
            H[L:, :L] = d[0] * M.T
            H[L:, L:] = d[0] * np.eye(R)

            H = (H + H.T) / 2 + 0e-5 * np.eye(L + R)
            H_cvx = matrix(H)

            # After updating d, build the linear term f = -[y;0] + L1_reg_on_h
            # For h = d0*(M^T z + gamma), L1(h) = lambda * d0 * (1^T M^T z + 1^T gamma)
            # => linear in z: lambda*d0*M*1_R ; in gamma: lambda*d0*1_R
            vec_z_reg = M @ np.ones((R, 1))  # size Lx1
            vec_g_reg = np.ones((R, 1))  # size Rx1
            reg_vec = lambda_reg * d[0] * np.vstack([vec_z_reg, vec_g_reg])

            # Base linear term: -[y; zeros]
            f_base = -np.vstack([y, np.zeros((R, 1))]) + reg_vec

            # solve QP: minimize 1/2 z'Hz + f^T z
            solvers.options['show_progress'] = False
            sol = solvers.qp(H_cvx, matrix(f_base), A_cvx, b_cvx)
            z0 = np.array(sol['x']).flatten()
            z = z0[:L]
            gam = z0[L:]

            # compute ht = (M'.beta + gam) * d[0]
            ht = (M.T @ z + gam) * d[0]

            # update mixing weights
            num = (d[1] ** 2) * (z.T @ (KM @ z))
            den = (ht.T @ ht)
            d_new0 = 1.0 / (1 + np.sqrt(num / den))
            d_new = np.array([d_new0, 1 - d_new0])

            # check convergence
            if iter > 0 and np.linalg.norm(d_new - dp) / R < tol:
                d = d_new
                break
            dp = d.copy()
            d = d_new
        return ht

    # Parallel solve for all pixels
    results = Parallel(n_jobs=n_jobs)(delayed(solve_pixel)(n) for n in range(N))
    a_est = np.column_stack(results)
    return a_est