
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import expm
from itertools import combinations_with_replacement
from functools import lru_cache
from math import factorial

# Shared helpers live in kalman_common
from kalman_common import (
    TAU_REF_DEFAULT,
    build_basis,
    _basis_cache,
    _multi_indices,
    build_H,
    build_dH,
    precompute_R,
    _precompute_Mp,
    build_seasonality_matrix,
    seasonality_bic,
)


# ---------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------

@dataclass
class LinearDiffusionParams:
    theta:   np.ndarray                       # (m,)
    mu:      np.ndarray                       # (m,), only mu[0] free
    lam:     np.ndarray
    c:       np.ndarray                       # (m,)
    d:       np.ndarray                       # (m,)
    rho:     np.ndarray                       # (m, m)
    p_delta: float
    p_beta:  float
    p_e:     float                            #noise scalar:
                                              # Var[v_t]_ii = p_e^2 * (tau_i / tau_ref)
    p_gamma: float = 0.0                      # degree-5 poly coefficient
    p_K:     float = 0.0                      #shift param


    #independent-poly
    p_beta_arr:  Optional[np.ndarray] = None  # (m,)
    p_gamma_arr: Optional[np.ndarray] = None  # (m,)
    p_K_arr:     Optional[np.ndarray] = None  # (m,)
    independent_poly: bool = False



    @property
    def a(self):
        return -self.theta

    @property
    def b(self):
        return self.theta * self.mu


# ---------------------------------------------------------------
# Graded-lex basis and polynomial helpers
# ---------------------------------------------------------------

def build_poly_1d(params, N):
    # 1D polynomial-map coefficients in standard basis 
    #
    # For N <= 3 phi_3(x) = m + x + b^2 x^3,
    #
    # For N == 5
    #phi_5(x) = a^2 x^5 + b^2 [(x-K)^3 + K^3] + x + m
    coefs = np.zeros(N + 1)
    coefs[0] = params.p_delta
    if N >= 1:
        coefs[1] = 1.0
    if N >= 3:
        coefs[3] = params.p_beta ** 2
    if N >= 5:
        b2 = params.p_beta ** 2
        K  = params.p_K
        coefs[1] += 3.0 * K * K * b2
        coefs[2]  = -3.0 * K * b2
        coefs[5]  = params.p_gamma ** 2
    return coefs


def build_poly_nd(params, m, N):
    # Multi-D polynomial coefficient vector aligned with `build_basis(m, N)`.
    basis, _ = build_basis(m, N)
    p = np.zeros(len(basis))

    if not getattr(params, "independent_poly", False):
        poly_1d = build_poly_1d(params, N)
        for k, alpha in enumerate(basis):
            n = sum(alpha)
            if n > N or poly_1d[n] == 0.0:
                continue
            multi = factorial(n)
            for ai in alpha:
                multi //= factorial(ai)
            p[k] = poly_1d[n] * multi
        return p


    b_arr = np.asarray(params.p_beta_arr, dtype=float)
    a_arr = (np.asarray(params.p_gamma_arr, dtype=float) if N >= 5
             else np.zeros(m))
    K_arr = (np.asarray(params.p_K_arr,    dtype=float) if N >= 5
             else np.zeros(m))

    for k, alpha in enumerate(basis):
        n = sum(alpha)
        if n == 0:
            # Single shared constant.
            p[k] = params.p_delta
            continue
        nonzero = [i for i, a in enumerate(alpha) if a > 0]
        if len(nonzero) != 1:
            continue
        i = nonzero[0]
        deg = alpha[i]
        b2  = b_arr[i] ** 2
        if deg == 1:
            p[k] = 1.0 + (3.0 * K_arr[i] * K_arr[i] * b2 if N >= 5 else 0.0)
        elif deg == 2 and N >= 5:
            p[k] = -3.0 * K_arr[i] * b2
        elif deg == 3 and N >= 3:
            p[k] = b2
        elif deg == 5 and N >= 5:
            p[k] = a_arr[i] ** 2
    return p


def infinitesimal_generator(a, b, c, d, rho, N):
    m = len(a)
    basis, idx = build_basis(m, N)
    dim = len(basis)
    G = np.zeros((dim, dim))

    for k, beta in enumerate(basis):
        beta = list(beta)
        diag = 0.0
        for i in range(m):
            diag += a[i] * beta[i]
            diag += 0.5 * d[i] ** 2 * beta[i] * (beta[i] - 1)
        for i in range(m):
            for j in range(i + 1, m):
                diag += rho[i, j] * d[i] * d[j] * beta[i] * beta[j]
        G[k, k] = diag

        for i in range(m):
            alpha = beta.copy()
            alpha[i] += 1
            if tuple(alpha) in idx:
                val  = (beta[i] + 1) * b[i]
                val += (beta[i] + 1) * c[i] * d[i] * beta[i]
                for j in range(m):
                    if j != i:
                        val += (beta[i] + 1) * c[i] * rho[i, j] * d[j] * beta[j]
                G[k, idx[tuple(alpha)]] += val

        for i in range(m):
            alpha = beta.copy()
            alpha[i] += 2
            if tuple(alpha) in idx:
                G[k, idx[tuple(alpha)]] += 0.5 * c[i] ** 2 * (beta[i] + 2) * (beta[i] + 1)

        for i in range(m):
            for j in range(i + 1, m):
                alpha = beta.copy()
                alpha[i] += 1
                alpha[j] += 1
                if tuple(alpha) in idx:
                    G[k, idx[tuple(alpha)]] += (rho[i, j] * c[i] * c[j] *
                                                 (beta[i] + 1) * (beta[j] + 1))

    return G


# ---------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------

def _mu_P(params):
    return params.mu + params.lam / params.theta


def stationary_cov(params):
    # Closed-form stationary covariance of the OU state with diagonal
    theta = np.asarray(params.theta, dtype=float)
    c     = np.asarray(params.c,dtype=float)
    rho   = np.asarray(params.rho,dtype=float)
    return rho * np.outer(c, c) / (theta[:, None] + theta[None, :])


def f_OU(params, x, dt):
    theta, c = params.theta, params.c
    mu_P  = _mu_P(params)
    alpha = np.exp(-theta * dt)
    x_nxt = alpha * x.flatten() + mu_P * (1 - alpha)
    m = len(theta)
    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Q[i, j] = (params.rho[i, j] * c[i] * c[j] *
                       (1.0 - np.exp(-(theta[i] + theta[j]) * dt)) /
                       (theta[i] + theta[j]))
    return x_nxt.reshape(-1, 1), Q


def A_OU(params, x, dt):
    return np.diag(np.exp(-params.theta * dt))


# ---------------------------------------------------------------
# Correlation parameterisation
# ---------------------------------------------------------------

def rho_from_chol(v, m):

    if m == 1:
        return np.array([[1.0]])
    L = np.zeros((m, m))
    idx = 0
    for i in range(m):
        for j in range(i):
            L[i, j] = v[idx]; idx += 1
        ss = 1.0 - np.sum(L[i, :i] ** 2)
        L[i, i] = np.sqrt(max(ss, 0.0))
    norms = np.linalg.norm(L, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    L = L / norms
    return L @ L.T


def vec_from_chol(rho):
    # Inverse of `rho_from_chol
    m = rho.shape[0]
    if m == 1:
        return np.array([])
    rho_jit = np.asarray(rho, dtype=float) + 1e-10 * np.eye(m)
    try:
        L = np.linalg.cholesky(rho_jit)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(rho_jit)
        L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    norms = np.linalg.norm(L, axis=1, keepdims=True)
    L = L / np.maximum(norms, 1e-12)
    return np.array([L[i, j] for i in range(m) for j in range(i)])


# ---------------------------------------------------------------
# Pack / unpack / bounds
# ---------------------------------------------------------------

def pack_ld(params, N_poly=3, fit_d=False):
    # Pack a LinearDiffusionParams into a flat vector.
    parts = [
        np.asarray(params.theta, float),
        np.array([params.mu[0]]),
        # factor = mu + lam / theta.
        np.asarray(params.lam, float),
        np.asarray(params.c, float),
    ]
    if fit_d:
        parts.append(np.asarray(params.d, float))
    parts += [
        vec_from_chol(params.rho),
        np.array([params.p_delta]),
    ]
    if getattr(params, "independent_poly", False):
        m_ld = len(np.asarray(params.theta).ravel())
        parts.append(np.asarray(params.p_beta_arr, float).reshape(-1))
        if N_poly >= 5:
            parts.append(np.asarray(params.p_gamma_arr, float).reshape(-1))
            parts.append(np.asarray(params.p_K_arr,     float).reshape(-1))
    else:
        parts.append(np.array([params.p_beta]))
        if N_poly >= 5:
            # Quintic: pack p_gamma then p_K
            parts.append(np.array([params.p_gamma]))
            parts.append(np.array([params.p_K]))
    parts.append(np.array([float(params.p_e)]))
    return np.concatenate(parts)


def unpack_ld(vec, m, N_poly=3, fit_d=False, independent_poly=False):
    vec = np.asarray(vec, dtype=float)
    i = 0
    theta = vec[i:i + m];                      i += m
    mu = np.zeros(m); mu[0] = vec[i];          i += 1
    lam = vec[i:i + m].copy();                 i += m
    c   = vec[i:i + m];                        i += m
    if fit_d:
        d = vec[i:i + m];                      i += m
    else:
        d = np.zeros(m)
    n_rho = m * (m - 1) // 2
    rho = rho_from_chol(vec[i:i + n_rho], m);  i += n_rho
    p_delta = vec[i];                          i += 1
    p_beta  = 0.0
    p_gamma = 0.0
    p_K     = 0.0
    p_beta_arr  = None
    p_gamma_arr = None
    p_K_arr     = None
    if independent_poly:
        p_beta_arr = vec[i:i + m].copy();       i += m
        if N_poly >= 5:
            p_gamma_arr = vec[i:i + m].copy();  i += m
            p_K_arr     = vec[i:i + m].copy();  i += m
    else:
        p_beta = vec[i];                        i += 1
        if N_poly >= 5:
            p_gamma = vec[i];                   i += 1
            p_K     = vec[i];                   i += 1
    p_e = float(vec[i]);                       i += 1
    return LinearDiffusionParams(
        theta=theta, mu=mu, lam=lam, c=c, d=d, rho=rho,
        p_delta=p_delta, p_beta=p_beta, p_gamma=p_gamma, p_K=p_K,
        p_e=p_e,
        p_beta_arr=p_beta_arr, p_gamma_arr=p_gamma_arr, p_K_arr=p_K_arr,
        independent_poly=independent_poly,
    )


def num_params_ld(m, N_poly, fit_d=False, independent_poly=False):
    k = (m                    # theta
         + 1                  # mu[0]
         + m                  # lam[0..m-1]
         + m                  # c
         + m * (m - 1) // 2   # rho
         + 1                  # p_delta
         + 1)                 # p_e
    if independent_poly:
        k += m                # p_beta_arr
        if N_poly >= 5:
            k += 2 * m        # p_gamma_arr + p_K_arr
    else:
        k += 1                # p_beta
        if N_poly >= 5:
            k += 2            # p_gamma + p_K (Sun 2024 eq. 3.3 cubic-shift)
    if fit_d:
        k += m                # d
    return k


def _theta_bands(m):
    # Per-factor OU mean-reversion bands.
    if m == 1:
        return [(0.1, 30.0)]
    if m == 2:
        return [(0.1, 1.0), (0.1, 60.0)]
    if m == 3:
        return [(0.1, 50.0), (0.1, 50.0), (0.1, 90.0)]
    if m == 4:
        return [(0.1, 50.0), (0.1, 50.0), (0.1, 50.0),(0.1,50.0)]


def _make_bounds_dynamics_block(m, fit_d):
    # Shared bounds
    n_rho = m * (m - 1) // 2
    head  = list(_theta_bands(m))          # theta
    head += [(-0.5, 3)]                  # mu[0] (residuals are zero-mean)
    head += [(-0.3, 1)] * m              # lam[0..m-1] (drift adjustment)
    head += [(0.01, 8)] * m            # c (vol on normalised scale)
    if fit_d:
        head += [(-1.0, 1.0)] * m        # d (drift adjustment)
    head += [(-0.95, 0.95)] * n_rho        # rho cholesky entries
    tail = [(1e-6, 1)]                     # p_e (R_ii = p_e^2 * tau)
    return head, tail


def make_bounds_shared(m, N_poly, fit_d=False):
    # Bounds for the SHARED
    head, tail = _make_bounds_dynamics_block(m, fit_d)

    if N_poly == 1:
        poly  = [(0.0, 0.0)]               # p_delta — pinned
        poly += [(0.0, 0.0)]               # p_beta  — pinned
    else:
        poly  = [(-0.5, 0.5)]              # p_delta
        poly += [(0.0, 9)]                # p_beta
        if N_poly >= 5:
            poly += [(0.00, 9)]           # p_gamma
            poly += [(-2, 2)]          # p_K 
    bounds = head + poly + tail
    expected = num_params_ld(m, N_poly, fit_d=fit_d, independent_poly=False)
    assert len(bounds) == expected, (
        f"make_bounds_shared produced {len(bounds)} bounds but num_params_ld "
        f"expects {expected} (m={m}, N_poly={N_poly}, fit_d={fit_d})"
    )
    return bounds


def make_bounds_independent(m, N_poly, fit_d=False):
    # Bounds for the INDEPENDENT 
    head, tail = _make_bounds_dynamics_block(m, fit_d)
    if N_poly == 1:
        poly  = [(0.0, 0.0)]               # p_delta — pinned
        poly += [(0.0, 0.0)] * m           # p_beta_arr — pinned
    else:
        poly  = [(-2.0, 2.0)]              # p_delta
        poly += [(0.001, 50)] * m          # p_beta
        if N_poly >= 5:
            poly += [(0.001, 0.4)] * m     # p_gamma_arr
            poly += [(-1.0, 1.0)] * m      # p_K
    bounds = head + poly + tail
    expected = num_params_ld(m, N_poly, fit_d=fit_d, independent_poly=True)
    assert len(bounds) == expected, (
        f"make_bounds_independent produced {len(bounds)} bounds but "
        f"num_params_ld expects {expected} (m={m}, N_poly={N_poly}, "
        f"fit_d={fit_d})"
    )
    return bounds


def make_bounds(m, N_poly, fit_d=False, independent_poly=False):
    if independent_poly:
        return make_bounds_independent(m, N_poly, fit_d=fit_d)
    return make_bounds_shared(m, N_poly, fit_d=fit_d)


# ---------------------------------------------------------------
# EKF step / run / MLE
# ---------------------------------------------------------------

def EKF_step(params, x, y, f, A, P, h, tau, delta, t_idx, dt, N, R_diag):
    x_prior, Q = f(params, x, dt)
    A_Jac   = A(params, x, dt)
    P_prior = A_Jac @ P @ A_Jac.T + Q

    y_pred, H_Jac = h(params, x_prior, tau, delta, N, t_idx)
    H_Jac = np.asarray(H_Jac)
    if H_Jac.ndim == 1:
        H_Jac = H_Jac.reshape(-1, 1)

    R_mat = np.diag(np.asarray(R_diag, dtype=float))
    S = H_Jac @ P_prior @ H_Jac.T + R_mat
    if not np.all(np.isfinite(S)):
        raise ValueError("Non-finite S")

    K = P_prior @ H_Jac.T @ np.linalg.inv(S)
    resid = (y - y_pred).flatten()
    x_post = x_prior + (K @ resid).reshape(-1, 1)
    I_KH = np.eye(len(x_prior)) - K @ H_Jac
    P_post = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
    return x_post, P_post, resid, S


def EKF_run(params, x0, y_obs, f, A, P0, h, tau, delta, dt, N, R_all):
    n_steps = len(y_obs)
    x = np.asarray(x0, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    P = np.atleast_2d(P0)
    log_lik     = 0.0
    n_obs_total = 0
    for t_idx in range(n_steps):
        try:
            x, P, resid, S = EKF_step(params, x, y_obs[t_idx], f, A, P, h,
                                        tau[t_idx], delta[t_idx], t_idx, dt, N,
                                        R_diag=R_all[t_idx])
        except (np.linalg.LinAlgError, ValueError):
            return -1e10, 0
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(log_det):
            return -1e10, 0
        n_obs_t     = len(resid)
        n_obs_total += n_obs_t
        log_lik    += -0.5 * (n_obs_t * np.log(2 * np.pi) + log_det
                              + resid @ np.linalg.solve(S, resid))
    return log_lik, n_obs_total


def EKF_MLE(params_vec, y_obs, T, delta, dt, N, m,
             tau_ref=TAU_REF_DEFAULT,
             fit_d=False,
             independent_poly=False):
    try:
        params = unpack_ld(params_vec, m, N_poly=N,
                             fit_d=fit_d,
                             independent_poly=independent_poly)
        x0 = _mu_P(params).reshape(-1, 1)

        P0 = stationary_cov(params)
        if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(P0))):
            return 1e10
    except Exception:
        return 1e10

    shift_cap = 3.0
    shift     = np.abs(np.asarray(x0).flatten() - params.mu)
    if independent_poly:
        cap_val = float(shift.max())
    else:
        cap_val = float(abs(np.asarray(x0).flatten().sum()
                            - np.asarray(params.mu).sum()))
    if cap_val > shift_cap:
        excess = cap_val - shift_cap
        return 1e8 + 1e4 * excess ** 2

    p_T = build_poly_nd(params, m, N)
    G   = infinitesimal_generator(params.a, params.b,
                                    params.c, params.d,
                                    params.rho, N)

    try:
        Mp_all, _ = _precompute_Mp(G, p_T, T, delta)
        R_all     = precompute_R(T, params.p_e, tau_ref=tau_ref)
    except Exception:
        return 1e10

    def h_timedep(_params, x, _tau, _delta, _N, t_idx):
        x_vec = x.flatten()
        H_x  = build_H(x_vec,  N)
        dH_x = build_dH(x_vec, N)
        h_vals = np.array([H_x  @ Mp for Mp in Mp_all[t_idx]])
        H_jac  = np.array([dH_x @ Mp for Mp in Mp_all[t_idx]])
        return h_vals, H_jac

    log_lik, _ = EKF_run(params, x0, y_obs, f_OU, A_OU, P0,
                           h_timedep, T, delta, dt, N, R_all)
    return -log_lik


