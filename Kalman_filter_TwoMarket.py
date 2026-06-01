# Two-market polynomial Kalman filter 

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.linalg import expm
from itertools import combinations_with_replacement
from functools import lru_cache
from math import factorial

# Shared helpers live in kalman_common;
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
# Constants
# ---------------------------------------------------------------

AB_MARGIN = 0.02      # Jacobi feasibility margin (a, b >= 1 + ab_margin)


PIN_THETA_R = 0.0


# ---------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------

@dataclass
class TwoMarketParams:
    kappa_Z:  np.ndarray
    theta_Z:  np.ndarray
    sigma_Z:  np.ndarray
    lam_Z:    np.ndarray
    # Market 2 (Y) OU factors
    kappa_Y:  np.ndarray
    sigma_Y:  np.ndarray
    lam_Y:    np.ndarray
    # Jacobi correlation factor R on (-1, 1)
    kappa_R:  float
    theta_R:  float                 # Q-mean of R (in (-1, 1))
    sigma_R:  float
    lam_R:    float = 0.0           # P-shift on R's mean
    # Per-market spot polynomial coefficients
    p_delta_1: float = 0.0
    p_beta_1:  float = 0.05
    p_gamma_1: float = 0.0
    p_K_1:     float = 0.0          # cubic-shift, market 1 (Sun eq. 3.3); active when N_poly>=5
    p_delta_2: float = 0.0
    p_beta_2:  float = 0.05
    p_gamma_2: float = 0.0
    p_K_2:     float = 0.0          # cubic-shift, market 2 (Sun eq. 3.3); active when N_poly>=5
    p_e_1:     float = 0.03
    p_e_2:     float = 0.03

    # ---- helpers ----
    @property
    def m_per_market(self) -> int:
        return len(self.kappa_Z)

    @property
    def n_state(self) -> int:
        return 2 * self.m_per_market + 1

    @property
    def Z_indices(self) -> np.ndarray:
        return np.arange(self.m_per_market)

    @property
    def Y_indices(self) -> np.ndarray:
        return np.arange(self.m_per_market, 2 * self.m_per_market)

    @property
    def R_index(self) -> int:
        return 2 * self.m_per_market


# ---------------------------------------------------------------
# Graded-lex basis and polynomial helpers
# ---------------------------------------------------------------

def build_poly_1d(params, market: int, N: int) -> np.ndarray:
    # 1D polynomial-map coefficients in standard basis [1, Z, Z^2, ..., Z^N]
    # for one market's spot.
    #
    # For N <= 3 
    #     Phi(Z) = p_delta + Z + p_beta^2 Z^3
    # For N == 5 
    #     Phi(Z) = p_delta + (1 + 3 K^2 b^2) Z  -  3 K b^2 Z^2 + b^2 Z^3  +  a^2 Z^5
    #                       
    if market == 1:
        p_delta, p_beta, p_gamma, p_K = (params.p_delta_1, params.p_beta_1,
                                          params.p_gamma_1, params.p_K_1)
    else:
        p_delta, p_beta, p_gamma, p_K = (params.p_delta_2, params.p_beta_2,
                                          params.p_gamma_2, params.p_K_2)
    coefs = np.zeros(N + 1)
    coefs[0] = p_delta
    if N >= 1:
        coefs[1] = 1.0
    if N >= 3:
        coefs[3] = p_beta ** 2
    if N >= 5:
        b2 = p_beta ** 2
        K  = p_K
        coefs[1] += 3.0 * K * K * b2
        coefs[2]  = -3.0 * K * b2
        coefs[5]  = p_gamma ** 2
    return coefs


def build_poly_market(params: TwoMarketParams, market: int, N: int) -> np.ndarray:

    n         = params.n_state
    basis, _  = build_basis(n, N)

    if market == 1:
        market_idx = params.Z_indices
    else:
        market_idx = params.Y_indices
    poly_1d = build_poly_1d(params, market, N)

    market_idx_set = {int(i) for i in market_idx}

    p = np.zeros(len(basis))
    for k, alpha in enumerate(basis):
        n_total = sum(alpha)
        if n_total > N or poly_1d[n_total] == 0.0:
            continue
        if any(alpha[i] != 0 for i in range(n) if i not in market_idx_set):
            continue
        multi = factorial(n_total)
        for i in market_idx:
            multi //= factorial(alpha[int(i)])
        p[k] = poly_1d[n_total] * multi
    return p


# ---------------------------------------------------------------
# Drift and diffusion structure
# ---------------------------------------------------------------

def _drift_matrix_and_const(params: TwoMarketParams, use_P: bool
                            ) -> Tuple[np.ndarray, np.ndarray]:


    m  = params.m_per_market
    n  = params.n_state
    A  = np.zeros((n, n))
    b  = np.zeros(n)

    Zi = params.Z_indices
    Yi = params.Y_indices
    Ri = params.R_index

    # Market 1 (Z) — independent OU per factor
    for k in range(m):
        zk = int(Zi[k])
        A[zk, zk] = -params.kappa_Z[k]
        b[zk]    += params.kappa_Z[k] * params.theta_Z[k]
        if use_P:
            b[zk] += params.kappa_Z[k] * params.lam_Z[k]

    for k in range(m):
        yk = int(Yi[k])
        if k == 0:
            zs = int(Zi[0])
            A[yk, zs] = +params.kappa_Y[0]
            A[yk, yk] = -params.kappa_Y[0]
        else:
            A[yk, yk] = -params.kappa_Y[k]
        if use_P:
            b[yk] += params.kappa_Y[k] * params.lam_Y[k]

    if use_P and m >= 1:
        sz0 = float(params.sigma_Z[0])
        if abs(sz0) > 1e-12:
            phi_Z_slow = (float(params.kappa_Z[0])
                          * float(params.lam_Z[0]) / sz0)
            A[int(Yi[0]), Ri] += float(params.sigma_Y[0]) * phi_Z_slow

    # Jacobi factor R
    A[Ri, Ri] = -params.kappa_R
    b[Ri]     = params.kappa_R * params.theta_R
    if use_P:
        b[Ri] += params.kappa_R * params.lam_R

    return A, b


def _sigma_terms(params: TwoMarketParams
                 ) -> List[Tuple[int, int, np.ndarray, float]]:
    # Entries of Sigma(x) := sigma sigma^T(x) as a sum of monomials.

    n   = params.n_state
    m   = params.m_per_market
    Zi  = params.Z_indices
    Yi  = params.Y_indices
    Ri  = params.R_index

    zero  = np.zeros(n, dtype=int)
    terms: List[Tuple[int, int, np.ndarray, float]] = []

    # OU diagonals
    for k in range(m):
        zk = int(Zi[k]); yk = int(Yi[k])
        terms.append((zk, zk, zero.copy(), float(params.sigma_Z[k]) ** 2))
        terms.append((yk, yk, zero.copy(), float(params.sigma_Y[k]) ** 2))

    # R diagonal: sigma_R^2 (1 - R^2)
    terms.append((Ri, Ri, zero.copy(), float(params.sigma_R) ** 2))
    mi_RR = zero.copy(); mi_RR[Ri] = 2
    terms.append((Ri, Ri, mi_RR, -float(params.sigma_R) ** 2))

    # Slow-pair cross: R sigma_Z0 sigma_Y0
    z0 = int(Zi[0]); y0 = int(Yi[0])
    mi_R = zero.copy(); mi_R[Ri] = 1
    terms.append((min(z0, y0), max(z0, y0), mi_R,
                  float(params.sigma_Z[0]) * float(params.sigma_Y[0])))

    return terms


# ---------------------------------------------------------------
# Generator
# ---------------------------------------------------------------

def infinitesimal_generator_two_market(params: TwoMarketParams,
                                       N: int,
                                       use_P: bool = False) -> np.ndarray:
    n               = params.n_state
    A_drift, b_const = _drift_matrix_and_const(params, use_P=use_P)
    sigma_terms     = _sigma_terms(params)

    basis, idx = build_basis(n, N)
    dim        = len(basis)
    G          = np.zeros((dim, dim))

    def _add(beta_target: np.ndarray, alpha_col: int, val: float) -> None:
        if val == 0.0:
            return
        if (beta_target < 0).any():
            return
        t = tuple(int(x) for x in beta_target)
        if t in idx:
            G[idx[t], alpha_col] += val

    for k, alpha in enumerate(basis):
        alpha_arr = np.array(alpha, dtype=int)

        # Drift: L x^alpha = sum_i alpha_i x^{alpha-e_i} b_i(x)
        for i in range(n):
            ai = int(alpha_arr[i])
            if ai == 0:
                continue
            base = alpha_arr.copy(); base[i] -= 1
            # b_const_i contribution -> beta = base
            _add(base, k, ai * b_const[i])
            # A_drift[i, l] x_l contribution -> beta = base + e_l
            for l in range(n):
                aa = A_drift[i, l]
                if aa == 0.0:
                    continue
                t = base.copy(); t[l] += 1
                _add(t, k, ai * aa)

        # Diffusion: L x^alpha += (1/2) tr(Sigma nabla^2 x^alpha)
        for (i, j, mi, coef) in sigma_terms:
            if i == j:
                prefactor = 0.5 * alpha_arr[i] * (alpha_arr[i] - 1)
            else:
                # i < j by construction; the (j, i) symmetric term is
                # absorbed into the prefactor (factor of 2 cancels the 1/2).
                prefactor = alpha_arr[i] * alpha_arr[j]
            if prefactor == 0:
                continue
            base = alpha_arr.copy(); base[i] -= 1; base[j] -= 1
            target = base + np.asarray(mi, dtype=int)
            _add(target, k, prefactor * coef)

    return G


# ---------------------------------------------------------------
# Polynomial-framework prediction
# ---------------------------------------------------------------

class _PredictHelper:
    # Caches expm(dt * G_P) on Pol_2 plus index lookups for e_i and e_i + e_j.
    # One instance per fit (constant dt and parameters).
    def __init__(self, params: TwoMarketParams, dt: float, N_pred: int = 2):
        self.n      = params.n_state
        self.N_pred = N_pred
        G_P         = infinitesimal_generator_two_market(params, N=N_pred,
                                                         use_P=True)
        self.M_pred = expm(dt * G_P)

        basis, idx  = build_basis(self.n, N_pred)
        eye_n       = np.eye(self.n, dtype=int)
        self.e_idx  = np.array([idx[tuple(eye_n[i])] for i in range(self.n)])
        pair_idx    = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                mi = np.zeros(self.n, dtype=int); mi[i] += 1; mi[j] += 1
                pair_idx[i, j] = idx[tuple(mi)]
        self.pair_idx = pair_idx


def _predict(x: np.ndarray, helper: _PredictHelper
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pol_2 polynomial-framework prediction.

    n      = helper.n
    M_pred = helper.M_pred
    h_x    = build_H(x, helper.N_pred)
    hM     = h_x @ M_pred                             # (dim,)

    x_prior = hM[helper.e_idx]                        # (n,)
    M2      = hM[helper.pair_idx]                     # (n, n)
    Q       = M2 - np.outer(x_prior, x_prior)
    Q       = 0.5 * (Q + Q.T)

    dH_x = build_dH(x, helper.N_pred)                 # (n, dim)
    # dH_x @ M_pred[:, e_idx] has shape (n_state_l, n_state_i) where
    # entry (l, i) = d x_prior_i / d x_l. Transpose to A_jac convention.
    A_jac = (dH_x @ M_pred[:, helper.e_idx]).T

    return x_prior, Q, A_jac


# ---------------------------------------------------------------
# R-feasibility (Jacobi on (-1, 1))
# ---------------------------------------------------------------

def lambda_bounds_R(kappa_R: float, theta_R: float, sigma_R: float,
                    ab_margin: float = AB_MARGIN) -> Tuple[float, float]:
    edge = sigma_R ** 2 * (1.0 + ab_margin) / max(kappa_R, 1e-12)
    return edge - 1.0 - theta_R, 1.0 - edge - theta_R


def _lamR_from_ratio(lam_ratio: float, kappa_R: float, theta_R: float,
                     sigma_R: float, ab_margin: float = AB_MARGIN) -> float:
    lam_lower, lam_upper = lambda_bounds_R(kappa_R, theta_R, sigma_R, ab_margin)
    width = lam_upper - lam_lower
    if width > 0:
        return lam_lower + lam_ratio * width
    return 0.5 * (lam_lower + lam_upper)


def _ratio_from_lamR(lam_R: float, kappa_R: float, theta_R: float,
                     sigma_R: float, ab_margin: float = AB_MARGIN,
                     clip_eps: float = 1e-3) -> float:
    lam_lower, lam_upper = lambda_bounds_R(kappa_R, theta_R, sigma_R, ab_margin)
    width = lam_upper - lam_lower
    ratio = (lam_R - lam_lower) / width if width > 0 else 0.5
    return float(np.clip(ratio, clip_eps, 1.0 - clip_eps))


# ---------------------------------------------------------------
# Pack / unpack / bounds
# ---------------------------------------------------------------
#

def pack(params: TwoMarketParams, N_poly: int = 3) -> np.ndarray:
    lam_R_ratio = _ratio_from_lamR(params.lam_R, params.kappa_R,
                                   params.theta_R, params.sigma_R)
    parts = [
        np.asarray(params.kappa_Z, float),
        np.asarray(params.theta_Z, float),
        np.asarray(params.sigma_Z, float),
        np.asarray(params.lam_Z,   float),
        np.asarray(params.kappa_Y, float),
        np.asarray(params.sigma_Y, float),
        np.asarray(params.lam_Y,   float),
        np.array([params.kappa_R, params.theta_R, params.sigma_R, lam_R_ratio]),
        np.array([params.p_delta_1, params.p_beta_1]),
    ]
    if N_poly >= 5:
        parts.append(np.array([params.p_gamma_1, params.p_K_1]))
    parts.append(np.array([params.p_delta_2, params.p_beta_2]))
    if N_poly >= 5:
        parts.append(np.array([params.p_gamma_2, params.p_K_2]))
    parts.append(np.array([params.p_e_1, params.p_e_2]))
    return np.concatenate(parts)


def unpack(vec: np.ndarray, m_per_market: int,
           N_poly: int = 3) -> TwoMarketParams:
    vec = np.asarray(vec, dtype=float)
    m   = m_per_market
    i   = 0
    kappa_Z = vec[i:i + m]; i += m
    theta_Z = vec[i:i + m]; i += m
    sigma_Z = vec[i:i + m]; i += m
    lam_Z   = vec[i:i + m]; i += m
    kappa_Y = vec[i:i + m]; i += m
    sigma_Y = vec[i:i + m]; i += m
    lam_Y   = vec[i:i + m]; i += m
    kappa_R     = float(vec[i]); i += 1
    theta_R     = float(vec[i]); i += 1
    sigma_R     = float(vec[i]); i += 1
    lam_R_ratio = float(vec[i]); i += 1

    p_delta_1 = float(vec[i]); i += 1
    p_beta_1  = float(vec[i]); i += 1
    p_gamma_1 = 0.0
    p_K_1     = 0.0
    if N_poly >= 5:
        p_gamma_1 = float(vec[i]); i += 1
        p_K_1     = float(vec[i]); i += 1

    p_delta_2 = float(vec[i]); i += 1
    p_beta_2  = float(vec[i]); i += 1
    p_gamma_2 = 0.0
    p_K_2     = 0.0
    if N_poly >= 5:
        p_gamma_2 = float(vec[i]); i += 1
        p_K_2     = float(vec[i]); i += 1

    p_e_1 = float(vec[i]); i += 1
    p_e_2 = float(vec[i]); i += 1

    assert i == len(vec), f"unpack consumed {i}/{len(vec)} entries"

    lam_R = _lamR_from_ratio(lam_R_ratio, kappa_R, theta_R, sigma_R)

    return TwoMarketParams(
        kappa_Z=kappa_Z, theta_Z=theta_Z, sigma_Z=sigma_Z, lam_Z=lam_Z,
        kappa_Y=kappa_Y, sigma_Y=sigma_Y, lam_Y=lam_Y,
        kappa_R=kappa_R, theta_R=theta_R, sigma_R=sigma_R, lam_R=lam_R,
        p_delta_1=p_delta_1, p_beta_1=p_beta_1, p_gamma_1=p_gamma_1, p_K_1=p_K_1,
        p_delta_2=p_delta_2, p_beta_2=p_beta_2, p_gamma_2=p_gamma_2, p_K_2=p_K_2,
        p_e_1=p_e_1, p_e_2=p_e_2,
    )


def num_params(m_per_market: int, N_poly: int) -> int:
    m = m_per_market
    k = (7 * m              # kappa_Z, theta_Z, sigma_Z, lam_Z, kappa_Y, sigma_Y, lam_Y
         + 4                # kappa_R, theta_R, sigma_R, lam_R_ratio
         + 4                # p_delta/p_beta per market (2 each * 2 markets)
         + 2)               # p_e_1, p_e_2
    if N_poly >= 5:
        k += 4              # p_gamma + p_K per market 
    return k


def _kappa_bands_market(m_per_market: int):
    # Disjoint OU rate bands within a single market (slow -> fast).
    if m_per_market == 1:
        return [(0.1, 50.0)]
    if m_per_market == 2:
        return [(0.1, 2.0), (2.0, 50.0)]
    raise ValueError(f"m_per_market must be 1 or 2 (got {m_per_market})")


def make_bounds(m_per_market: int, N_poly: int):
    # Bounds in the same order as pack/unpack.
    m = m_per_market
    bounds: list = []

    # Market 1 OU
    bounds += _kappa_bands_market(m)        # kappa_Z
    bounds += [(-1.0, 1.0)] * m             # theta_Z
    bounds += [(0.01, 2.0)] * m             # sigma_Z
    bounds += [(-1.0, 1.0)] * m             # lam_Z

    # Market 2 OU
    bounds += _kappa_bands_market(m)        # kappa_Y
    bounds += [(0.01, 2.0)] * m             # sigma_Y
    bounds += [(-1.0, 1.0)] * m             # lam_Y

    bounds += [(0.1, 50)]                              # kappa_R
    # theta_R (mu_3): pinned when PIN_THETA_R is not None (degenerate axis,
    # see note at PIN_THETA_R), else free in (-0.95, 0.95).
    if PIN_THETA_R is not None:
        bounds += [(float(PIN_THETA_R), float(PIN_THETA_R))]
    else:
        bounds += [(-0.95, 0.95)]
    bounds += [(0.01, 10.0)]                            # sigma_R
    bounds += [(1e-3, 1.0 - 1e-3)]                     # lam_R_ratio

    pin_pbeta = (N_poly == 1)

    # Market 1 spot poly
    bounds += [(-1, 1)]                                          # p_delta_1
    bounds += [(0.0, 0.0)] if pin_pbeta else [(0.001, 5.0)]          # p_beta_1
    if N_poly >= 5:
        bounds += [(0.001, 7.0)]            # p_gamma_1
        bounds += [(-1.0, 1.0)]             # p_K_1 — cubic-shift 

    # Market 2 spot poly
    bounds += [(-1, 1)]                                          # p_delta_2
    bounds += [(0.0, 0.0)] if pin_pbeta else [(0.001, 5.0)]          # p_beta_2
    if N_poly >= 5:
        bounds += [(0.001, 5.0)]            # p_gamma_2
        bounds += [(-1.0, 1.0)]             # p_K_2 — cubic-shift 

    # Observation-noise scalars 
    bounds += [(1e-3, 0.8)]                 # p_e_1
    bounds += [(1e-3, 0.8)]                 # p_e_2

    expected = num_params(m, N_poly)
    assert len(bounds) == expected, (
        f"make_bounds produced {len(bounds)} bounds; "
        f"num_params expects {expected} "
        f"(m_per_market={m}, N_poly={N_poly})"
    )
    return bounds


# ---------------------------------------------------------------
# Initial state and EKF run
# ---------------------------------------------------------------

def _initial_state(params: TwoMarketParams) -> Tuple[np.ndarray, np.ndarray]:

    n   = params.n_state
    m   = params.m_per_market
    x0  = np.zeros(n)
    p0  = np.zeros(n)

    Zi  = params.Z_indices
    Yi  = params.Y_indices
    Ri  = params.R_index

    for k in range(m):
        x0[int(Zi[k])] = params.theta_Z[k] + params.lam_Z[k]
        p0[int(Zi[k])] = params.sigma_Z[k] ** 2 / (2.0 * params.kappa_Z[k])

    for k in range(m):
        if k == 0:
            x0[int(Yi[0])] = (params.theta_Z[0] + params.lam_Z[0]
                              + params.lam_Y[0])
        else:
            x0[int(Yi[k])] = params.lam_Y[k]
        p0[int(Yi[k])] = params.sigma_Y[k] ** 2 / (2.0 * params.kappa_Y[k])

    th_P  = params.theta_R + params.lam_R
    x0[Ri] = th_P
    p0[Ri] = (params.sigma_R ** 2 * (1.0 - th_P ** 2)
              / (2.0 * params.kappa_R + params.sigma_R ** 2))

    return x0, np.diag(p0)


def EKF_step_two_market(params, x, P, helper, Mp_1_t, Mp_2_t,
                        y_t, R_diag_t, N_pricing,
                        R_clip: float = 1.0 - 1e-4):
    # Single EKF step combining markets 1 and 2 into one stacked observation
    # vector of length n_c1 + n_c2. R is clipped to (-R_clip, R_clip) post-update.
    x_prior, Q, A_jac = _predict(np.asarray(x).flatten(), helper)

    P_prior = A_jac @ P @ A_jac.T + Q

    H_x  = build_H(x_prior,  N_pricing)
    dH_x = build_dH(x_prior, N_pricing)

    h_1 = np.array([H_x  @ Mp for Mp in Mp_1_t])      # (n_c1,)
    j_1 = np.array([dH_x @ Mp for Mp in Mp_1_t])      # (n_c1, n_state)
    h_2 = np.array([H_x  @ Mp for Mp in Mp_2_t])      # (n_c2,)
    j_2 = np.array([dH_x @ Mp for Mp in Mp_2_t])      # (n_c2, n_state)

    y_pred = np.concatenate([h_1, h_2])
    H_jac  = np.vstack([j_1, j_2])

    R_mat = np.diag(np.asarray(R_diag_t, dtype=float))
    S     = H_jac @ P_prior @ H_jac.T + R_mat
    if not np.all(np.isfinite(S)):
        raise ValueError("Non-finite S in EKF_step_two_market")

    K     = P_prior @ H_jac.T @ np.linalg.inv(S)
    resid = (np.asarray(y_t).flatten() - y_pred)
    x_post = x_prior + K @ resid

    Ri = params.R_index
    x_post[Ri] = float(np.clip(x_post[Ri], -R_clip, R_clip))

    I_KH   = np.eye(len(x_prior)) - K @ H_jac
    P_post = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
    return x_post.reshape(-1, 1), P_post, resid, S


def EKF_run_two_market(params, x0, P0,
                       y_obs_1, y_obs_2,
                       T1, delta1, T2, delta2,
                       dt, N_pricing,
                       tau_ref: float = TAU_REF_DEFAULT):
    # Run the two-market EKF over a common trading-day axis. y_obs_k has shape
    # (n_steps, n_c_k); T_k, delta_k have shape (n_steps, n_c_k).
    #
    # Returns (log_lik, n_obs_total).
    n_steps = len(y_obs_1)
    if len(y_obs_2) != n_steps:
        raise ValueError("y_obs_1 and y_obs_2 must share the trading-day axis")

    helper = _PredictHelper(params, dt, N_pred=2)

    G_Q   = infinitesimal_generator_two_market(params, N=N_pricing,
                                               use_P=False)
    p_T_1 = build_poly_market(params, market=1, N=N_pricing)
    p_T_2 = build_poly_market(params, market=2, N=N_pricing)

    # Share the expm cache between markets if their maturity grids overlap.
    Mp_1_all, expm_cache = _precompute_Mp(G_Q, p_T_1, T1, delta1)
    Mp_2_all, _          = _precompute_Mp(G_Q, p_T_2, T2, delta2,
                                          expm_cache=expm_cache)

    R_all_1 = precompute_R(T1, params.p_e_1, tau_ref=tau_ref)
    R_all_2 = precompute_R(T2, params.p_e_2, tau_ref=tau_ref)

    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)
    log_lik     = 0.0
    n_obs_total = 0

    for t in range(n_steps):
        y_t      = np.concatenate([np.asarray(y_obs_1[t]).flatten(),
                                   np.asarray(y_obs_2[t]).flatten()])
        R_diag_t = np.concatenate([R_all_1[t], R_all_2[t]])
        try:
            x, P, resid, S = EKF_step_two_market(
                params, x, P, helper,
                Mp_1_all[t], Mp_2_all[t],
                y_t, R_diag_t, N_pricing,
            )
        except (np.linalg.LinAlgError, ValueError):
            return -1e10, 0
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(log_det):
            return -1e10, 0
        n_obs_t      = len(resid)
        n_obs_total += n_obs_t
        log_lik     += -0.5 * (n_obs_t * np.log(2 * np.pi) + log_det
                               + resid @ np.linalg.solve(S, resid))
    return log_lik, n_obs_total


def EKF_MLE(params_vec, y_obs_1, y_obs_2,
            T1, delta1, T2, delta2,
            dt, N_poly, m_per_market,
            tau_ref: float = TAU_REF_DEFAULT):
    # Negative log-likelihood callable for `differential_evolution` / `minimize`.
    try:
        params = unpack(params_vec, m_per_market=m_per_market, N_poly=N_poly)
    except Exception:
        return 1e10

    # Q-feasibility (Feller) soft penalty for the correlation factor R, a
    # Jacobi process on (-1, 1).
    target = 1.0 + AB_MARGIN
    a_R_Q  = params.kappa_R * (1.0 + params.theta_R) / params.sigma_R ** 2
    b_R_Q  = params.kappa_R * (1.0 - params.theta_R) / params.sigma_R ** 2
    pen    = (max(0.0, target - a_R_Q) ** 2
              + max(0.0, target - b_R_Q) ** 2)
    if pen > 0:
        return 1e8 + 1e4 * pen

    try:
        x0, P0     = _initial_state(params)
        log_lik, _ = EKF_run_two_market(
            params, x0, P0, y_obs_1, y_obs_2,
            T1, delta1, T2, delta2,
            dt, N_pricing=N_poly, tau_ref=tau_ref,
        )
    except Exception:
        return 1e10

    return -log_lik


# ---------------------------------------------------------------
# Phi envelope helper
# ---------------------------------------------------------------

def phi_bounds_from_gbar(g_bar, price_scale,
                         y_raw=None,
                         raw_min: float = -500.0, raw_max: float = 4000.0,
                         slack: float = 50.0,
                         margin_rel: float = 0.25, margin_abs: float = 50.0):

    if y_raw is not None:
        y_arr  = np.asarray(y_raw, dtype=float)
        finite = y_arr[np.isfinite(y_arr)]
        if finite.size == 0:
            raise ValueError("phi_bounds_from_gbar: y_raw has no finite values")
        y_lo = float(finite.min()); y_hi = float(finite.max())
        rng  = y_hi - y_lo
        raw_min = y_lo - margin_abs - margin_rel * rng
        raw_max = y_hi + margin_abs + margin_rel * rng

    yn_min  = raw_min / price_scale
    yn_max  = raw_max / price_scale
    slack_n = slack    / price_scale
    phi_min = (yn_min - g_bar.max()) - slack_n
    phi_max = (yn_max - g_bar.min()) + slack_n
    return float(phi_min), float(phi_max)
