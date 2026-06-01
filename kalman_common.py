# Shared helpers for the LD, Jacobi and Two-Market polynomial Kalman filters.

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from itertools import combinations_with_replacement
from functools import lru_cache


# ---------------------------------------------------------------
# Observation-noise reference horizon (1 year). Variance is
# p_e^2 * (tau / tau_ref)
# ---------------------------------------------------------------

TAU_REF_DEFAULT = 1.0


# ---------------------------------------------------------------
# Graded-lex monomial basis
# ---------------------------------------------------------------

@lru_cache(maxsize=None)
def build_basis(m, N):
    basis = []
    for n in range(N + 1):
        for combo in combinations_with_replacement(range(m), n):
            alpha = [0] * m
            for i in combo:
                alpha[i] += 1
            basis.append(tuple(alpha))
    idx = {alpha: k for k, alpha in enumerate(basis)}
    return basis, idx


_basis_cache: dict = {}

def _multi_indices(m, N):
    key = (m, N)
    if key not in _basis_cache:
        idx = []
        for n in range(N + 1):
            for combo in combinations_with_replacement(range(m), n):
                alpha = np.zeros(m, dtype=int)
                for i in combo:
                    alpha[i] += 1
                idx.append(alpha)
        _basis_cache[key] = np.stack(idx, axis=0)    # (K, m)
    return _basis_cache[key]


# ---------------------------------------------------------------
# Monomial vector H(x) and its Jacobian dH(x)
# ---------------------------------------------------------------

def build_H(x, N):
    x = np.atleast_1d(x).flatten().astype(float)
    A = _multi_indices(len(x), N)                    # (K, m)
    return np.prod(x ** A, axis=1)                   # (K,)


def build_dH(x, N):
    # Jacobian of H. Always (m, dim), including m=1.
    x = np.atleast_1d(x).flatten().astype(float)
    m = x.size
    A = _multi_indices(m, N)                         # (K, m)
    K = A.shape[0]
    dH = np.zeros((m, K))
    for i in range(m):
        Ai = A.copy()
        mask = A[:, i] > 0
        coef = np.where(mask, A[:, i], 0).astype(float)
        Ai[mask, i] -= 1
        dH[i, mask] = coef[mask] * np.prod(x ** Ai[mask], axis=1)
    return dH


# ---------------------------------------------------------------
# Observation-noise precomputation
# ---------------------------------------------------------------

def precompute_R(maturity_matrix: np.ndarray,
                 p_e:             float,
                 tau_ref:         float = TAU_REF_DEFAULT,
                 tau_floor:       float = 1e-4) -> np.ndarray:
    # R_ti = p_e^2 * (tau_ti / tau_ref). Sun (2024) p.74.
    tau = np.maximum(np.asarray(maturity_matrix, dtype=float), tau_floor)
    return float(p_e) ** 2 * (tau / float(tau_ref))


# ---------------------------------------------------------------
# Precompute M_i p_T (Simpson rule over each contract's delivery window).
# Returns (Mp_all, expm_cache). Caller can pass in an existing expm_cache
# (TwoMarket uses this to share the cache across DE and FR).
# ---------------------------------------------------------------

def _precompute_Mp(G, p_T, T, delta, expm_cache=None):
    ROUND = 1e-4
    digits = -int(np.log10(ROUND))
    n_steps, n_maturities = T.shape

    if expm_cache is None:
        expm_cache = {}

    s_set = set()
    for t in range(n_steps):
        for i in range(n_maturities):
            tau_i, delta_i = T[t, i], delta[t, i]
            s_set.add(round(tau_i,               digits))
            s_set.add(round(tau_i + delta_i / 2, digits))
            s_set.add(round(tau_i + delta_i,     digits))
    for s in s_set:
        if s not in expm_cache:
            expm_cache[s] = expm(G * s)

    Mp_all = []
    for t in range(n_steps):
        row = []
        for i in range(n_maturities):
            tau_i, delta_i = T[t, i], delta[t, i]
            k0 = round(tau_i,               digits)
            k1 = round(tau_i + delta_i / 2, digits)
            k2 = round(tau_i + delta_i,     digits)
            M_i = (1.0 / 6.0) * (expm_cache[k0]
                                 + 4.0 * expm_cache[k1]
                                 +       expm_cache[k2])
            row.append(M_i @ p_T)
        Mp_all.append(row)
    return Mp_all, expm_cache


# ---------------------------------------------------------------
# Stage-A seasonality 
# ---------------------------------------------------------------

def build_seasonality_matrix(t_years, maturity, delivery_dur, y_obs,
                             annual_h=2):
    # Stage-A seasonality design matrix.
    #
    #     g(t) = c + m * t + sum_{k=1}^{annual_h} [ a_k cos(2 pi k t)+ b_k sin(2 pi k t) ]
    #                                              
    d1 = (t_years[:, None] + maturity).ravel()
    d2 = d1 + delivery_dur.ravel()
    delta = d2 - d1
    d_mid = 0.5 * (d1 + d2)

    n = len(d1)
    cols = [np.ones(n), d_mid]

    for k in range(1, annual_h + 1):
        omega = 2.0 * np.pi * k
        cos_col = (np.sin(omega * d2) - np.sin(omega * d1)) / (omega * delta)
        sin_col = (np.cos(omega * d1) - np.cos(omega * d2)) / (omega * delta)
        cols.append(cos_col)
        cols.append(sin_col)

    S = np.column_stack(cols)

    y    = y_obs.ravel()
    mask = ~np.isnan(y)
    beta, _, _, _ = np.linalg.lstsq(S[mask], y[mask], rcond=None)
    return beta, S, mask


def seasonality_bic(t_years, maturity, delivery_dur, y_obs,
                    annual_h, n_eff=None):
    # Stage-A seasonality BIC.

    beta, S, mask = build_seasonality_matrix(t_years, maturity, delivery_dur,
                                             y_obs, annual_h)
    y     = y_obs.ravel()[mask]
    y_hat = S[mask] @ beta
    resid = y - y_hat
    n_obs = len(resid)
    if n_eff is None:
        n_eff = int(np.asarray(t_years).shape[0])
    ssr   = float(np.sum(resid ** 2))
    sigma2 = ssr / n_obs
    log_lik = -0.5 * n_obs * (np.log(2 * np.pi) + 1.0 + np.log(sigma2))
    # const + linear + 2*annual_h Fourier pairs + sigma^2
    k = 2 + 2 * annual_h + 1
    bic = k * np.log(n_eff) - 2 * log_lik
    try:
        cond_S = float(np.linalg.cond(S[mask]))
    except np.linalg.LinAlgError:
        cond_S = float("inf")
    return dict(annual_h=annual_h,
                n=n_obs, n_obs=n_obs, n_eff=int(n_eff), k=k,
                logL=log_lik, sigma2=sigma2, BIC=bic,
                cond_S=cond_S,
                beta=beta, S=S, mask=mask)
