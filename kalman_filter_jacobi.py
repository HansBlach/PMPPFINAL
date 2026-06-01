# Jacobi PMPP

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import expm
from itertools import combinations_with_replacement
from functools import lru_cache
from numpy.polynomial.polynomial import polymul, polyint, polyval

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

_LAMBDA_TANH = 1.0


# ---------------------------------------------------------------
# alpha / beta / c parameter maps (Ware 2025 Appendix B)
# ---------------------------------------------------------------

def beta_max_of_alpha(alpha):
    # cone for alpha <= 3/4, ellipse arc above; 0 outside [-3/2, 3].
    a = np.asarray(alpha, dtype=float)
    out = np.where(
        a <= 0.75,
        0.5 + a / 3.0,
        np.sqrt(np.maximum(a * (1.0 - a / 3.0), 0.0)),
    )
    out = np.where((a < -1.5) | (a > 3.0), 0.0, out)
    return out


def map_alpha(alpha_tilde):
    # tanh + piecewise-linear → alpha ∈ (-3/2, 3)
    u = np.tanh(_LAMBDA_TANH * np.asarray(alpha_tilde, dtype=float))
    return np.where(u >= 0.0, 3.0 * u, 1.5 * u)


def map_beta(beta_tilde, alpha):
    v = np.tanh(_LAMBDA_TANH * np.asarray(beta_tilde, dtype=float))
    return beta_max_of_alpha(alpha) * v


def map_c(c_tilde):
    # Raw c_tilde in R → c > 0 via exp.
    return np.exp(np.asarray(c_tilde, dtype=float))


# ---------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------

@dataclass
class jacobiParams:
    kappa:       np.ndarray                       # (m,)
    theta:       np.ndarray                       # (m,)
    lam:         np.ndarray                       # (m,)
    sigma:       np.ndarray                       # (m,)
    p_e:         float                            # noise scalar

    p_delta:     float = 0.0
    alpha_tilde: Optional[np.ndarray] = None
    beta_tilde:  Optional[np.ndarray] = None
    c_tilde:     Optional[np.ndarray] = None
    per_factor_c: bool = True

    @property
    def k(self) -> int:
        if self.alpha_tilde is None:
            return 0
        a = np.asarray(self.alpha_tilde)
        return 0 if a.size == 0 else int(a.shape[1])

    @property
    def a(self):
        return -self.kappa

    @property
    def b(self):
        return self.kappa * self.theta

    @property
    def alpha(self):
        return map_alpha(self.alpha_tilde) if self.k > 0 else np.zeros((0, 0))

    @property
    def beta(self):
        if self.k == 0:
            return np.zeros((0, 0))
        return map_beta(self.beta_tilde, self.alpha)

    @property
    def c(self):
        # Returns shape-(m,) array. If c_tilde is None, returns all-ones.
        m = int(np.asarray(self.kappa).size)
        if self.c_tilde is None:
            return np.ones(m)
        c_raw = map_c(np.asarray(self.c_tilde).reshape(-1))
        if self.per_factor_c:
            if c_raw.size != m:
                raise ValueError(
                    f"per_factor_c=True expects c_tilde of size {m}, "
                    f"got {c_raw.size}.")
            return c_raw
        if c_raw.size != 1:
            raise ValueError(
                f"per_factor_c=False expects c_tilde of size 1, "
                f"got {c_raw.size}.")
        return np.full(m, float(c_raw[0]))


# ---------------------------------------------------------------
# Polynomial-map helpers
# ---------------------------------------------------------------

def k_from_N(N_poly: int) -> int:
    # Number of quadratic factors in Phi_i for a given polynomial degree.
    return N_poly // 2


def k_free_from_N(N_poly: int) -> int:
    # Number of alpha slots per factor in the optimiser vector.
    return (N_poly - 1) // 2


def build_poly_1d(params, i):
    # Returns coefficients in [1, x, ..., x^(2k+1)] for factor i.
    alpha_tilde_i = np.asarray(params.alpha_tilde[i], dtype=float).reshape(-1)
    beta_tilde_i  = np.asarray(params.beta_tilde[i],  dtype=float).reshape(-1)

    k = alpha_tilde_i.size
    if k == 0:
        return np.array([0.0, 1.0])

    alpha = map_alpha(alpha_tilde_i)
    beta  = map_beta(beta_tilde_i, alpha)

    # q_j(2x - 1) expanded in x.
    phi = np.array([1.0])
    for j in range(k):
        a_j, b_j = float(alpha[j]), float(beta[j])
        q_x = np.array([
            1.0 + 2.0 * a_j / 3.0 - 2.0 * b_j,
            -4.0 * a_j + 4.0 * b_j,
            4.0 * a_j,
        ])
        phi = polymul(phi, q_x)

    Phi_anti = polyint(phi)
    Z = polyval(1.0, Phi_anti)
    return Phi_anti / Z


def build_poly_nd(params, m, N):
    # Sum-mode: Phi_total(X) = p_delta + sum_i c_i Phi_i(X_i). Cross terms are zero.
    basis, idx = build_basis(m, N)
    p = np.zeros(len(basis))
    zero_mono = tuple([0] * m)
    if zero_mono in idx:
        p[idx[zero_mono]] = float(getattr(params, "p_delta", 0.0))

    c_vec = np.asarray(params.c, dtype=float).reshape(-1)

    k = params.k
    if k == 0:
        # identity: Phi_i(X_i) = c_i X_i
        for i in range(m):
            unit = tuple(1 if j == i else 0 for j in range(m))
            if unit in idx:
                p[idx[unit]] = float(c_vec[i])
        return p

    for i in range(m):
        phi_i = build_poly_1d(params, i)
        c_i   = float(c_vec[i])
        for deg in range(1, phi_i.size):
            coef = float(phi_i[deg]) * c_i
            if coef == 0.0:
                continue
            mono = tuple(deg if j == i else 0 for j in range(m))
            if mono in idx:
                p[idx[mono]] += coef
    return p


# ---------------------------------------------------------------
# Infinitesimal generator (graded-lex monomial basis)
# ---------------------------------------------------------------

def infinitesimal_generator_jacobi(a, b, sigma, N):
    # Generator of m independent Jacobi processes on the graded-lex monomial basis.
    #     For dX = kappa(theta - X) dt + sigma sqrt(X(1-X)) dW use a = -kappa, b = kappa*theta.
    a     = np.asarray(a,     dtype=float)
    b     = np.asarray(b,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    m = len(a)
    assert b.shape == (m,) and sigma.shape == (m,)

    basis, idx = build_basis(m, N)
    dim = len(basis)
    G = np.zeros((dim, dim))

    for k, beta in enumerate(basis):
        beta = list(beta)

        # diagonal: beta -> beta
        diag = 0.0
        for i in range(m):
            diag += a[i] * beta[i]
            diag -= 0.5 * sigma[i] ** 2 * beta[i] * (beta[i] - 1)
        G[k, k] = diag

        # beta -> beta + e_i
        for i in range(m):
            alpha = beta.copy()
            alpha[i] += 1
            if tuple(alpha) not in idx:
                continue
            val  = (beta[i] + 1) * b[i]
            val += 0.5 * sigma[i] ** 2 * (beta[i] + 1) * beta[i]
            G[k, idx[tuple(alpha)]] += val

    return G


# ---------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------

def f_Jacobi(params, x, dt):
    # Predictive mean and process-noise covariance for m independent Jacobi factors.
    kappa   = params.kappa
    theta_P = params.theta + params.lam / params.kappa
    sigma   = params.sigma

    x  = np.asarray(x).reshape(-1)
    m  = x.size

    alpha = np.exp(-kappa * dt)
    x_nxt = alpha * x + theta_P * (1.0 - alpha)

    # second moment via the 3x3 generator on (1, x, x^2)
    Q = np.zeros((m, m))
    for i in range(m):
        k, th, s = kappa[i], theta_P[i], sigma[i]
        G3 = np.array([[0.0, k * th,           0.0              ],
                       [0.0, -k,               2.0 * k * th + s**2],
                       [0.0,  0.0,             -(2.0 * k + s**2) ]])
        h0   = np.array([1.0, x[i], x[i] ** 2])
        m2_i = (expm(dt * G3.T) @ h0)[2]
        Q[i, i] = max(m2_i - x_nxt[i]**2, 0.0)        # clip tiny negatives

    return x_nxt.reshape(-1, 1), Q


def A_Jacobi(params, x, dt):
    return np.diag(np.exp(-params.kappa * dt))


AB_MARGIN = 0.02


def lambda_bounds(kappa, theta, sigma, ab_margin=AB_MARGIN):
    """lampda enforcing min{a^P, b^P} ≥ 1 + ab_margin.

    """
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    edge  = sigma ** 2 * (1.0 + ab_margin) / 2.0
    lam_lower = edge - kappa * theta
    lam_upper = kappa * (1.0 - theta) - edge
    return lam_lower, lam_upper


def _lam_from_ratio(lam_ratio, kappa, theta, sigma, ab_margin=AB_MARGIN):
    # Map lam_ratio ∈ [0, 1] → λ in the box. Degenerate box → midpoint.
    lam_lower, lam_upper = lambda_bounds(kappa, theta, sigma, ab_margin)
    width   = lam_upper - lam_lower
    midpt   = 0.5 * (lam_lower + lam_upper)
    lr      = np.asarray(lam_ratio, dtype=float)
    return np.where(width > 0, lam_lower + lr * width, midpt)


def _ratio_from_lam(lam, kappa, theta, sigma,
                     ab_margin=AB_MARGIN, clip_eps=1e-3):
    # Inverse of _lam_from_ratio. Used in pack_Jacobi. Clipped just inside
    # the unit interval
    lam_lower, lam_upper = lambda_bounds(kappa, theta, sigma, ab_margin)
    width = lam_upper - lam_lower
    ratio = np.where(width > 0,
                      (np.asarray(lam, dtype=float) - lam_lower) / width,
                      0.5)
    return np.clip(ratio, clip_eps, 1.0 - clip_eps)


# ---------------------------------------------------------------
# Pack / unpack / bounds — canonical optimiser vector layout.
# ---------------------------------------------------------------

def pack_Jacobi(params, N_poly=3):
    k_exp  = k_from_N(N_poly)
    k_free = k_free_from_N(N_poly)

    kappa = np.asarray(params.kappa, float)
    theta = np.asarray(params.theta, float)
    lam   = np.asarray(params.lam,   float)
    sigma = np.asarray(params.sigma, float)
    lam_ratio = _ratio_from_lam(lam, kappa, theta, sigma)
    parts = [
        kappa,
        theta,
        lam_ratio,
        sigma,
        np.array([float(getattr(params, "p_delta", 0.0))]),
    ]
    # c_tilde block
    m_factors = int(np.asarray(params.kappa).size)
    c_size = m_factors if params.per_factor_c else 1
    if params.c_tilde is None:
        c_block = np.zeros(c_size)
    else:
        c_block = np.asarray(params.c_tilde, float).reshape(-1)

    parts.append(c_block)
    if k_exp > 0:
        alpha_full = np.asarray(params.alpha_tilde, float)
        if k_free > 0:
            parts.append(alpha_full[:, :k_free].reshape(-1))
        parts.append(np.asarray(params.beta_tilde, float).reshape(-1))
    parts.append(np.array([float(params.p_e)]))
    return np.concatenate(parts)


def unpack_Jacobi(vec, m, N_poly=3, per_factor_c=True):
    k      = k_from_N(N_poly)
    k_free = k_free_from_N(N_poly)
    vec = np.asarray(vec, dtype=float)

    i = 0
    kappa     = vec[i:i + m]; i += m
    theta     = vec[i:i + m]; i += m
    lam_ratio = vec[i:i + m]; i += m
    sigma     = vec[i:i + m]; i += m
    p_delta   = float(vec[i]); i += 1
    c_size    = m if per_factor_c else 1
    c_tilde   = vec[i:i + c_size].copy(); i += c_size
    if k > 0:
        alpha_tilde = np.zeros((m, k))
        if k_free > 0:
            alpha_tilde[:, :k_free] = (
                vec[i:i + m * k_free].reshape(m, k_free).copy()
            )
            i += m * k_free
        beta_tilde  = vec[i:i + m * k].reshape(m, k).copy(); i += m * k
    else:
        alpha_tilde = None
        beta_tilde  = None
    p_e = float(vec[i]); i += 1
    assert i == vec.size, f"unpack consumed {i}/{vec.size}"

    lam = _lam_from_ratio(lam_ratio, kappa, theta, sigma)

    return jacobiParams(
        kappa=kappa, theta=theta, lam=lam, sigma=sigma, p_e=p_e,
        p_delta=p_delta,
        alpha_tilde=alpha_tilde, beta_tilde=beta_tilde,
        c_tilde=c_tilde, per_factor_c=per_factor_c,
    )


def num_params_ld(m, N_poly, per_factor_c=True):
    # 4m (kappa, theta, lam_ratio, sigma) + 1 (p_delta) + (m or 1) (c_tilde)
    # + k_free*m (free α) + k*m (β) + 1 (p_e).
    # k_free = k for odd N_poly, k_free = k-1 for even N_poly (last α forced to 0).
    k      = k_from_N(N_poly)
    k_free = k_free_from_N(N_poly)
    c_size = m if per_factor_c else 1
    return 4 * m + 1 + c_size + (k_free + k) * m + 1


def _kappa_bands(m):
    # Per-factor kappa bands.
    if m == 1:
        return [(0.1, 16.0)]
    if m == 2:
        return [(0.1, 55), (0.1, 55)]
    if m == 3:
        return [(0.1, 30), (0.1, 30), (0.1, 30.0)]



def _make_bounds_dynamics_block(m):
    head = (
        _kappa_bands(m) +
        [(0.1, 0.9)] * m +      # theta inside (0, 1)
        [(1e-3, 1.0 - 1e-3)] * m +  # lam_ratio in (0, 1); maps to P-feasible λ
        [(1e-3, 1.5)] * m       
    )
    tail = [(1e-3, 1)]          # p_e has been changed from 0.4
    return head, tail
# weekly
def make_bounds(m, N_poly, per_factor_c=True, spot_envelope=None):
    k      = k_from_N(N_poly)
    k_free = k_free_from_N(N_poly)
    head, tail = _make_bounds_dynamics_block(m)
    c_size = m if per_factor_c else 1
    if spot_envelope is not None:
        ps   = float(spot_envelope["price_scale"])
        gS0  = float(spot_envelope["g_S_min"])
        gS1  = float(spot_envelope["g_S_max"])
        s_lo = float(spot_envelope["spot_lo_eur"])
        s_hi = float(spot_envelope["spot_hi_eur"])
        p_delta_pin = s_lo / ps - gS0
        joint_upper = s_hi / ps - gS1
        sum_c_pin   = joint_upper - p_delta_pin

        per_factor_c_pin = sum_c_pin / m
        c_tilde_pin      = float(np.log(per_factor_c_pin))
        poly  = [(p_delta_pin, p_delta_pin)]            # p_delta — pinned
        poly += [(c_tilde_pin, c_tilde_pin)] * c_size   # c_tilde — pinned
    else:
        poly  = [(-7, -7)]                              # p_delta (static fallback)
        poly += [(3.97, 3.97)] * c_size                 # c_tilde (static fallback)
    if k > 0:
        poly += [(-3.0, 3.0)] * (m * k_free)   # alpha_tilde
        poly += [(-3.0, 3.0)] * (m * k)        # beta_tilde
    bounds = head + poly + tail
    expected = num_params_ld(m, N_poly, per_factor_c=per_factor_c)
    assert len(bounds) == expected, (
        f"make_bounds produced {len(bounds)} bounds but num_params_ld "
        f"expects {expected} (m={m}, N_poly={N_poly}, "
        f"per_factor_c={per_factor_c})."
    )
    return bounds
#monthly
# def make_bounds(m, N_poly, per_factor_c=True):
#     k      = k_from_N(N_poly)
#     k_free = k_free_from_N(N_poly)
#     head, tail = _make_bounds_dynamics_block(m)
#     poly = [(-float(m), 1.0)]                  # p_delta
#     c_size = m if per_factor_c else 1
#     poly += [(-3.0, 3.0)] * c_size             # c_tilde
#     if k > 0:
#         poly += [(-3.0, 3.0)] * (m * k_free)   # alpha_tilde (free slots only)
#         poly += [(-3.0, 3.0)] * (m * k)        # beta_tilde (always full size)
#     bounds = head + poly + tail
#     expected = num_params_ld(m, N_poly, per_factor_c=per_factor_c)

#     return bounds


# ---------------------------------------------------------------
# EKF step / run / MLE
# ---------------------------------------------------------------

# Cap variance at 0.25 = max of x(1-x). Earlier I used the stationary
# variance which had an implicit pull toward theta_P = 0.5.
P_POST_DIAG_CAP = 0.25


def EKF_step(params, x, y, f, A, P, h, tau, delta, t_idx, dt, N, R_diag):
    m = np.asarray(x).reshape(-1).shape[0]
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

    # Keep state 1D end-to-end, reshape to (m, 1) at the very end.
    x_prior_flat = np.asarray(x_prior).reshape(-1)
    x_post_flat  = x_prior_flat + K @ resid
    eps = 1e-4
    x_post_flat  = np.clip(x_post_flat, eps, 1.0 - eps)
    assert x_post_flat.shape == (m,), (
        f"EKF_step: x_post has shape {x_post_flat.shape}, expected ({m},)")
    x_post = x_post_flat.reshape(-1, 1)

    I_KH = np.eye(m) - K @ H_Jac
    P_post = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
    for i in range(m):
        P_post[i, i] = min(P_post[i, i], P_POST_DIAG_CAP)
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
                                        R_all[t_idx])
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
             tau_ref=TAU_REF_DEFAULT, per_factor_c=True):
    # Negative log-likelihood for DE / L-BFGS-B.
    try:
        params = unpack_Jacobi(params_vec, m, N_poly=N,
                                per_factor_c=per_factor_c)
    except Exception:
        return 1e10

    theta_P = params.theta + params.lam / params.kappa

    # Q-feasibility soft penalty: a^Q, b^Q ≥ 1 + AB_MARGIN.
    target = 1.0 + AB_MARGIN
    a_Q = 2.0 * params.kappa * params.theta         / params.sigma ** 2
    b_Q = 2.0 * params.kappa * (1.0 - params.theta) / params.sigma ** 2
    pen  = np.sum(np.maximum(0.0, target - a_Q) ** 2)
    pen += np.sum(np.maximum(0.0, target - b_Q) ** 2)
    if pen > 0:
        return 1e8 + 1e4 * pen

    a = 2 * params.kappa * theta_P          / params.sigma ** 2
    b = 2 * params.kappa * (1 - theta_P)    / params.sigma ** 2

    x0 = theta_P.reshape(-1, 1)
    P0 = np.diag(theta_P * (1 - theta_P) / (a + b + 1))

    p_T = build_poly_nd(params, m, N)
    G   = infinitesimal_generator_jacobi(
        -params.kappa, params.kappa * params.theta,
        params.sigma, N,
    )

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

    log_lik, _ = EKF_run(params, x0, y_obs, f_Jacobi, A_Jacobi, P0,
                          h_timedep, T, delta, dt, N, R_all)
    return -log_lik


