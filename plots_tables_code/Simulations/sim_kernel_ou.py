"""OU state-evolution and pricing kernels, extracted verbatim from
simulate_paths_monthly.py.
"""
import os
import sys
import types
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import expm

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.isfile(
        os.path.join(_root, "kalman_common.py")):
    _root = os.path.dirname(_root)
for _p in (_here, _root,
           os.path.join(_root, "plots_tables_code"),
           os.path.join(_root, "plots_tables_code", "BIC"),
           os.path.join(_root, "plots_tables_code", "Simulations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Kalman_filter_LD as ld


def filter_to_end(params, x0, P0, y_obs, T, delta,
                   dt, N_pricing,
                   tau_ref=ld.TAU_REF_DEFAULT):
    """EKF pass over the historical horizon. Captures post (x_post,
    P_post), prior (x_prior, P_prior), and the per-contract one-step-
    ahead prediction h(x_prior[t]) in normalised-residual space.
    Returns
        (x_final, P_final, state_filt, state_cov_d,
         state_prior, prior_cov_d, y_pred_norm).
    """
    n_steps = len(y_obs)
    m = len(params.theta)
    n_c = T.shape[1] if T.ndim == 2 else 1

    p_T = ld.build_poly_nd(params, m, N_pricing)
    G   = ld.infinitesimal_generator(params.a, params.b,
                                      params.c, params.d,
                                      params.rho, N_pricing)
    Mp_all, _ = ld._precompute_Mp(G, p_T, T, delta)
    R_all     = ld.precompute_R(T, params.p_e, tau_ref=tau_ref)

    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)

    state_filt  = np.full((n_steps, m), np.nan)
    state_cov_d = np.full((n_steps, m), np.nan)
    state_prior = np.full((n_steps, m), np.nan)
    prior_cov_d = np.full((n_steps, m), np.nan)
    y_pred_norm = np.full((n_steps, n_c), np.nan)

    for t in range(n_steps):
        x_prior, Q = ld.f_OU(params, x, dt)
        A_jac      = ld.A_OU(params, x, dt)
        P_prior    = A_jac @ P @ A_jac.T + Q

        x_vec  = np.asarray(x_prior).flatten()
        state_prior[t] = x_vec
        prior_cov_d[t] = np.diag(P_prior)

        H_x    = ld.build_H(x_vec, N_pricing)
        dH_x   = ld.build_dH(x_vec, N_pricing)
        h_vals = np.array([H_x  @ Mp for Mp in Mp_all[t]])
        H_jac  = np.array([dH_x @ Mp for Mp in Mp_all[t]])
        if H_jac.ndim == 1:
            H_jac = H_jac.reshape(-1, 1)

        # h(x_prior) in normalised-residual space.
        y_pred_norm[t] = h_vals

        R_mat  = np.diag(R_all[t].astype(float))
        S      = H_jac @ P_prior @ H_jac.T + R_mat
        K      = P_prior @ H_jac.T @ np.linalg.inv(S)
        resid  = (y_obs[t] - h_vals).flatten()
        x_post = x_vec + K @ resid
        I_KH   = np.eye(m) - K @ H_jac
        P_post = I_KH @ P_prior @ I_KH.T + K @ R_mat @ K.T
        x = x_post.reshape(-1, 1)
        P = P_post

        state_filt [t] = x.flatten()
        state_cov_d[t] = np.diag(P)

    return (x.flatten(), P,
            state_filt, state_cov_d,
            state_prior, prior_cov_d,
            y_pred_norm)


def simulate_state_paths(params, x_start, P_start, dt, n_steps, n_paths,
                          rng, sample_init=True):
    """Simulate OU trajectories forward. sample_init draws each path's start from N(x_start, P_start)."""
    m = len(params.theta)
    theta = params.theta
    mu_P  = ld._mu_P(params)
    c     = params.c
    rho   = params.rho

    alpha = np.exp(-theta * dt)
    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Q[i, j] = (rho[i, j] * c[i] * c[j] *
                       (1.0 - np.exp(-(theta[i] + theta[j]) * dt)) /
                       (theta[i] + theta[j]))
    L_Q = np.linalg.cholesky(Q + 1e-12 * np.eye(m))

    if sample_init and m > 0:
        try:
            L0 = np.linalg.cholesky(P_start + 1e-12 * np.eye(m))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(P_start)
            w = np.clip(w, 0.0, None)
            L0 = V @ np.diag(np.sqrt(w))
    else:
        L0 = np.zeros((m, m))

    paths = np.empty((n_paths, n_steps + 1, m))
    x0 = np.asarray(x_start).reshape(-1)
    for p in range(n_paths):
        x = x0 + L0 @ rng.standard_normal(m)
        paths[p, 0] = x
        for t in range(n_steps):
            eps = L_Q @ rng.standard_normal(m)
            x = alpha * x + mu_P * (1.0 - alpha) + eps
            paths[p, t + 1] = x
    return paths


def compute_observations(params, state_paths, T_step, delta_step, N_pricing):
    """state_paths: (n_paths, n_t, m); T_step / delta_step: (n_c,) or (n_t, n_c) rolling.

    Distinct (tau, delta) pairs are deduplicated before expm to keep the cost down.
    """
    m = len(params.theta)
    n_paths, n_t, _ = state_paths.shape

    T_arr = np.asarray(T_step, dtype=float)
    D_arr = np.asarray(delta_step, dtype=float)
    rolling = (T_arr.ndim == 2)
    n_c = T_arr.shape[-1]
    if rolling and T_arr.shape[0] != n_t:
        raise ValueError(
            f"compute_observations: rolling T_step has {T_arr.shape[0]} "
            f"rows but state_paths has n_t={n_t}.")

    A = []
    for n in range(N_pricing + 1):
        for combo in combinations_with_replacement(range(m), n):
            alpha = np.zeros(m, dtype=int)
            for i in combo:
                alpha[i] += 1
            A.append(alpha)
    A = np.stack(A, axis=0)
    dim = A.shape[0]

    x_b = state_paths[:, :, None, :]
    H_all = np.prod(x_b ** A[None, None, :, :], axis=-1)

    p_T = ld.build_poly_nd(params, m, N_pricing)
    G   = ld.infinitesimal_generator(params.a, params.b,
                                      params.c, params.d,
                                      params.rho, N_pricing)

    def _Mp(tau, dlt):
        M = (1.0 / 6.0) * (expm(G * tau)
                           + 4.0 * expm(G * (tau + dlt / 2.0))
                           +       expm(G * (tau + dlt)))
        return M @ p_T

    if rolling:
        pairs = np.stack([T_arr.ravel(), D_arr.ravel()], axis=-1)
        pairs_round = np.round(pairs, decimals=12)
        uniq, inv = np.unique(pairs_round, axis=0, return_inverse=True)
        Mp_uniq = np.empty((dim, len(uniq)))
        for k, (tau, dlt) in enumerate(uniq):
            Mp_uniq[:, k] = _Mp(float(tau), float(dlt))
        Mp_step = Mp_uniq[:, inv].reshape(dim, n_t, n_c)
        return np.einsum('ptd,dtc->ptc', H_all, Mp_step)
    Mp_per_contract = np.stack(
        [_Mp(float(T_arr[c]), float(D_arr[c])) for c in range(n_c)],
        axis=-1,
    )
    return H_all @ Mp_per_contract


ADAPTER = types.SimpleNamespace(
    simulate_state_paths=simulate_state_paths,
    compute_observations=compute_observations,
    build_seasonality_matrix=ld.build_seasonality_matrix,
    precompute_R=ld.precompute_R,
    tau_ref_default=ld.TAU_REF_DEFAULT,
)
