"""Jacobi state-evolution and pricing kernels, extracted verbatim from
simulate_paths_monthly_jacobi.py. The thin Jacobi script re-exports these so
existing imports keep working. Nothing here is changed numerically.
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

import kalman_filter_jacobi as jac


# Keep X strictly inside (0, 1) so sqrt(X(1-X)) stays real.
JACOBI_EPS = 1e-4


def compute_observations_jacobi(params, state_paths, T_step, delta_step,
                                  N_pricing):
    """state_paths: (n_paths, n_t, m); T_step / delta_step: (n_c,) or (n_t, n_c)."""
    m = len(params.kappa)
    n_paths, n_t, _ = state_paths.shape

    T_arr = np.asarray(T_step, dtype=float)
    D_arr = np.asarray(delta_step, dtype=float)
    rolling = (T_arr.ndim == 2)
    n_c = T_arr.shape[-1]
    if rolling and T_arr.shape[0] != n_t:
        raise ValueError(
            f"compute_observations_jacobi: rolling T_step has {T_arr.shape[0]} "
            f"rows but state_paths has n_t={n_t}.")

    # graded-lex basis, same convention as jac.build_basis
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

    p_T = jac.build_poly_nd(params, m, N_pricing)
    G = jac.infinitesimal_generator_jacobi(
        -params.kappa, params.kappa * params.theta, params.sigma, N_pricing,
    )

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


def simulate_state_paths_jacobi(params, x_start, P_start, dt, n_steps, n_paths,
                                  rng, sample_init=True):
    """Per-factor Euler-Maruyama with hard clipping to (eps, 1-eps). Factors are independent."""
    m = len(params.kappa)
    kappa   = np.asarray(params.kappa,                              dtype=float)
    theta_P = np.asarray(params.theta + params.lam / params.kappa,  dtype=float)
    sigma   = np.asarray(params.sigma,                              dtype=float)
    sqrt_dt = np.sqrt(dt)

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
    x0 = np.asarray(x_start, dtype=float).reshape(-1)

    for p in range(n_paths):
        x = x0 + L0 @ rng.standard_normal(m)
        x = np.clip(x, JACOBI_EPS, 1.0 - JACOBI_EPS)
        paths[p, 0] = x
        for t in range(n_steps):
            drift = kappa * (theta_P - x) * dt
            vol   = sigma * np.sqrt(np.maximum(x * (1.0 - x), 0.0)) * sqrt_dt
            x = x + drift + vol * rng.standard_normal(m)
            x = np.clip(x, JACOBI_EPS, 1.0 - JACOBI_EPS)
            paths[p, t + 1] = x
    return paths


def filter_to_end_jacobi(params, x0, P0, y_obs, T, delta,
                          dt, N_pricing,
                          tau_ref=jac.TAU_REF_DEFAULT):
    """EKF pass over the history. Captures the posterior (x_post, P_post)
    used for the latent-state band, the prior (x_prior, P_prior) used as
    the one-step-ahead state diagnostic, and the per-contract
    h(x_prior[t]) prediction in normalised-residual space (recovered
    from the EKF innovation via y - resid). Returns
        (x_final, P_final, state_filt, state_cov_d,
         state_prior, prior_cov_d, y_pred_norm).
    """
    n_steps = len(y_obs)
    m = len(params.kappa)
    n_c = T.shape[1] if T.ndim == 2 else 1

    p_T   = jac.build_poly_nd(params, m, N_pricing)
    G     = jac.infinitesimal_generator_jacobi(
        -params.kappa, params.kappa * params.theta, params.sigma, N_pricing,
    )
    Mp_all, _ = jac._precompute_Mp(G, p_T, T, delta)
    R_all     = jac.precompute_R(T, params.p_e, tau_ref=tau_ref)

    def h_timedep(_params, x, _tau, _delta, _N, t_idx):
        x_vec  = np.asarray(x).flatten()
        H_x    = jac.build_H(x_vec,  N_pricing)
        dH_x   = jac.build_dH(x_vec, N_pricing)
        h_vals = np.array([H_x  @ Mp for Mp in Mp_all[t_idx]])
        H_jac  = np.array([dH_x @ Mp for Mp in Mp_all[t_idx]])
        return h_vals, H_jac

    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    P = np.atleast_2d(P0)
    state_filt  = np.full((n_steps, m), np.nan)
    state_cov_d = np.full((n_steps, m), np.nan)
    state_prior = np.full((n_steps, m), np.nan)
    prior_cov_d = np.full((n_steps, m), np.nan)
    y_pred_norm = np.full((n_steps, n_c), np.nan)
    for t_idx in range(n_steps):
        # Replay the predict step to capture x_prior, P_prior. f_Jacobi
        # returns (x_prior, Q) and A_Jacobi returns the drift Jacobian.
        x_prior_t, Q_t = jac.f_Jacobi(params, x, dt)
        A_jac_t        = jac.A_Jacobi(params, x, dt)
        P_prior_t      = A_jac_t @ np.asarray(P) @ A_jac_t.T + Q_t
        state_prior[t_idx] = np.asarray(x_prior_t).flatten()
        prior_cov_d[t_idx] = np.diag(P_prior_t)

        x, P, resid, _ = jac.EKF_step(
            params, x, y_obs[t_idx], jac.f_Jacobi, jac.A_Jacobi, P,
            h_timedep, T[t_idx], delta[t_idx], t_idx, dt, N_pricing,
            R_all[t_idx],
        )
        state_filt [t_idx] = np.asarray(x).flatten()
        state_cov_d[t_idx] = np.diag(P)
        # resid = y_obs - h(x_prior); recover y_pred = h(x_prior) =
        # y_obs - resid in normalised-residual units.
        y_pred_norm[t_idx] = np.asarray(y_obs[t_idx]).flatten() - resid
    return (np.asarray(x).flatten(), P,
            state_filt, state_cov_d,
            state_prior, prior_cov_d,
            y_pred_norm)


# Adapter consumed by sim_engine.
ADAPTER = types.SimpleNamespace(
    simulate_state_paths=simulate_state_paths_jacobi,
    compute_observations=compute_observations_jacobi,
    build_seasonality_matrix=jac.build_seasonality_matrix,
    precompute_R=jac.precompute_R,
    tau_ref_default=jac.TAU_REF_DEFAULT,
)
