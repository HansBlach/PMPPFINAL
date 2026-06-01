"""Single-market simulation engine shared by the OU and Jacobi scripts.

"""
import numpy as np


def simulate_in_history(model, params, n_paths, dt, maturity_hist, delivery_hist,
                        t_years_hist, seas_beta, annual_h,
                        price_scale, N_pricing,
                        x_start, P_start, rng=None):
    """Forward-simulate paths over the historical horizon from N(x_start, P_start),
    priced on the observed (maturity, delivery) grid."""
    if rng is None:
        rng = np.random.default_rng(0)

    n_days, n_c = maturity_hist.shape

    state_paths = model.simulate_state_paths(
        params, x_start, P_start, dt, n_steps=n_days,
        n_paths=n_paths, rng=rng, sample_init=True,
    )
    # Drop t=0 to line up with the historical days.
    state_paths_obs = state_paths[:, 1:, :]

    y_norm_sim = model.compute_observations(
        params, state_paths_obs,
        T_step=maturity_hist, delta_step=delivery_hist,
        N_pricing=N_pricing,
    )

    _, S_hist, _ = model.build_seasonality_matrix(
        np.asarray(t_years_hist), maturity_hist, delivery_hist,
        np.zeros((n_days, n_c)), annual_h=annual_h,
    )
    g_bar = (S_hist @ seas_beta).reshape(n_days, n_c)

    sim_prices_eur = price_scale * (g_bar[None, :, :] + y_norm_sim)
    return sim_prices_eur, state_paths


def simulate_extension(model, params, x_final, P_final, n_paths, n_steps, dt,
                       fut_t, fut_mat, fut_del,
                       seas_beta, annual_h, price_scale, N_pricing,
                       add_obs_noise=True,
                       last_hist_mat=None, last_hist_del=None, rng=None):
    """Forward extension from the EKF posterior, rolling the observed delivery
    periods forward (fut_mat / fut_del are (n_steps, n_c))."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_c = fut_mat.shape[1]

    state_paths = model.simulate_state_paths(
        params, x_final, P_final, dt, n_steps=n_steps,
        n_paths=n_paths, rng=rng, sample_init=True,
    )

    if last_hist_mat is None or last_hist_del is None:
        T_obs = np.vstack([fut_mat[0:1], fut_mat])
        D_obs = np.vstack([fut_del[0:1], fut_del])
    else:
        T_obs = np.vstack([last_hist_mat, fut_mat])
        D_obs = np.vstack([last_hist_del, fut_del])

    y_norm_sim = model.compute_observations(params, state_paths, T_obs, D_obs,
                                            N_pricing=N_pricing)
    y_norm_sim_fut = y_norm_sim[:, 1:, :]

    _, S_fut, _ = model.build_seasonality_matrix(
        fut_t, fut_mat, fut_del, np.zeros((n_steps, n_c)), annual_h=annual_h,
    )
    g_bar_fut = (S_fut @ seas_beta).reshape(n_steps, n_c)

    if add_obs_noise:
        R_diag = model.precompute_R(
            fut_mat, params.p_e, tau_ref=model.tau_ref_default,
        )
        noise = rng.standard_normal((n_paths, n_steps, n_c)) \
            * np.sqrt(R_diag)[None, :, :]
        y_norm_sim_fut = y_norm_sim_fut + noise

    sim_prices_eur = price_scale * (g_bar_fut[None, :, :] + y_norm_sim_fut)
    return sim_prices_eur, state_paths
