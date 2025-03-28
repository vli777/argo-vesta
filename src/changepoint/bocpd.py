import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import BayesianRidge

from utils import logger


def bocpd(
    data: np.ndarray,
    hazard_rate0: float = 1 / 60,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    epsilon: float = 1e-8,
    truncation_threshold: float = 1e-4,
    rolling_window: int = 10,
):
    """
    Advanced Bayesian Online Change Point Detection using:
    - Robust Student-t likelihood
    - Adaptive hazard estimation via Bayesian Ridge Regression
    - Sparse posterior approximation (dynamic run-length truncation)

    Parameters:
        data: 1D array-like of observations
        mu0, kappa0, alpha0, beta0: Hyperparameters for Normal-Inverse-Gamma prior
        hazard_rate0: Initial hazard rate for adaptation
        epsilon: Small value for numerical stability
        truncation_threshold: Threshold for dynamic truncation of run-length probabilities
        rolling_window: Rolling window size

    Returns:
        R: Run-length probability matrix
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    mu_vec = np.array([mu0])
    kappa_vec = np.array([kappa0])
    alpha_vec = np.array([alpha0])
    beta_vec = np.array([beta0])

    hazard_model = BayesianRidge()
    run_lengths = np.array([0])
    hazard_rate = 1 / 50  # initial default hazard rate

    for t_idx in range(1, T + 1):
        x = data.iloc[t_idx - 1]

        # Predictive probability with robust Student-t likelihood
        df_vec = 2 * alpha_vec
        scale_vec = np.sqrt(
            beta_vec * (kappa_vec + 1) / (alpha_vec * kappa_vec) + epsilon
        )
        pred_probs = t.pdf(x, df=df_vec, loc=mu_vec, scale=scale_vec) + epsilon

        # Improved adaptive hazard rate estimation with smoothing
        if t_idx > rolling_window and len(run_lengths) > rolling_window:
            hazard_features = run_lengths[-rolling_window:].reshape(-1, 1)
            hazard_target = np.log(pred_probs[-rolling_window:])
            hazard_model.fit(hazard_features, hazard_target)
            estimated_hazard = np.exp(hazard_model.predict([[run_lengths[-1]]])[0])
            hazard_rate = 0.5 * hazard_rate + 0.5 * np.clip(
                estimated_hazard, epsilon, 1
            )
        else:
            hazard_rate = hazard_rate0

        # if t_idx % 20 == 0:
        #     logger.info(f"t = {t_idx}, hazard_rate = {hazard_rate:.5f}")

        prev_R = R[t_idx - 1, : len(run_lengths)]

        growth_probs = prev_R * pred_probs * (1 - hazard_rate)
        cp_prob = np.sum(prev_R * pred_probs * hazard_rate)

        R[t_idx, 0] = cp_prob
        R[t_idx, 1 : len(run_lengths) + 1] = growth_probs

        # Normalization
        R_sum = np.sum(R[t_idx, : len(run_lengths) + 1]) + epsilon
        R[t_idx, : len(run_lengths) + 1] /= R_sum

        # Dynamic truncation of small probabilities
        mask = R[t_idx, : len(run_lengths) + 1] > truncation_threshold
        R[t_idx, : len(run_lengths) + 1] *= mask
        R[t_idx, : len(run_lengths) + 1] /= np.sum(R[t_idx, : len(run_lengths) + 1])
        run_lengths = np.arange(mask.sum())

        # Posterior updates with robust priors
        new_mu = np.empty(len(run_lengths))
        new_kappa = np.empty(len(run_lengths))
        new_alpha = np.empty(len(run_lengths))
        new_beta = np.empty(len(run_lengths))

        # Change point reset
        new_mu[0] = (kappa0 * mu0 + x) / (kappa0 + 1)
        new_kappa[0] = kappa0 + 1
        new_alpha[0] = alpha0 + 0.5
        new_beta[0] = beta0 + 0.5 * (kappa0 / (kappa0 + 1)) * (x - mu0) ** 2

        # Continuing segments
        if len(run_lengths) > 1:
            new_mu[1:] = (kappa_vec[mask[1:]] * mu_vec[mask[1:]] + x) / (
                kappa_vec[mask[1:]] + 1
            )
            new_kappa[1:] = kappa_vec[mask[1:]] + 1
            new_alpha[1:] = alpha_vec[mask[1:]] + 0.5
            new_beta[1:] = (
                beta_vec[mask[1:]]
                + 0.5
                * (kappa_vec[mask[1:]] / (kappa_vec[mask[1:]] + 1))
                * (x - mu_vec[mask[1:]]) ** 2
            )

        mu_vec, kappa_vec, alpha_vec, beta_vec = new_mu, new_kappa, new_alpha, new_beta

    return R
