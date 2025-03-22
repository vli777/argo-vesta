import numpy as np
from scipy.stats import t


def bocpd(data, hazard_rate=1 / 100, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """
    Bayesian Online Change Point Detection using a Gaussian model with
    unknown mean and variance (Normal-Inverse-Gamma prior).

    Parameters:
      data: 1D array of observations (e.g., log returns)
      hazard_rate: constant hazard rate (probability of a change point at any time)
      mu0, kappa0, alpha0, beta0: hyperparameters for the prior

    Returns:
      R: A (T+1) x (T+1) matrix of run-length probabilities.
         R[t, r] = probability that at time t the current run (or regime) has lasted r steps.
         Note: Row 0 is the prior (before any observations), and rows 1...T correspond to each observation.
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Initial posterior parameters for run-length 0.
    mu_vec = np.array([mu0])
    kappa_vec = np.array([kappa0])
    alpha_vec = np.array([alpha0])
    beta_vec = np.array([beta0])

    for t_idx in range(1, T + 1):
        x = data.iloc[t_idx - 1]  # use data[t_idx-1] if data is a numpy array

        # Vectorized predictive probability calculation for each run-length.
        df_vec = 2 * alpha_vec
        scale_vec = np.sqrt(beta_vec * (kappa_vec + 1) / (alpha_vec * kappa_vec))
        pred_probs = t.pdf(x, df=df_vec, loc=mu_vec, scale=scale_vec)

        # Compute the growth and change point probabilities.
        prev_R = R[t_idx - 1, :t_idx]
        growth_probs = prev_R * pred_probs * (1 - hazard_rate)
        cp_prob = np.sum(prev_R * pred_probs * hazard_rate)

        R[t_idx, 0] = cp_prob
        R[t_idx, 1 : t_idx + 1] = growth_probs
        R[t_idx, : t_idx + 1] /= np.sum(R[t_idx, : t_idx + 1])  # normalize

        # Update the posterior parameters vectorized.
        new_mu = np.empty(t_idx + 1)
        new_kappa = np.empty(t_idx + 1)
        new_alpha = np.empty(t_idx + 1)
        new_beta = np.empty(t_idx + 1)

        # For the change point (run-length = 0): restart from the prior updated with x.
        new_mu[0] = (kappa0 * mu0 + x) / (kappa0 + 1)
        new_kappa[0] = kappa0 + 1
        new_alpha[0] = alpha0 + 0.5
        new_beta[0] = beta0 + 0.5 * (kappa0 / (kappa0 + 1)) * (x - mu0) ** 2

        # For continuing segments (run-length > 0): update all at once.
        new_mu[1:] = (kappa_vec * mu_vec + x) / (kappa_vec + 1)
        new_kappa[1:] = kappa_vec + 1
        new_alpha[1:] = alpha_vec + 0.5
        new_beta[1:] = (
            beta_vec + 0.5 * (kappa_vec / (kappa_vec + 1)) * (x - mu_vec) ** 2
        )

        # Update parameter vectors for the next iteration.
        mu_vec, kappa_vec, alpha_vec, beta_vec = new_mu, new_kappa, new_alpha, new_beta

    return R
