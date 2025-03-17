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
    # R[t, r]: run-length probability matrix.
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Posterior parameter vectors for each possible run-length.
    # At t=0, there is only run-length 0 with the prior.
    mu_vec = np.array([mu0])
    kappa_vec = np.array([kappa0])
    alpha_vec = np.array([alpha0])
    beta_vec = np.array([beta0])

    for t in range(1, T + 1):
        x = data[t - 1]
        # Array to store predictive probabilities for each run length r at time t-1
        pred_probs = np.zeros(t)
        # Compute predictive probability for each possible run-length segment
        for r in range(t):
            # For run-length r, use the current posterior parameters.
            mu_r = mu_vec[r]
            kappa_r = kappa_vec[r]
            alpha_r = alpha_vec[r]
            beta_r = beta_vec[r]

            # Student-t predictive distribution parameters:
            # degrees of freedom: 2*alpha_r,
            # location: mu_r,
            # scale: sqrt( beta_r*(kappa_r+1) / (alpha_r*kappa_r) )
            df = 2 * alpha_r
            scale = np.sqrt(beta_r * (kappa_r + 1) / (alpha_r * kappa_r))
            pred_probs[r] = t.pdf(x, df=df, loc=mu_r, scale=scale)

        # Calculate growth probabilities for extending current runs.
        growth_probs = R[t - 1, :t] * pred_probs * (1 - hazard_rate)
        # Change point probability: reset run length to 0.
        cp_prob = np.sum(R[t - 1, :t] * pred_probs * hazard_rate)
        # Update run-length probability matrix for time t:
        R[t, 0] = cp_prob
        R[t, 1 : t + 1] = growth_probs
        # Normalize to avoid numerical issues
        R[t, : t + 1] /= np.sum(R[t, : t + 1])

        # Update posterior parameters for each possible run-length.
        new_mu = np.zeros(t + 1)
        new_kappa = np.zeros(t + 1)
        new_alpha = np.zeros(t + 1)
        new_beta = np.zeros(t + 1)

        # For a change point: restart from the prior updated with x.
        new_mu[0] = (kappa0 * mu0 + x) / (kappa0 + 1)
        new_kappa[0] = kappa0 + 1
        new_alpha[0] = alpha0 + 0.5
        new_beta[0] = beta0 + 0.5 * (kappa0 / (kappa0 + 1)) * (x - mu0) ** 2

        # For continuing existing segment runs: update the parameters recursively.
        for r in range(1, t + 1):
            # Use the posterior from run length r-1 at the previous time step.
            mu_prev = mu_vec[r - 1]
            kappa_prev = kappa_vec[r - 1]
            alpha_prev = alpha_vec[r - 1]
            beta_prev = beta_vec[r - 1]

            new_mu[r] = (kappa_prev * mu_prev + x) / (kappa_prev + 1)
            new_kappa[r] = kappa_prev + 1
            new_alpha[r] = alpha_prev + 0.5
            new_beta[r] = (
                beta_prev + 0.5 * (kappa_prev / (kappa_prev + 1)) * (x - mu_prev) ** 2
            )

        # Replace the old parameter vectors with the updated ones.
        mu_vec = new_mu
        kappa_vec = new_kappa
        alpha_vec = new_alpha
        beta_vec = new_beta

    return R
