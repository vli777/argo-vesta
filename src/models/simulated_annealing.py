import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import dual_annealing

from utils import logger


def perturb_candidate(
    candidate: np.ndarray,
    bounds: list,
    perturb_scale: float = 0.3,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Generates a perturbed candidate by blending the candidate with a random positive vector.
    This ensures that if the candidate is near zero, we still obtain nonzero perturbations.
    The candidate is then clipped to the bounds and normalized to sum to target_sum.
    """
    # Generate a random vector with values in [0, 1)
    random_vector = np.random.uniform(0.0, 1.0, size=candidate.shape)
    # Blend candidate with the random vector
    perturbed = (1 - perturb_scale) * candidate + perturb_scale * random_vector
    # Enforce bounds for each dimension
    for i, (low, high) in enumerate(bounds):
        perturbed[i] = np.clip(perturbed[i], low, high)
    # Normalize so that the candidate sums to target_sum
    s = perturbed.sum()
    if s > 0:
        perturbed = perturbed / s * target_sum
    else:
        perturbed = np.full_like(perturbed, target_sum / len(perturbed))
    return perturbed


def multi_seed_dual_annealing(
    penalized_obj,
    bounds,
    num_runs: int = 10,
    maxiter: int = 10000,
    initial_temp: float = 10000,
    visit: float = 10.0,
    accept: float = -10.0,
    callback=None,
    initial_candidate: np.ndarray = None,
    perturb_scale: float = 0.3,
    target_sum: float = 1.0,
):
    """
    Performs global optimization using dual annealing with multiple random seeds.
    If an initial candidate is provided, the first run uses the candidate unmodified,
    and subsequent runs use randomly perturbed versions. This ensures the search covers
    the area around the candidate, including the candidate itself. The function will
    break early if two runs converge to the same result.

    Parameters:
        penalized_obj (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension.
        num_runs (int): Number of random seeds to try.
        maxiter (int): Maximum number of iterations.
        initial_temp (float): Initial temperature for the annealing algorithm.
        visit (float): Visit parameter controlling neighborhood search.
        accept (float): Acceptance parameter.
        callback (callable, optional): Optional callback function.
        initial_candidate (np.ndarray, optional): An initial candidate to seed the search.
        perturb_scale (float): Scale factor for perturbing the candidate.
        target_sum (float): Desired sum of the candidate (e.g. 1.0).

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """
    cb = callback if callback is not None else (lambda x, f, context: False)
    results = []

    # Helper function to determine if two results are effectively the same.
    def is_same_result(res1, res2, tol=1e-6):
        return (
            np.allclose(res1.x, res2.x, rtol=tol, atol=tol)
            and abs(res1.fun - res2.fun) < tol
        )

    # Create reproducible seeds
    global_rng = np.random.default_rng(42)
    seeds = [global_rng.integers(0, 1e6) for _ in range(num_runs)]

    with ProcessPoolExecutor() as executor:
        futures = {}
        for idx, seed in enumerate(seeds):
            if initial_candidate is not None:
                # Use the unmodified candidate for the first run;
                # perturb for subsequent runs.
                x0 = (
                    initial_candidate
                    if idx == 0
                    else perturb_candidate(
                        initial_candidate, bounds, perturb_scale, target_sum
                    )
                )
            else:
                x0 = None  # dual_annealing selects its own starting point.
            future = executor.submit(
                dual_annealing,
                penalized_obj,
                bounds=bounds,
                maxiter=maxiter,
                initial_temp=initial_temp,
                visit=visit,
                accept=accept,
                callback=cb,
                seed=seed,
                x0=x0,
            )
            futures[future] = seed

        early_stop = False
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Dual Annealing"
        ):
            result = future.result()
            results.append(result)
            # Check if this result matches any previous result
            for prev in results[:-1]:
                if is_same_result(prev, result):
                    logger.info("Early stopping: identical result found.")
                    early_stop = True
                    break
            if early_stop:
                break

        if early_stop:
            for fut in futures:
                if not fut.done():
                    fut.cancel()

    best_result = min(results, key=lambda r: r.fun)

    if not best_result.success:
        logger.warning("Dual annealing did not converge: " + best_result.message)

    # Fallback normalization if result is nearly all zeros.
    if np.isclose(best_result.x.sum(), 0, atol=1e-8):
        logger.warning(
            "Best candidate sums to zero; using uniform allocation as fallback."
        )
        best_result.x = np.full_like(best_result.x, target_sum / len(best_result.x))

    return best_result
