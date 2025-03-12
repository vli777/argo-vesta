import numpy as np
from scipy.optimize import dual_annealing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import logger


def perturb_candidate(
    candidate: np.ndarray, bounds: list, perturb_scale: float = 0.05
) -> np.ndarray:
    """
    Generates a perturbed candidate from the given candidate by adding a small random perturbation
    to each dimension based on the bound range.
    """
    new_candidate = candidate.copy()
    for i, (low, high) in enumerate(bounds):
        range_val = high - low
        perturbation = np.random.uniform(
            -perturb_scale * range_val, perturb_scale * range_val
        )
        new_candidate[i] = np.clip(new_candidate[i] + perturbation, low, high)
    return new_candidate


def multi_seed_dual_annealing(
    penalized_obj,
    bounds,
    num_runs: int = 10,
    maxiter: int = 1000,
    initial_temp: float = 5000,
    visit: float = 3.0,
    accept: float = -3.0,
    callback=None,
    initial_candidate: np.ndarray = None,
    perturb_scale: float = 0.05,
):
    """
    Performs global optimization using dual annealing with multiple random seeds,
    optionally starting from an initial candidate solution. If an initial candidate is provided,
    each run perturbs it slightly so that the search starts in different neighborhoods.

    Parameters:
        penalized_obj (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension.
        num_runs (int): Number of random seeds to try.
        maxiter (int): Maximum number of iterations.
        initial_temp (float): Initial temperature for the annealing algorithm.
        visit (float): The visit parameter controlling the neighborhood search.
        accept (float): The acceptance parameter.
        callback (callable, optional): Optional callback function.
        initial_candidate (np.ndarray, optional): An initial candidate solution to seed the search.
        perturb_scale (float): Scale factor for perturbing the initial candidate.

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """
    cb = callback if callback is not None else (lambda x, f, context: False)
    results = []

    # Create an independent random generator for reproducibility.
    global_rng = np.random.default_rng(42)
    seeds = [global_rng.integers(0, 1e6) for _ in range(num_runs)]

    with ProcessPoolExecutor() as executor:
        futures = {}
        for seed in seeds:
            # If an initial candidate is provided, perturb it for this run.
            if initial_candidate is not None:
                x0 = perturb_candidate(initial_candidate, bounds, perturb_scale)
            else:
                x0 = None

            # Submit the dual_annealing run with the (possibly perturbed) x0.
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

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Dual Annealing"
        ):
            result = future.result()
            results.append(result)

    best_result = min(results, key=lambda r: r.fun)

    if not best_result.success:
        raise ValueError("Dual annealing optimization failed: " + best_result.message)

    return best_result
