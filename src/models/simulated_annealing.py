import numpy as np
from scipy.optimize import dual_annealing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import logger


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
):
    """
    Performs global optimization using dual annealing with multiple random seeds,
    optionally starting from an initial candidate solution.

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

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """
    cb = callback if callback is not None else (lambda x, f, context: False)
    results = []

    # Create an independent random generator for reproducibility.
    global_rng = np.random.default_rng(42)
    seeds = [global_rng.integers(0, 1e6) for _ in range(num_runs)]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                dual_annealing,
                penalized_obj,
                bounds=bounds,
                maxiter=maxiter,
                initial_temp=initial_temp,
                visit=visit,
                accept=accept,
                callback=cb,
                seed=seed,
                x0=initial_candidate,  # Pass the initial candidate if provided.
            ): seed
            for seed in seeds
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Dual Annealing"
        ):
            result = future.result()
            results.append(result)

    best_result = min(results, key=lambda r: r.fun)

    if not best_result.success:
        raise ValueError("Dual annealing optimization failed: " + best_result.message)

    return best_result
