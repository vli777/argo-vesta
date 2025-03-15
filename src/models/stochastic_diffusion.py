import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm
import time

from utils import logger


def multi_seed_diffusion(
    penalized_obj,
    bounds,
    num_runs: int = 10,
    popsize: int = 15,
    maxiter: int = 1000,
    mutation=(0.5, 1),
    recombination: float = 0.7,
    callback=None,
    time_limit: float = 300.0,
    initial_candidate: np.ndarray = None,
    perturb_scale: float = 0.05,  # fraction of each bound range to perturb
):
    """
    Performs global optimization using stochastic diffusion (differential evolution)
    with multiple random seeds. If an initial candidate is provided, it generates an
    initial population around that candidate by random perturbations, so the global
    search can explore nearby regions rather than being stuck in the local optimum.

    Parameters:
        penalized_obj (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension.
        num_runs (int): Number of random seeds to try.
        popsize (int): Population size.
        maxiter (int): Maximum number of generations.
        mutation (float or tuple): Mutation factor.
        recombination (float): Recombination rate between 0 and 1.
        callback (callable, optional): Optional callback function.
        time_limit (float): Maximum time (in seconds) to wait for diffusion runs.
        initial_candidate (np.ndarray, optional): A candidate solution from the local solver.
        perturb_scale (float): Scale for random perturbation as a fraction of the bound range.

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """
    cb = callback if callback is not None else (lambda x, convergence: False)
    results = []

    # Create a custom initial population
    if initial_candidate is not None:
        ndim = len(initial_candidate)
        init_pop = np.empty((popsize, ndim))
        # First candidate is the local candidate.
        init_pop[0] = initial_candidate
        for i in range(1, popsize):
            candidate = initial_candidate.copy()
            for j, (low, high) in enumerate(bounds):
                # Compute perturbation scale relative to bound range.
                range_val = high - low
                perturbation = np.random.uniform(
                    -perturb_scale * range_val, perturb_scale * range_val
                )
                candidate[j] = np.clip(candidate[j] + perturbation, low, high)
            init_pop[i] = candidate
        init_param = init_pop
    else:
        init_param = "latinhypercube"

    # Create an independent random generator for reproducibility.
    global_rng = np.random.default_rng(42)
    seeds = [global_rng.integers(0, 1e6) for _ in range(num_runs)]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                differential_evolution,
                penalized_obj,
                bounds=bounds,
                strategy="best1bin",
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                seed=seed,
                callback=cb,
                polish=True,
                init=init_param,
            ): seed
            for seed in seeds
        }

        start_time = time.monotonic()
        try:
            for future in tqdm(
                as_completed(futures, timeout=time_limit),
                total=len(futures),
                desc="Stochastic Diffusion",
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning("A future failed with exception: " + str(e))
                if time.monotonic() - start_time > time_limit:
                    logger.info("Time limit reached during diffusion runs.")
                    break
        except TimeoutError:
            logger.info("Time limit reached while waiting for futures.")

    if not results:
        logger.warning(
            "No diffusion runs completed within the time limit; waiting for one result."
        )
        for future in futures:
            try:
                result = future.result(timeout=5)
                results.append(result)
                break
            except Exception:
                continue
        if not results:
            raise TimeoutError("No diffusion results available after waiting.")

    best_result = min(results, key=lambda r: r.fun)
    return best_result
