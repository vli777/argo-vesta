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
    time_limit: float = 180.0,
    initial_candidate: np.ndarray = None,
):
    """
    Performs global optimization using stochastic diffusion (differential evolution)
    with multiple random seeds, using an initial candidate (from a local solver) if provided.
    Returns the best optimization result obtained within the time limit.

    Parameters:
        penalized_obj (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension.
        num_runs (int): Number of random seeds to try.
        popsize (int): Population size.
        maxiter (int): Maximum number of generations.
        mutation (float or tuple): Mutation factor.
        recombination (float): Recombination rate between 0 and 1.
        callback (callable, optional): Optional callback function.
        time_limit (float): Maximum time in seconds to wait for diffusion runs.
        initial_candidate (np.ndarray, optional): A candidate solution to seed the population.

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """
    cb = callback if callback is not None else (lambda x, convergence: False)
    results = []

    # Determine population initialization.
    if initial_candidate is not None:
        ndim = len(initial_candidate)
        init_pop = np.empty((popsize, ndim))
        init_pop[0] = initial_candidate
        # Fill the rest uniformly from bounds.
        for i in range(1, popsize):
            init_pop[i] = np.array(
                [np.random.uniform(low, high) for (low, high) in bounds]
            )
        init_param = init_pop
    else:
        init_param = "latinhypercube"

    # Create random seeds.
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

    # If no results are available, wait briefly for at least one.
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
