import numpy as np
from scipy.optimize import dual_annealing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import logger


def multi_seed_dual_annealing(
    penalized_obj,
    bounds,
    num_runs=10,
    maxiter=1000,
    initial_temp=5000,
    visit=3.0,
    accept=-3.0,
    callback=None,
):
    cb = callback if callback is not None else (lambda x, f, context: False)
    results = []

    # Create an independent random generator
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
