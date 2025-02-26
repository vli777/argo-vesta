import numpy as np
from scipy.optimize import dual_annealing
from tqdm import tqdm


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
    global_rng = np.random.default_rng(42)  # Global RNG for reproducibility
    
    # Wrap the iteration in tqdm to show a progress bar
    for _ in tqdm(
        range(num_runs), desc="Running dual annealing with multiple seeds"
    ):
        seed = global_rng.integers(0, 1e6)  # Generate a random seed for each run
        result = dual_annealing(
            penalized_obj,
            bounds=bounds,
            maxiter=maxiter,
            initial_temp=initial_temp,
            visit=visit,
            accept=accept,
            callback=cb,
            seed=seed,  # Different random seed for each run
        )
        results.append(result)

    # Select the best result based on the objective function value
    best_result = min(results, key=lambda r: r.fun)

    if not best_result.success:
        raise ValueError("Dual annealing optimization failed: " + best_result.message)

    return best_result
