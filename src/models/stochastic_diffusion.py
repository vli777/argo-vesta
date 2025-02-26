import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm


def multi_seed_diffusion(
    penalized_obj,
    bounds,
    num_runs=10,
    popsize=15,
    maxiter=1000,
    mutation=(0.5, 1),
    recombination=0.7,
    callback=None,
):
    """
    Performs global optimization using stochastic diffusion (differential evolution)
    with multiple random seeds to increase the likelihood of finding the global optimum.

    Parameters:
        penalized_obj (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension of the input.
        num_runs (int): Number of random seeds to try.
        popsize (int): The population size.
        maxiter (int): Maximum number of generations.
        mutation (float or tuple): The mutation factor for population diversity.
        recombination (float): The recombination rate between [0, 1].
        callback (callable, optional): Optional callback function.

    Returns:
        scipy.optimize.OptimizeResult: The best optimization result found.
    """

    cb = callback if callback is not None else (lambda x, convergence: False)
    results = []

    # Wrap the iteration in tqdm to show a progress bar
    for seed in tqdm(
        range(num_runs), desc="Running stochastic diffusion with multiple seeds"
    ):
        result = differential_evolution(
            penalized_obj,
            bounds=bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            callback=cb,
            polish=True,  # Enable local optimization after global search
        )
        results.append(result)

    # Select the best result based on the objective function value
    best_result = min(results, key=lambda r: r.fun)

    if not best_result.success:
        raise ValueError(
            "Stochastic diffusion optimization failed: " + best_result.message
        )

    return best_result
