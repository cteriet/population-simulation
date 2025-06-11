from simulation.population import Population
import numpy as np
import polars as pl


def create_train_val_test_split_series(
    population: Population,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    seed: int = None,
):
    """
    Generates boolean Polars Series for train, validation, and test sets
    based on specified proportions, and stores them in the Population object.

    Args:
        population (Population): The Population object containing the Polars DataFrame
                                 in its `_state` attribute.
        test_size (float): The proportion of the dataset to include in the test split.
                           Must be between 0.0 and 1.0.
        validation_size (float): The proportion of the dataset to include in the validation split.
                                 Must be between 0.0 and 1.0.
        seed (int, optional): Seed for the random number generator for reproducibility.
                              Defaults to None (no fixed seed).

    Raises:
        ValueError: If the sum of test_size and validation_size is not between 0 and 1 (exclusive).
    """

    if not (0.0 <= test_size < 1.0 and 0.0 <= validation_size < 1.0):
        raise ValueError(
            "test_size and validation_size must be between 0.0 and 1.0 (exclusive)."
        )
    if test_size + validation_size >= 1.0:
        raise ValueError(
            "The sum of test_size and validation_size must be less than 1.0."
        )

    n_rows = population._state.height
    train_size = 1.0 - test_size - validation_size

    if seed is not None:
        np.random.seed(seed)

    # Generate random numbers for each row
    ratios = np.random.rand(n_rows)

    # Determine which set each row belongs to based on the ratios
    test_mask = ratios < test_size
    val_mask = (ratios >= test_size) & (ratios < (test_size + validation_size))
    train_mask = ratios >= (test_size + validation_size)

    # Convert numpy boolean arrays to Polars Series
    population.test_set = pl.Series("test_set", test_mask, dtype=pl.Boolean)
    population.val_set = pl.Series("val_set", val_mask, dtype=pl.Boolean)
    population.train_set = pl.Series("train_set", train_mask, dtype=pl.Boolean)
