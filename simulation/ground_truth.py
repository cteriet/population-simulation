import random
import polars as pl
import numpy as np
from typing import Dict, Any


def create_ground_truth_function(simulation_config: Dict[str, Any]):
    """
    Creates a closure that calculates ground truth values for each product.

    Args:
        simulation_config (dict): The dictionary containing ground truth weights.

    Returns:
        function: A function `ground_truth(df)` that takes a Polars DataFrame of the
                  expanded population state and returns a new Polars DataFrame
                  with ground truth columns.
    """
    product_weights = simulation_config["GROUND_TRUTH_WEIGHTS"]

    def ground_truth(df: pl.DataFrame) -> pl.DataFrame:
        output_exprs = []

        for product, weights in product_weights.items():
            # Start with the intercept for the product column expression
            # Use pl.lit() to create a literal series, then scale it
            product_expr = pl.lit(weights["intercept"], dtype=pl.Float32)

            # Add weights for X features
            for feature, weight in weights["X"].items():
                if feature in df.columns:
                    product_expr = product_expr + (pl.col(feature) * weight)

            # Add weights for A actions
            for action, weight in weights["A"].items():
                if action in df.columns:
                    product_expr = product_expr + (pl.col(action) * weight)

            # Add weights for interaction terms
            for interaction_name, weight in weights["interaction"].items():
                parts = interaction_name.split("_x_")
                if len(parts) == 2:
                    action, feature = parts[0], parts[1]
                    if action in df.columns and feature in df.columns:
                        product_expr = product_expr + (
                            pl.col(action) * pl.col(feature) * weight
                        )

            output_exprs.append(
                product_expr.alias(product)
            )  # Alias the expression with the product name

        # Select all computed expressions to form the new DataFrame
        return df.select(output_exprs)

    return ground_truth


def calculate_probabilities_from_logits(logits_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the probabilities for each cell in the input DataFrame
    using the sigmoid function on logit scores.

    Args:
        df: A Polars DataFrame where each column contains logit scores (f64).

    Returns:
        A Polars DataFrame of the same shape, containing probabilities (f64)
        between 0 and 1.
    """

    # Define the sigmoid function as a UDF (User Defined Function) for Polars
    # The sigmoid function converts logit scores to probabilities: P = e^s / (1 + e^s)
    def sigmoid_udf(s: pl.Series) -> pl.Series:
        return pl.Series(np.exp(s) / (1 + np.exp(s)))

    # Apply the sigmoid function to all columns to convert logits to probabilities.
    # We use a list comprehension with pl.col(col).map_batches() for efficiency
    # across each column, ensuring the operation is applied element-wise.
    prob_df = logits_df.with_columns(
        [pl.col(col).map_batches(sigmoid_udf) for col in logits_df.columns]
    )
    return prob_df


def sample_bernoulli_from_probabilities(
    prob_df: pl.DataFrame, seed: int | None = None
) -> pl.DataFrame:
    """
    Samples a binary (0, 1) draw for each observation based on the provided
    probabilities.

    Args:
        prob_df: A Polars DataFrame where each column contains probabilities (f64)
                 between 0 and 1.
        seed: An optional integer seed for reproducibility. If None, no seed
              is set, and results will vary with each run.

    Returns:
        A Polars DataFrame of the same shape, containing only 0s and 1s,
        representing the binary (Bernoulli) draws.
    """
    if seed is not None:
        # Set the seed for NumPy's global random state to ensure reproducibility.
        np.random.seed(seed)

    # Generate a DataFrame of random uniform numbers between 0 and 1.
    # The shape of this DataFrame matches the input prob_df.
    random_draws_df = pl.DataFrame(
        {col: np.random.rand(len(prob_df)) for col in prob_df.columns}
    )

    # Compare the random draws with the probabilities to get binary outcomes.
    # If a random draw is less than the corresponding probability, the outcome is 1 (success),
    # otherwise it's 0 (failure).
    # .cast(pl.UInt8) converts the boolean result (True/False) to 1/0.
    binary_df = (random_draws_df < prob_df).cast(pl.UInt8)

    return binary_df


def calculate_rewards(results: pl.DataFrame, simulation_config: Dict):
    reward_df = pl.DataFrame(
        {
            col: results[col].cast(pl.Float32)
            * simulation_config["PRODUCT_REWARDS"][col]
            for col in results.columns
        }
    )

    # Sum across each row to give the reward or expected reward of an individual
    return reward_df.sum_horizontal()


def calculate_total_reward(reward_df):
    # Sum all individual rows to get the total reward of a population
    return reward_df.sum()


def create_ground_truth_weights(features_X, marketing_actions_A, products_B):
    """
    Creates a dictionary with ground truth weights for each product.
    """
    ground_truth_weights = {"GROUND_TRUTH_WEIGHTS": {}}

    for product in products_B:
        product_weights = {
            "intercept": round(random.uniform(-5.0, 5.0), 2),
            "X": {},
            "A": {},
            "interaction": {},
        }

        # Randomly select a few features from X and assign weights
        selected_x_features = random.sample(
            features_X, min(len(features_X), random.randint(2, 4))
        )
        for feature in selected_x_features:
            product_weights["X"][feature] = round(random.uniform(-1.0, 1.0), 2)

        # Randomly select a few marketing actions from A and assign weights
        selected_a_actions = random.sample(
            marketing_actions_A, min(len(marketing_actions_A), random.randint(1, 3))
        )
        for action in selected_a_actions:
            product_weights["A"][action] = round(random.uniform(-1.0, 2.0), 2)

        # Randomly select a few interactions between A and X and assign weights
        num_interactions = random.randint(0, 2)
        for _ in range(num_interactions):
            random_action = random.choice(marketing_actions_A)
            random_feature = random.choice(features_X)
            interaction_name = f"{random_action}_x_{random_feature}"
            product_weights["interaction"][interaction_name] = round(
                random.uniform(-1.0, 1.0), 2
            )

        ground_truth_weights["GROUND_TRUTH_WEIGHTS"][product] = product_weights
    return ground_truth_weights
