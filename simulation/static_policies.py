import polars as pl
import numpy as np
from .policy import Policy
from .population import Population
from .ground_truth import (
    create_ground_truth_function,
    calculate_probabilities_from_logits,
    calculate_rewards,
)

from .policy_utils import arg_max_horizontal


class RandomPolicy(Policy):
    """A policy that chooses a random action for each member of the population"""

    def __init__(self):
        super().__init__()

    def choose(self, population: Population) -> pl.Series:
        """
        Chooses a random action for each customer in the population.

        Args:
            population_state_df (pl.DataFrame): The current state of the population.
                                                This DataFrame is expected to contain all
                                                features and historical data, but for this
                                                policy, only the number of customers matters.
            action_hierarchy: The ActionHierarchy object, used to get available actions.
            product_list (list[str]): List of product names. Not directly used by this policy.
            ground_truth_weights (dict): Ground truth weights. Not directly used by this policy.

        Returns:
            pl.Series: A Polars Series containing the randomly chosen action name for each customer.
                       The series will have the same number of rows as population_state_df.
        """
        # Get all actionable nodes from the action hierarchy (those with defined costs)
        # We need to exclude 'NoAction' if it's meant to be treated separately or always available
        available_actions = list(
            list(set(population.action_hierarchy.get_leaves() + ["NoAction"]))
        )

        n_customers = population._state.height

        # Randomly choose an action for each customer
        chosen_actions = np.random.choice(available_actions, n_customers)

        return pl.Series("chosen_action", chosen_actions, dtype=pl.Categorical)


class OptimalPolicy(Policy):
    """A policy that chooses the best action for each member of the population by comparing expected values"""

    def __init__(self):
        super().__init__()

    def choose(self, population: Population) -> pl.Series:
        """
        Chooses the optimal action for each customer based on maximizing expected reward.

        This involves iterating through all possible actions, calculating the expected
        reward for each action for every customer, and then selecting the action
        that yields the highest expected reward for each customer.

        Args:
            population_state_df (pl.DataFrame): The current state of the population,
                                                containing customer features (X) and
                                                potentially current product ownership.
                                                This DataFrame is already expanded.
            action_hierarchy: The ActionHierarchy object, used to get all actionable nodes.
            product_list (list[str]): A list of all relevant product names (leaf nodes).
            ground_truth_weights (dict): The dictionary containing the ground truth
                                         weights for each product, used to calculate
                                         expected logits.

        Returns:
            pl.Series: A Polars Series containing the optimally chosen action name for each customer.
                       The series will have the same number of rows as population_state_df.
        """
        n_customers = population._state.height

        # Get all actionable nodes (actions that have a defined cost)
        # These are the actions the policy can actually 'choose'.
        available_actions = list(
            list(set(population.action_hierarchy.get_leaves() + ["NoAction"]))
        )

        # Prepare a list to store the expected rewards for each action for each customer
        all_actions_expected_rewards = []

        # Create the ground truth function once
        ground_truth_func = create_ground_truth_function(population.config)

        all_actions_expected_rewards = []
        for action in available_actions:
            # Create a temporary DataFrame for this action scenario
            # First, create a series representing this action choice for all customers
            current_action_series = pl.Series(
                "A_t", [action] * n_customers, dtype=pl.Categorical
            )

            population.update_state_with_actions(current_action_series)

            # Calculate expected logits for this action scenario
            # Ensure only columns relevant to the ground truth model are passed to the function
            # The ground_truth_func handles selecting relevant columns based on weights.
            expected_logits_for_action = ground_truth_func(population._state)

            # Convert logits to probabilities
            expected_probabilities_for_action = calculate_probabilities_from_logits(
                expected_logits_for_action
            )

            # Calculate expected rewards based on these probabilities
            # This gives a Series of expected reward per customer for this 'action'
            expected_individual_rewards = calculate_rewards(
                expected_probabilities_for_action,
                population.config,
            )

            # Alias the expected reward series with the action name
            all_actions_expected_rewards.append(
                expected_individual_rewards.alias(action)
            )

        expected_rewards_df = pl.DataFrame(all_actions_expected_rewards)

        # Find the action with the maximum expected reward for each customer
        # We need to get the column name (action) that corresponds to the max value in each row.
        # This requires a bit of manipulation in Polars.

        max_column_expr = arg_max_horizontal(*expected_rewards_df.columns)

        chosen_actions = (
            expected_rewards_df.select(max_column_expr)
            .to_series()
            .cast(pl.Categorical)
            .alias("chosen_action")
        )

        return chosen_actions
