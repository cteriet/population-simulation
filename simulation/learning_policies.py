import polars as pl
import numpy as np

from simulation.inverse_propensity_weighting import (
    calculate_truncated_stabilized_ipw,
    lookup_ipw_from_predicted,
)
from .policy import Policy
from .population import Population
from .ground_truth import (
    calculate_probabilities_from_logits,
    calculate_rewards,
)
from sklearn.linear_model import SGDClassifier
import warnings

from .policy_utils import arg_max_horizontal


class LearningPolicy(Policy):
    """
    Base class for policies that can learn.
    Introduces the `learn` method and `can_learn` / `train_on_use` flags.
    """

    def __init__(self):
        super().__init__()
        self.train_on_use = True  # Default: learn when used in engine

    def learn(
        self,
        population: Population,
        product_outcome_df: pl.DataFrame,
        chosen_action_series: pl.Series,
        action_propensity_series: pl.Series,
        current_step: int,
    ):
        """
        Abstract method for learning. Implement in subclasses.
        """
        raise NotImplementedError("Learning policies must implement a 'learn' method.")


class CausalBanditPolicy(LearningPolicy):
    """
    A causal bandit policy that learns the ground truth probabilities
    and makes choices to maximize expected reward using an epsilon-greedy strategy.
    It trains an SGDClassifier for each product incrementally, using inverse propensity weighting.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.01,  # SGDClassifier handles its own learning rate via 'eta0'
        eta0: float = 0.01,
        fit_interval: int = 1,
        train_on_use: bool = True,
        experiment_fraction: float = 1.0,  # Fraction of population to experiment with
        experiment_min_rounds_since_last_experiment: int = 0,  # Min rounds before subject allowed to re-experiment with
    ):
        super().__init__()
        self.epsilon = epsilon  # Exploration rate
        self.fit_interval = fit_interval  # How often to refit the models
        self.train_on_use = train_on_use  # Whether to train the model during simulation
        self.experiment_fraction = experiment_fraction
        self.experiment_min_rounds_since_last_experiment = (
            experiment_min_rounds_since_last_experiment
        )

        self.models = {}  # Stores SGDClassifier models for each product

        # Parameters for SGDClassifier
        self.sgd_params = {
            "loss": "log_loss",  # Logistic regression
            "penalty": "l2",  # L2 regularization
            "alpha": alpha,  # Regularization strength (acts like 1/learning_rate in some contexts)
            "max_iter": 1,  # Process one sample at a time
            "learning_rate": "adaptive",  # Adaptive learning rate
            "eta0": eta0,  # Initial learning rate for adaptive
            "random_state": 42,
            "warm_start": True,  # Keep parameters from previous fit
        }

    def choose(self, population: Population, current_step: int) -> pl.Series:
        """
        Chooses an action for each customer based on an epsilon-greedy strategy.
        Explores randomly or exploits learned models to maximize expected reward.

        Args:
            population: The Population object containing the current state and config.
            current_step: The current simulation step number (for recency tracking).

        Returns:
            tuple[pl.Series, pl.Series]: A tuple containing:
                - A Polars Series of chosen action names for each customer.
                - A Polars Series of propensity scores for each chosen action.
        """

        n_customers = population._state.height

        # The actions we can choose from
        actions = population.action_hierarchy.get_leaves() + ["NoAction"]

        # If the model has been trained, a model should exist for each of these products
        product_list = population.products

        # Load the rewards per product and costs per action
        product_rewards = population.config["PRODUCT_REWARDS"]
        action_costs = population.config["ACTION_COSTS"]

        action_results = []
        # Loop over choosing any action...
        for action in actions:

            action_cost = action_costs[action]

            # Temporarily set the chosen action of the population to the newly proposed action
            temporary_action = pl.Series([action] * n_customers)
            population.update_state_with_actions(temporary_action)

            # Get the current updated features of the population, including the newly proposed action
            X = population.X.filter(population.train_set).to_numpy()

            # Calculate the expected result for each row, for each product
            prediction_results = []
            for product in product_list:
                model = self.models[product]

                # The model returns a list [1-p, p], we only need the second item
                prediction = model.predict_proba(X)[:, 1]
                result = pl.Series(prediction).alias(product)

                prediction_results.append(result)

            # Create a dataframe of the expected outcome for each product
            expected_outcome = pl.DataFrame(prediction_results)

            # Multiply the expected outcome for each product with the weight of that product
            # subtract the action cost for this specific action
            expected_reward = (
                expected_outcome.with_columns(
                    *[
                        pl.col(column) * product_rewards[column]
                        for column in expected_outcome.columns
                    ]
                )
                - action_cost
            )

            # Calculate the total expected reward of choosing a certain action and append it to the action_results
            expected_total_outcome_expression = pl.sum_horizontal(expected_reward)

            expected_total_outcome = expected_reward.select(
                expected_total_outcome_expression
            )

            action_results.append(expected_total_outcome.to_series().alias(action))

            # return expected_total_outcome

        total_action_rewards = pl.DataFrame(action_results)

        # TODO:
        # 1. [x] som of rij om totale verwachte result van actie te leren
        # 2. [x] verwachte positieve resultaat berekenen door (verwachte_reward_actie - kosten_actie) - (verwachte_reward_geen_actie - kosten_geen_actie)
        # 3. [ ] eligibility rules toepassen om acties uit te sluiten
        # 3.b [ ] tijd sinds laatste bericht meenemen
        # 4. [x] actie kiezen met grootste verwachte opbrengst
        # 5. [ ] random actie kiezen in x% van de gevallen van het eindresultaat

        return total_action_rewards

        # best_action = arg_max_horizontal(*total_action_rewards.columns)

        # chosen_actions = (
        #     total_action_rewards.select(best_action)
        #     .to_series()
        #     .cast(pl.Categorical)
        #     .alias("chosen_action")
        # )

        # return chosen_actions

    #     available_actions = list(population.action_hierarchy.get_leaves())
    #     no_action_node = "NoAction"  # Assuming 'NoAction' is handled as a special case

    #     product_list = population.product_hierarchy.get_leaves()

    #     # Initialize chosen actions and propensity scores
    #     chosen_actions_series = pl.Series(
    #         "chosen_action", [""] * n_customers, dtype=pl.Categorical
    #     )
    #     action_propensity_series = pl.Series(
    #         "action_propensity_score", [0.0] * n_customers, dtype=pl.Float64
    #     )

    #     eligible_for_experiment_mask = (
    #         population._state["last_experiment_round"] == -1
    #     ) | (
    #         current_step - population._state["last_experiment_round"]
    #         >= self.experiment_min_rounds_since_last_experiment
    #     )
    #     eligible_indices = (
    #         population._state.lazy()
    #         .select(pl.col("last_experiment_round").arg_sort())
    #         .collect()["last_experiment_round"]
    #         .to_numpy()
    #     )
    #     eligible_indices = eligible_indices[eligible_for_experiment_mask.to_numpy()]

    #     num_to_experiment = int(self.experiment_fraction * n_customers)
    #     if len(eligible_indices) < num_to_experiment:
    #         warnings.warn(
    #             f"Not enough eligible customers ({len(eligible_indices)}) for experimentation fraction ({self.experiment_fraction}). Experimenting on all eligible."
    #         )
    #         experiment_indices = eligible_indices
    #     else:
    #         # Randomly select a subset of eligible customers for experimentation
    #         experiment_indices = np.random.choice(
    #             eligible_indices, size=num_to_experiment, replace=False
    #         )

    #     # Identify customers not selected for experimentation
    #     non_experiment_indices = np.setdiff1d(
    #         np.arange(n_customers), experiment_indices
    #     )

    #     # For non-experimented customers, choose 'NoAction'
    #     if len(non_experiment_indices) > 0:
    #         chosen_actions_series = (
    #             chosen_actions_series.to_frame()
    #             .with_columns(
    #                 pl.Series(
    #                     non_experiment_indices,
    #                     [no_action_node] * len(non_experiment_indices),
    #                     dtype=pl.Categorical,
    #                 ).alias("chosen_action")
    #             )
    #             .get_column("chosen_action")
    #         )
    #         # Propensity for 'NoAction' when forced is 1.0, but practically 0 for learning purposes
    #         # For IPW, the propensity of *choosing* it by design is 1.0, but if it's forced, it's not a choice.
    #         # We will assign 1.0 for these.
    #         action_propensity_series = (
    #             action_propensity_series.to_frame()
    #             .with_columns(
    #                 pl.Series(
    #                     non_experiment_indices,
    #                     [1.0] * len(non_experiment_indices),
    #                     dtype=pl.Float64,
    #                 ).alias("action_propensity_score")
    #             )
    #             .get_column("action_propensity_score")
    #         )

    #     # Process customers selected for experimentation
    #     if len(experiment_indices) > 0:
    #         experimental_population_state = population._state.row(
    #             experiment_indices, named=True
    #         )
    #         num_experimental_customers = len(experiment_indices)

    #         # Prepare to store actions and propensities for experimental group
    #         exp_chosen_actions = np.empty(num_experimental_customers, dtype=object)
    #         exp_action_propensities = np.empty(
    #             num_experimental_customers, dtype=np.float64
    #         )

    #         # Loop through each customer in the experimental group
    #         for i, customer_idx in enumerate(experiment_indices):
    #             # Epsilon-greedy decision for this individual customer
    #             if np.random.rand() < self.epsilon or not self.models:
    #                 # Exploration: Choose a random action
    #                 chosen_action = np.random.choice(available_actions)
    #                 # Propensity for random choice: 1 / (number of available actions)
    #                 propensity = 1.0 / len(available_actions)
    #             else:
    #                 # Exploitation: Predict optimal action based on learned models
    #                 # Create a temporary single-row DataFrame for prediction
    #                 single_customer_df = pl.DataFrame(experimental_population_state[i])

    #                 # Get features for this single customer
    #                 X_for_prediction = single_customer_df.select(
    #                     population.x_columns
    #                     + population.action_columns
    #                     + population.interaction_columns
    #                     + ["intercept"]
    #                 )

    #                 best_action = None
    #                 max_expected_reward = -np.inf
    #                 action_probabilities_for_ipw = {}  # Store probabilities for IPW

    #                 # Iterate through available actions to find the best one for this customer
    #                 for action in available_actions:
    #                     # Temporarily modify X_for_prediction to reflect this hypothetical action
    #                     # We need to simulate the expanded action features and interaction terms
    #                     # for this single customer and this hypothetical action.
    #                     temp_df = single_customer_df.clone()
    #                     temp_df = temp_df.with_columns(
    #                         pl.Series("A_t", [action], dtype=pl.Categorical)
    #                     )
    #                     temp_expanded_actions = population.transform_actions_for_policy(
    #                         temp_df["A_t"]
    #                     )
    #                     temp_interactions = (
    #                         population.calculate_interactions_for_policy(
    #                             temp_expanded_actions, temp_df
    #                         )
    #                     )

    #                     # Create the full feature vector for prediction for this hypothetical action
    #                     # Ensure column order matches training
    #                     features_for_single_pred = temp_df.select(
    #                         population.x_columns + ["intercept"]
    #                     ).with_columns(temp_expanded_actions, temp_interactions)

    #                     # Reorder columns to match the order used during training (important for SGD)
    #                     # This requires knowing the exact column order that SGDClassifier expects.
    #                     # This is a bit tricky, as SGDClassifier doesn't store feature names directly.
    #                     # For now, let's assume the column order is consistent with how `learn` creates X_train.

    #                     # A more robust solution might involve a ColumnTransformer or similar preprocessing pipeline
    #                     # that stores feature order, or training on Polars DataFrames directly if a Polars-native
    #                     # LR/SGD is available. Given the current setup, we need to manually ensure order.

    #                     # The simplest assumption for now is that the order of `population.x_columns`,
    #                     # `population.action_columns`, `population.interaction_columns`, and `intercept`
    #                     # is maintained consistently.

    #                     expected_feature_order = (
    #                         population.x_columns
    #                         + population.action_columns
    #                         + population.interaction_columns
    #                         + ["intercept"]
    #                     )

    #                     # Filter features_for_single_pred to only include expected columns and reorder
    #                     actual_cols = features_for_single_pred.columns

    #                     missing_in_features = set(expected_feature_order) - set(
    #                         actual_cols
    #                     )
    #                     if missing_in_features:
    #                         warnings.warn(
    #                             f"During policy exploitation, expected features missing: {missing_in_features}. This might lead to incorrect predictions."
    #                         )
    #                         # Add missing columns with zeros
    #                         for col in missing_in_features:
    #                             features_for_single_pred = (
    #                                 features_for_single_pred.with_columns(
    #                                     pl.lit(0).alias(col)
    #                                 )
    #                             )

    #                     # Drop extra columns not in expected order and reorder
    #                     features_for_single_pred = features_for_single_pred.select(
    #                         expected_feature_order
    #                     )

    #                     predicted_probabilities_for_product_dict = {}
    #                     for product in product_list:
    #                         model = self.models.get(product)
    #                         if model is not None and hasattr(model, "predict_proba"):
    #                             try:
    #                                 # Use .to_numpy() on the single row DataFrame
    #                                 proba = model.predict_proba(
    #                                     features_for_single_pred.to_numpy()
    #                                 )[0, 1]
    #                                 predicted_probabilities_for_product_dict[
    #                                     product
    #                                 ] = proba
    #                             except Exception as e:
    #                                 warnings.warn(
    #                                     f"Prediction failed for product {product} with action {action}: {e}. Setting prob to 0."
    #                                 )
    #                                 predicted_probabilities_for_product_dict[
    #                                     product
    #                                 ] = 0.0
    #                         else:
    #                             predicted_probabilities_for_product_dict[product] = (
    #                                 0.0  # No model, no prediction
    #                             )

    #                     # Calculate expected reward for this action for this customer
    #                     expected_reward = sum(
    #                         predicted_probabilities_for_product_dict.get(p, 0.0)
    #                         * population.product_hierarchy.get_reward(p)
    #                         for p in product_list
    #                     ) - population.action_hierarchy.get_cost(action)

    #                     # Store for IPW calculation later: probability of choosing *this* action
    #                     # P(action) = P(exploit) * P(action|exploit) + P(explore) * P(action|explore)
    #                     # P(action|exploit) is 1.0 for the chosen action if it's the argmax, 0 otherwise.
    #                     # P(action|explore) is 1/len(available_actions)

    #                     # For IPW, we need the probability of the chosen action *under the current policy*.
    #                     # If policy chose exploitation: the probability of the chosen action is 1 - epsilon
    #                     # If policy chose exploration: the probability of the chosen action is epsilon / num_actions
    #                     action_probabilities_for_ipw[action] = (
    #                         expected_reward  # Store the expected reward for argmax selection
    #                     )

    #                     if expected_reward > max_expected_reward:
    #                         max_expected_reward = expected_reward
    #                         best_action = action

    #                 chosen_action = (
    #                     best_action
    #                     if best_action is not None
    #                     else np.random.choice(available_actions)
    #                 )

    #                 # Calculate propensity score for the chosen action
    #                 # This is the probability that the policy selected this action for this customer
    #                 # It's (1-epsilon) if exploited or epsilon/N_actions if explored
    #                 propensity = (1 - self.epsilon) * (
    #                     1.0 if chosen_action == best_action else 0.0
    #                 ) + self.epsilon * (1.0 / len(available_actions))

    #                 # A small epsilon-greedy adjustment to ensure propensity is never zero for observed actions
    #                 # to avoid division by zero in IPW. Add a tiny constant.
    #                 propensity = max(propensity, 1e-9)

    #             # Assign chosen action and propensity for this individual
    #             chosen_actions_series = (
    #                 chosen_actions_series.to_frame()
    #                 .with_columns(
    #                     pl.Series(
    #                         [customer_idx], [chosen_action], dtype=pl.Categorical
    #                     ).alias("chosen_action")
    #                 )
    #                 .get_column("chosen_action")
    #             )

    #             action_propensity_series = (
    #                 action_propensity_series.to_frame()
    #                 .with_columns(
    #                     pl.Series([customer_idx], [propensity], dtype=pl.Float64).alias(
    #                         "action_propensity_score"
    #                     )
    #                 )
    #                 .get_column("action_propensity_score")
    #             )

    #     # Update last_experiment_round for all customers
    #     updated_last_experiment_round = (
    #         population._state["last_experiment_round"].to_numpy().copy()
    #     )

    #     # Increment for non-experimented customers (if not -1)
    #     # For customers who were not selected for experimentation in this round, their counter increases.
    #     # This includes those who were ineligible and those eligible but not chosen.
    #     for idx in non_experiment_indices:
    #         if updated_last_experiment_round[idx] != -1:
    #             updated_last_experiment_round[idx] = (
    #                 updated_last_experiment_round[idx] + 1
    #             )
    #         else:
    #             # If they were never experimented on and not chosen this round, they remain -1
    #             pass

    #     # For experimented customers, set to current_step
    #     updated_last_experiment_round[experiment_indices] = current_step

    #     population._state = population._state.with_columns(
    #         pl.Series(
    #             "last_experiment_round", updated_last_experiment_round, dtype=pl.Int32
    #         )
    #     )

    #     return chosen_actions_series, action_propensity_series

    def learn(
        self,
        population: Population,
        current_step: int,
    ):
        """
        Trains or updates the internal models based on observed data using SGDClassifier.
        This method is called by the Engine after each simulation step.
        It uses Inverse Propensity Weighting (IPW) for unbiased learning.

        Args:
            population: The current Population object.
            product_outcome_df: DataFrame with binary product purchase outcomes for the current step.
            chosen_action_series: Series of actions chosen by the policy for the current step.
            action_propensity_series: Series of propensity scores for each chosen action.
            current_step: The current simulation step number.
        """
        if not self.train_on_use:
            return  # Do not learn if train_on_use is False

        product_list = population.products

        # Get the full expanded state for the current step, which now contains the chosen actions
        # and interactions based on the policy's actual choices.

        # Add the binary product outcomes to the expanded state for training
        y = population.y.filter(population.train_set)

        # The features for training are X, A (expanded), and interactions, plus intercept
        # Convert to numpy outside of the loop as these variables are the same for each y
        X = population.X.filter(population.train_set).to_numpy()

        # Calculate the Inverse Propensity Weights using the calculate_truncated_stabilized_ipw function, based on the actions in the population
        ipw_matrix = calculate_truncated_stabilized_ipw(population)
        ipw = lookup_ipw_from_predicted(ipw_matrix, population._state["A_t"]).to_numpy()

        # Store the ipw weights in the population class for external use
        population.ipw = ipw

        train_ipw = population.ipw[population.train_set]

        print(
            f"  Policy: Collected data for step {current_step}. Total history rows: {y.height}"
        )

        # Use partial_fit at every step for online learning
        if (current_step + 1) % self.fit_interval == 0 and y.height > 0:
            print(
                f"  Policy: Incrementally training models at step {current_step+1}..."
            )

            # Define classes for SGDClassifier's partial_fit. Crucial for first fit.
            classes = np.array([0, 1])

            for product in product_list:
                y_batch = y[f"{product}_t"].to_numpy()

                if product not in self.models or self.models[product] is None:
                    self.models[product] = SGDClassifier(**self.sgd_params)
                    try:
                        self.models[product].fit(
                            X,
                            y_batch,
                            # classes=classes,
                            sample_weight=train_ipw,
                        )
                        print(
                            f"    Initialized and partially fitted model for product: {product}"
                        )
                    except ValueError as e:
                        warnings.warn(
                            f"Could not initialize and partially fit model for product {product}: {e}. Skipping model update for this product."
                        )
                        self.models[product] = None  # Mark as untrainable for now
                else:
                    try:
                        self.models[product].partial_fit(
                            X, y_batch, sample_weight=train_ipw
                        )
                        print(f"    Partially fitted model for product: {product}")
                    except Exception as e:
                        warnings.warn(
                            f"Could not partially fit model for product {product}: {e}. Model for this product might be stale."
                        )
