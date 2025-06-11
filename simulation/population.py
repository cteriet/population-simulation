# population.py
import polars as pl
import numpy as np
import warnings

# from .config import SIMULATION_CONFIG
from .hierarchy import ProductHierarchy, ActionHierarchy
from .action_transformer import transform_actions
from .ground_truth import (
    create_ground_truth_function,
    calculate_probabilities_from_logits,
    sample_bernoulli_from_probabilities,
)


class Population:
    # Type hint config as a dictionary
    def __init__(
        self,
        config: dict,
        product_hierarchy: ProductHierarchy,
        action_hierarchy: ActionHierarchy,
    ):
        self.config = config
        self.product_hierarchy = product_hierarchy
        self.action_hierarchy = action_hierarchy

        # REMOVE THIS
        # self.features_names = self._get_x_column_names()
        self.continuous_cols = self.config["CUSTOMER_FEATURES_X"]["continuous"]
        self.binary_cols = self.config["CUSTOMER_FEATURES_X"]["binary"]
        self.categorical_cols = self.config["CUSTOMER_FEATURES_X"]["categorical"]

        # These will be set once we expand the initial state
        self.x_columns = None
        self.eligibility_columns = [
            f"eligibility_{x}" for x in self.config["INITIAL_ELIGIBILITY_PROBS"]
        ]

        self.action_columns = list(
            set(self._get_action_column_names()) - set(["NoAction"])
        )

        self.interaction_columns = None

        self.products = self.product_hierarchy.get_leaves()

        self.current_product_names = [f"{p}_t" for p in self.products]
        self.previous_product_names = [f"{p}_t_minus_1" for p in self.products]

        # Initialize the state DataFrame S_t based on config access
        self._state = self.create_customer_dataframe(self.config, self.products)
        self._state = self._expand_state()

        # Set the experimentation counter for the population to -1
        self._state = self._state.with_columns(
            pl.lit(9999).alias("last_experiment_round").cast(pl.Int32)
        )

        # train, test and validation indexes of the population of the last evolution of the population
        self.train_set: pl.Series = None
        self.test_set: pl.Series = None
        self.val_set: pl.Series = None

        # inverse propensity weights of the last treatment assignment for reference
        self.ipw: pl.Series = None

    @property
    def X_variables(self):
        return (
            self.x_columns
            + self.action_columns
            + self.interaction_columns
            + ["intercept"]
        )

    @property
    def X(self):
        return self._state[self.X_variables]

    def create_customer_dataframe(
        self, config: dict, product_names: list[str]
    ) -> pl.DataFrame:
        """
        Creates a Polars DataFrame based on the provided simulation configuration
        and adds additional columns related to product actions.

        Args:
            config (dict): A dictionary containing simulation parameters, including
                        customer features.
            product_names (list[str]): A list of product names to create related columns.

        Returns:
            pl.DataFrame: A Polars DataFrame with simulated customer and product data.
        """
        n_customers = config["N_CUSTOMERS"]
        customer_features_x = config["CUSTOMER_FEATURES_X"]

        data = {}

        # Create continuous columns
        for col_name in customer_features_x["continuous"]:
            data[col_name] = np.random.normal(0, 1, n_customers).astype(np.float32)

        # Create binary columns
        for col_name in customer_features_x["binary"]:
            data[col_name] = np.random.randint(0, 2, n_customers).astype(
                np.int8
            )  # Use int8 for 0/1

        # Initialize the Polars DataFrame with the existing data
        df = pl.DataFrame(data)

        # Create categorical columns
        for col_name, categories in customer_features_x["categorical"].items():
            df = df.with_columns(
                pl.Series(
                    col_name,
                    np.random.choice(categories, n_customers),
                    dtype=pl.Categorical,
                )
            )

        # 1. Add 'A_t' column with 'NoAction'
        df = df.with_columns(
            pl.Series("A_t", ["NoAction"] * n_customers, dtype=pl.Categorical)
        )

        # 2. Add product_name_t columns filled with 0s (u8)
        for product in product_names:
            column_name = f"{product}_t"
            df = df.with_columns(
                pl.Series(column_name, [0] * n_customers, dtype=pl.UInt8)
            )

        # 3. Add product_name_t_minus_1 columns filled with 0s (u8)
        for product in product_names:
            column_name = f"{product}_t_minus_1"
            df = df.with_columns(
                pl.Series(column_name, [0] * n_customers, dtype=pl.UInt8)
            )

        # 4. Add eligibility indicators
        for action in self.config["INITIAL_ELIGIBILITY_PROBS"]:
            action_eligibility_name = f"eligibility_{action}"
            df = df.with_columns(
                pl.Series(
                    action_eligibility_name,
                    np.random.binomial(
                        n=1,
                        p=self.config["INITIAL_ELIGIBILITY_PROBS"][action],
                        size=n_customers,
                    ),
                )
            )

        # 5. Add a constant 'constant' to be used in models
        df = df.with_columns(pl.Series("intercept", [1] * n_customers, dtype=pl.UInt8))

        return df

    def _expand_state(self) -> pl.DataFrame:
        """
        Expands the population state to a dataframe that can be used in models.

        Returns:
            pl.DataFrame: The expanded Polars DataFrame.
        """
        # Combine all columns to keep directly
        x_cols = self.continuous_cols + self.binary_cols

        n_rows = self._state.shape[0]

        # Start with a selection of the columns we want to retain
        expanded_df = self._state.select(x_cols)

        # 1. One-hot encode categorical variables
        for col_name in self.categorical_cols:

            level_to_exclude = self.config[
                "CUSTOMER_FEATURES_CATEGORICAL_LEVELS_TO_DROP"
            ][col_name]

            # Polars one-hot encoding: use pl.Categorical and then to_dummies
            # We need to leave out one category. By default, to_dummies will drop the first one.
            # Ensure the series is categorical for efficient one-hot encoding
            categorical_series = self._state[col_name]

            # Generate dummy variables, dropping the category that is names in the config file
            dummy_df = categorical_series.to_dummies(separator="_")
            dummy_df = dummy_df.select(pl.exclude(f"{col_name}_{level_to_exclude}"))

            expanded_df = pl.concat([expanded_df, dummy_df], how="horizontal")

        # The new x_cols are all the continuous/binary/one hot encoded columns
        self.x_columns = expanded_df.columns

        # 2. Add the outcome columns to the dataframe
        y_cols = self.current_product_names + self.previous_product_names

        expanded_df = pl.concat(
            [expanded_df, self._state.select(y_cols)], how="horizontal"
        )

        # 4. Expand A_t column using transform_actions
        actions_expanded_df = transform_actions(
            self._state["A_t"], self.action_hierarchy
        )

        expanded_df = pl.concat([expanded_df, actions_expanded_df], how="horizontal")

        # Add the original A_t column
        expanded_df = expanded_df.with_columns(self._state["A_t"])

        # 5. Create Cartesian product columns (x_features * actions)
        interactions = []
        interaction_column_names = []
        for action_col in self.action_columns:
            for x_feature in self.x_columns:
                new_col_name = (
                    f"{action_col}_x_{x_feature}"  # Naming convention for new columns
                )
                interactions.append(
                    (expanded_df[action_col] * expanded_df[x_feature]).alias(
                        new_col_name
                    )
                )

                interaction_column_names.append(new_col_name)

        expanded_df = expanded_df.with_columns(interactions)
        self.interaction_columns = interaction_column_names

        # 6. Add 'intercept' column filled with 1s
        expanded_df = expanded_df.with_columns(
            pl.Series("intercept", [1] * n_rows, dtype=pl.UInt8)
        )

        # 7. Add eligibility indicators
        for action in self.config["INITIAL_ELIGIBILITY_PROBS"]:
            action_eligibility_name = f"eligibility_{action}"
            expanded_df = expanded_df.with_columns(
                pl.Series(
                    action_eligibility_name,
                    np.random.binomial(
                        n=1,
                        p=self.config["INITIAL_ELIGIBILITY_PROBS"][action],
                        size=n_rows,
                    ),
                )
            )

        return expanded_df

    def update_state_with_actions(self, actions: pl.Series, update_last_round=False):
        actions_df = transform_actions(actions, self.action_hierarchy)

        interactions = []
        for action_col in self.action_columns:
            for x_feature in self.x_columns:
                new_col_name = (
                    f"{action_col}_x_{x_feature}"  # Naming convention for new columns
                )
                interactions.append(
                    (actions_df[action_col] * self._state[x_feature]).alias(
                        new_col_name
                    )
                )

        # Update the state for actions and interactions between X and action A
        # A_t is kept in the state of the population, so that we can calculate the IPW
        self._state = (
            self._state.with_columns(
                *[actions_df[col_name] for col_name in actions_df.columns]
            )
            .with_columns(interactions)
            .with_columns(actions.alias("A_t"))
        )

        if update_last_round:
            self._state = self._state.with_columns(
                pl.when(actions == "NoAction")
                .then(pl.lit(0))
                .otherwise(pl.col("last_experiment_round"))
                .alias("last_experiment_round")
            )

    def simulate_product_possession_outcome(
        self, seed: int | None = None
    ) -> pl.DataFrame:
        """
        Simulates the outcome of product possession based on the current state
        and ground truth weights. This involves:
        1. Calculating logit scores using the ground truth function.
        2. Converting logits to probabilities using the sigmoid function.
        3. Sampling binary (0/1) outcomes based on these probabilities.
        4. Updating the population's product ownership state based on the regime.

        Args:
            ground_truth_weights (dict): The dictionary containing the ground truth
                                         weights for each product.
            seed (int | None): An optional integer seed for reproducibility of the
                                Bernoulli sampling.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the binary (0/1) product
                          possession outcomes for the current time step for each product.
                          Columns are named after the products (e.g., 'Coffee', 'Tea').
        """
        # Create the ground truth function dynamically
        ground_truth_func = create_ground_truth_function(self.config)

        # Calculate logit scores for all products for the current state
        # The ground_truth_func expects the expanded state DataFrame
        logits_df = ground_truth_func(self._state)

        # Calculate probabilities from logits
        probabilities_df = calculate_probabilities_from_logits(logits_df)
        self.probabilities = probabilities_df

        # Sample binary outcomes from probabilities
        binary_outcomes_df = sample_bernoulli_from_probabilities(
            probabilities_df, seed=seed
        )

        # Update the population's internal product ownership state
        # This handles the shifting of _t to _t_minus_1 and updating _t based on the regime
        self.update_product_ownership(binary_outcomes_df)

        return binary_outcomes_df

    def _get_action_column_names(self):
        return self.action_hierarchy.get_all_non_root_nodes()

    def get_state(self) -> pl.DataFrame:
        """Returns the current state DataFrame S_t."""
        return self._state

    def get_product_ownership(self) -> pl.DataFrame:
        return self._state.select(self.current_product_names)

    def get_previous_product_ownership(self) -> pl.DataFrame:
        return self._state.select(self.previous_product_names)

    def get_current_actions(self) -> pl.Series:
        """Returns the current actions taken (A_t) part of the state."""
        # Ensure the Series has the correct name 'A_t' and return it if it exists
        if "A_t" in self._state.columns:
            return self._state["A_t"].alias("A_t")
        else:
            # Should not happen if initialization was successful and N_CUSTOMERS > 0
            warnings.warn(
                "A_t column not found in state. Returning empty String series."
            )
            return pl.Series("A_t", [], dtype=pl.String)

    def update_product_ownership(self, new_product_ownership: pl.DataFrame):
        """
        Updates the product ownership columns in the internal state DataFrame.

        This function performs two main updates:
        1. Shifts current product ownership ({product_name}_t) to become
           previous product ownership ({product_name}_t_minus_1).
        2. Overwrites current product ownership ({product_name}_t) with the
           new sampled binary results.

        Args:
            new_product_ownership (pl.DataFrame): A Polars DataFrame containing
                                                  the new binary (0/1) product
                                                  ownership for the current timestep.
                                                  Expected columns are {product_name}_t.
        """

        updates = []
        for i, product in enumerate(self.products):

            current_product_possesion = f"{product}_t"
            previous_product_possession = f"{product}_t_minus_1"

            updates.append(
                pl.Series(
                    previous_product_possession, self._state[current_product_possesion]
                )
            )
            updates.append(
                pl.Series(
                    current_product_possesion,
                    self._state[current_product_possesion]
                    + new_product_ownership[product],
                )
            )

        # Apply all updates in a single with_columns call for efficiency
        self._state = self._state.with_columns(updates)

    @property
    def y(self):
        return self.get_product_ownership() - self.get_previous_product_ownership()
