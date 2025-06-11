import polars as pl
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold  # For calibration
import numpy as np
import warnings
from .population import Population


WEIGHT_TRUNCATION_PERCENTILE = (
    99  # Truncate weights at this percentile (e.g., 99 for 99th percentile)
)


def calculate_treatment_probabilities(population: Population) -> pl.DataFrame:

    x_columns = population.x_columns
    actions = population.action_hierarchy.get_leaves() + ["NoAction"]

    # Define the model to be fitted
    lgbm_clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(actions), random_state=42, verbosity=0
    )

    # Wrap the LGBMClassifier with CalibratedClassifierCV
    # method='isotonic' or 'sigmoid' can be chosen. 'isotonic' generally performs better
    # if you have enough data, otherwise 'sigmoid' is more robust.
    calibrated_clf = CalibratedClassifierCV(lgbm_clf, method="isotonic", cv=5)

    # Fit the calibrated propensity model
    # Convert A_t to numerical labels as LightGBM works with integers for multiclass
    action_mapping = {label: i for i, label in enumerate(actions)}

    # Convert to pandas for sklearn compatibility
    df_pandas = population._state.to_pandas()
    df_pandas["A_t_encoded"] = df_pandas["A_t"].map(action_mapping)

    # Fit the model
    calibrated_clf.fit(df_pandas[x_columns], df_pandas["A_t_encoded"])

    # Get predicted probabilities for all possible treatments for each observation
    # predict_proba returns probabilities in the order of classes in the CalibratedClassifierCV
    # Reconstruct original class order for clarity
    calibrated_propensity_classes = calibrated_clf.classes_

    # Convert back to original treatment labels ensuring that the order is correct
    reordered_propensity_classes = [
        actions[idx] for idx in calibrated_propensity_classes
    ]

    # Predict the proba scores
    predicted_probs_array = calibrated_clf.predict_proba(df_pandas[x_columns])

    # Add predicted probabilities to the main DataFrame
    predicted_probs_df_pl = pl.DataFrame(
        predicted_probs_array,
        schema=[
            f"{c}" for c in reordered_propensity_classes
        ],  # Use the reordered original treatment names
    )

    return predicted_probs_df_pl


# --- Step 2.1: Incorporate Eligibility and Re-normalize ---
def calculate_adjusted_treatment_probabilities(population: Population) -> pl.DataFrame:
    """
    Adjusts propensity scores based on eligibility and re-normalizes.
    """
    treatment_probabilities = calculate_treatment_probabilities(population)
    eligibility_columns = population._state[population.eligibility_columns]

    actions = population.action_hierarchy.get_leaves() + ["NoAction"]

    # Calculate the adjusted treatment probabilities by multiplying by the eligibility mask and renormalizing
    for column in actions:
        treatment_probabilities = treatment_probabilities.with_columns(
            (
                treatment_probabilities[column]
                * eligibility_columns[f"eligibility_{column}"]
            ).alias(column)
        )

    row_sum = treatment_probabilities.sum_horizontal().fill_null(1.0)

    treatment_probabilities = treatment_probabilities.with_columns(
        treatment_probabilities[col] / row_sum
        for col in treatment_probabilities.columns
    )

    return treatment_probabilities


def calculate_marginal_probabilities(series: pl.DataFrame) -> pl.Series:
    """
    Calculates the marginal probability (relative count) of each unique value in a Polars Series.

    Args:
        series: The input Polars Series.

    Returns:
        A dictionary of frequencies per category
    """
    # Calculate value counts
    counts = series.group_by(series).len()

    # Calculate total number of elements
    total_elements = len(series)

    results = counts.with_columns(counts["len"] / total_elements)

    return {key: value for key, value in zip(results["A_t"], results["len"])}


def calculate_truncated_stabilized_ipw(
    population: Population,
    truncation_lower_bound: float = 0.01,  # e.g., 1st percentile or 0.01
    truncation_upper_bound: float = 0.85,  # e.g., 99th percentile or 0.99
) -> pl.DataFrame:
    """
    Calculates truncated and stabilized inverse propensity weights (IPW).

    Args:
        population (Population): The Population object containing the state
                                 with observed actions ('A_t').
        truncation_lower_bound (float): The lower percentile (e.g., 0.01 for 1st percentile)
                                        or a fixed minimum value for truncating weights.
                                        If 0 < truncation_lower_bound < 1, it's treated as a percentile.
                                        Otherwise, it's a fixed lower bound for the weights.
        truncation_upper_bound (float): The upper percentile (e.g., 0.99 for 99th percentile)
                                        or a fixed maximum value for truncating weights.
                                        If 0 < truncation_upper_bound < 1, it's treated as a percentile.
                                        Otherwise, it's a fixed upper bound for the weights.

    Returns:
        pl.DataFrame: A Polars DataFrame with a single column 'IPW' containing
                      the truncated and stabilized inverse propensity weights.
    """
    observed_actions = population._state.select(pl.col("A_t"))

    # Calculate marginal probabilities of each action 'a'
    marginal_prob_map = calculate_marginal_probabilities(observed_actions)

    # Calculate the treatment probabilities
    treatment_probabilities = calculate_adjusted_treatment_probabilities(population)

    # Stabilize the calculated treatment probabilites by dividing by the marginal probabilities of each treatment
    stabilized_treatment_probabilities = treatment_probabilities.with_columns(
        treatment_probabilities[col] / marginal_prob_map[col]
        for col in treatment_probabilities.columns
    )

    # Calculate the clipped IPW by doing 1/probability
    ipw = stabilized_treatment_probabilities.with_columns(
        1 / stabilized_treatment_probabilities[col].clip(1e-6, 1e6)
        for col in stabilized_treatment_probabilities.columns
    )

    # Apply a clipping to the IPW to truncate to high or low values
    expressions = []
    for column in ipw.columns:
        # Calculate the 1st and 99th quantiles as expressions
        lower_bound = ipw[column].quantile(truncation_lower_bound)
        upper_bound = ipw[column].quantile(truncation_upper_bound)

        # Clip the column using these bounds and alias it back to the original column name
        expressions.append(ipw[column].clip(lower_bound, upper_bound))

    # Apply the clipping to the DataFrame
    clipped_ipw = ipw.with_columns(expressions)

    return clipped_ipw


def lookup_ipw_from_predicted(
    predicted_ipw: pl.DataFrame,
    action: pl.Series,
) -> pl.Series:
    """
    Performs a row-wise lookup from columns in DataFrame A based on values in a Polars Series.
    It's assumed that df_A and choice_series are implicitly aligned by row index.

    Args:
        df_A: The source DataFrame containing numerical columns. Its column names
              should correspond to the values in choice_series.
        choice_series: A Polars Series containing categorical entries (strings)
                       that match column names in df_A.
        choice_series_name: A temporary name given to the choice_series when
                            it's horizontally concatenated with df_A.
        output_series_name: The desired name for the resulting looked-up Polars Series.

    Returns:
        A new Polars Series containing the looked-up numerical values.

    Raises:
        ValueError: If df_A and choice_series do not have the same length.
        ValueError: If df_A has no columns, as there's nothing to look up.
    """

    output_series_name: str = "ipw"
    choice_series_name: str = "choice_column_for_lookup"

    if predicted_ipw.height != len(action):
        raise ValueError(
            f"DataFrame A (height={predicted_ipw.height}) and choice_series (length={len(action)}) "
            "must have the same number of rows/elements for row-wise lookup."
        )

    if not predicted_ipw.columns:
        # If df_A has no columns, there's nothing to look up, return a Series of nulls.
        return pl.Series(
            name=output_series_name, values=[None] * len(action), dtype=pl.Float64
        )

    # 1. Convert the choice_series into a temporary DataFrame for horizontal concatenation.
    choice_df_temp = action.to_frame(name=choice_series_name)

    # 2. Horizontally concatenate df_A and the temporary choice_df_temp.
    # This brings all relevant columns into a single context for expression evaluation.
    # The implicit row alignment is preserved here.
    combined_df = pl.concat([predicted_ipw, choice_df_temp], how="horizontal")

    # 3. Build the conditional expression dynamically.
    # Iterate through the column names of df_A to create the when/then chain.
    conditional_expression = None
    for column in predicted_ipw.columns:
        if conditional_expression is None:
            # Initialize the expression with the first condition
            conditional_expression = pl.when(pl.col(choice_series_name) == column).then(
                pl.col(column)
            )
        else:
            # Chain subsequent conditions
            conditional_expression = conditional_expression.when(
                pl.col(choice_series_name) == column
            ).then(pl.col(column))

    # 4. Apply the conditional expression to the combined DataFrame to compute the new column.
    # Then select just this column and convert it to a Series.
    result_series = (
        combined_df.with_columns(conditional_expression.alias(output_series_name))
        .select(pl.col(output_series_name))
        .to_series()
    )

    return result_series
