# ./simulation/action_transformer.py

import polars as pl
import warnings
from .hierarchy import ActionHierarchy

# For now, let's keep the transformed action prefix blank; the columns are thus names after the leaf nodes of the action hierarchy
prefix = ""


def transform_actions(
    actions_series: pl.Series, action_hierarchy: ActionHierarchy
) -> pl.DataFrame:
    """
    Transforms a Series of chosen action names into a DataFrame of binary
    features representing all active nodes in the action hierarchy for each customer.

    An action hierarchy node A_j is active for a chosen action A_chosen (which
    must be one of the keys in action_hierarchy.costs) if A_chosen is a
    descendant of A_j in the hierarchy (including A_j itself).
    This is equivalent to checking if A_j is an ancestor of A_chosen.

    The columns for the root node(s) and the 'NoAction' node are EXCLUDED
    from the output DataFrame. 'NoAction' is implicitly represented by a row
    of all zeros in the remaining columns (if 'NoAction' is an actionable node).

    Args:
        actions_series: A Polars Series where each row is the chosen action name.
                        Shape (|N|,). Assumes values are potentially valid keys in action_hierarchy.costs
                        or 'NoAction' or other strings.
        action_hierarchy: The ActionHierarchy object.

    Returns:
        A Polars DataFrame. Shape (|N|, |ActionHierarchyNodes| - |Roots| - |NoAction node|).
        Columns correspond to nodes in the hierarchy (excluding root(s) and NoAction),
        prefixed with '{prefix}'. Value is 1 if the node is active, 0 otherwise.
    """
    # Get the root node(s). Assuming single root or treating all as roots to exclude.
    root_nodes_to_exclude = action_hierarchy.get_roots()

    # Identify the 'NoAction' node name to exclude if it exists and is actionable
    noaction_node_to_exclude = "NoAction"

    # Get all possible nodes in the action hierarchy
    all_hierarchy_nodes = action_hierarchy.get_all_nodes()

    # Determine which nodes should become columns in the output DataFrame
    output_nodes = sorted(
        [
            node
            for node in all_hierarchy_nodes
            if node not in root_nodes_to_exclude and node != noaction_node_to_exclude
        ]
    )

    # If after removing nodes, no columns are left, handle empty DF case
    if not output_nodes or actions_series.len() == 0:
        column_order = [
            f"{prefix}{node}" for node in output_nodes
        ]  # Still define potential columns
        schema = {col: pl.UInt8 for col in column_order}
        return pl.DataFrame(schema=schema)

    # Ensure the input series has a name for use in expressions
    input_series_name = actions_series.name
    if input_series_name is None:
        input_series_name = "chosen_action_input"
        actions_series = actions_series.alias(input_series_name)
    # else: use existing actions_series.name

    # Get actionable nodes (those with defined costs). These are the actual actions that can be "chosen".
    actionable_nodes = set(action_hierarchy.costs.keys())

    # --- Input Validation ---
    # Filter the series directly using Series methods to find invalid actions
    invalid_actions_series = actions_series.filter(
        ~actions_series.is_in(list(actionable_nodes))
    )
    invalid_actions = invalid_actions_series.unique().to_list()

    if invalid_actions:
        invalid_actions_clean = [
            action for action in invalid_actions if action is not None
        ]
        if invalid_actions_clean:
            warnings.warn(
                f"Input series contains action names not defined in action_hierarchy.costs: {invalid_actions_clean}. Rows with these actions will result in all zero features in the output columns."
            )

    # Pre-calculate descendants for all hierarchy nodes.
    # We need this to check if the *chosen* action is a descendant of a given node (which means the node is active).
    # This calculation must include root/NoAction nodes if they are in the hierarchy, as chosen actions
    # might be descendants of them.
    node_descendants_map = {
        node: action_hierarchy.get_descendants(node)
        for node in all_hierarchy_nodes  # Use all nodes for this calculation
    }

    # Create expressions for each node that should become an output column
    expressions = []
    # Iterate *only* over the nodes we want as output columns
    for node in output_nodes:
        # Get the set of descendants for this 'node' from the pre-calculated map
        node_descendants = node_descendants_map.get(node, set())  # Use .get for safety

        # Filter these descendants to include only those that are actionable (can be chosen)
        descendant_actionable_actions = [
            d for d in node_descendants if d in actionable_nodes
        ]

        # The column value for 'A_node' is 1 if the chosen action (value in actions_series for a row)
        # is present in the list of `descendant_actionable_actions` for this 'node'.
        expr = (
            pl.col(input_series_name)
            .is_in(descendant_actionable_actions)
            .cast(pl.UInt8)  # Cast boolean to 0/1
            .alias(f"{prefix}{node}")
        )
        expressions.append(expr)

    # Apply all expressions to the input series by wrapping it in a DataFrame
    actions_df_temp = pl.DataFrame({input_series_name: actions_series})

    # Use select with the defined expressions. This creates the new columns
    # and naturally excludes the original series column and any other columns
    # not included in the expressions list.
    hierarchical_actions_df = actions_df_temp.select(expressions)

    # Ensure columns are in the sorted order of output_nodes for consistency.
    column_order = [f"{prefix}{node}" for node in output_nodes]

    # Check if the generated columns match the expected ones.
    if set(hierarchical_actions_df.columns) != set(column_order):
        missing_cols = set(column_order) - set(hierarchical_actions_df.columns)
        extra_cols = set(hierarchical_actions_df.columns) - set(column_order)
        warnings.warn(
            f"Generated hierarchical columns do not match expected. Missing: {missing_cols}, Extra: {extra_cols}"
        )

    # Reorder columns using select to match the consistent order.
    # This will also error if a column is in column_order but wasn't generated, which is good.
    hierarchical_actions_df = hierarchical_actions_df.select(column_order)

    return hierarchical_actions_df.cast(pl.UInt8)  # Final cast to binary


def transform_actions_to_series(
    hierarchical_actions_df: pl.DataFrame, action_hierarchy: ActionHierarchy
) -> pl.Series:
    """
    Transforms a DataFrame of binary hierarchical action features (EXCLUDING
    root and NoAction columns) back into a Series of chosen action names.

    It infers the chosen action based on the pattern of activated nodes in the
    provided columns. If a row is all zeros across the provided actionable node
    columns, it is mapped to 'NoAction', provided 'NoAction' is defined as
    an actionable node (has a cost). Otherwise, it maps to the unique actionable
    node whose column is 1, prioritizing alphabetically if multiple match.
    If no matching actionable node is found (e.g., row of zeros but NoAction
    not actionable, or pattern doesn't match any single actionable node),
    the result is None for that row.

    Args:
        hierarchical_actions_df: A Polars DataFrame. Shape (|N|, |Subset of HierarchyNodes|).
                                 Columns correspond to nodes (excluding root(s) and NoAction),
                                 prefixed with '{prefix}'. Values are binary (0 or 1).
        action_hierarchy: The ActionHierarchy object.

    Returns:
        A Polars Series. Shape (|N|,). Values are the original chosen action names (strings)
        or None if a unique actionable action cannot be determined for a row based on the remaining columns.
    """
    # Use .height to get the number of rows in a DataFrame
    if hierarchical_actions_df.height == 0:
        return pl.Series("chosen_action", [], dtype=pl.String)

    # Get all nodes considered actionable (those with defined costs)
    actionable_nodes = sorted(list(action_hierarchy.costs.keys()))

    # Identify the 'NoAction' node name
    noaction_node = "NoAction"

    # Identify the actionable nodes *other than* NoAction
    other_actionable_nodes = [
        node for node in actionable_nodes if node != noaction_node
    ]

    # Identify the columns in the input DataFrame that correspond to these other actionable nodes
    other_actionable_cols_in_df = [
        f"{prefix}{node}"
        for node in other_actionable_nodes
        if f"{prefix}{node}" in hierarchical_actions_df.columns
    ]

    # Build the core expression chain
    # Default value if no action is matched
    reverse_expr = pl.lit(None, dtype=pl.String)

    # --- Logic for inferring 'NoAction' ---
    # If 'NoAction' is an actionable node (i.e., has a cost) AND
    # if all columns corresponding to *other* actionable nodes in the DF are 0.
    # Summing the relevant columns is an efficient way to check if they are all zero.
    # Start with a false condition, will update if NoAction is actionable.
    is_noaction_condition = pl.lit(False)

    if (
        noaction_node in actionable_nodes
    ):  # Can only map to NoAction if it's a defined actionable action
        if other_actionable_cols_in_df:
            # Sum the columns corresponding to other actionable nodes present in the DF
            sum_of_other_actionable_cols = sum(
                pl.col(c) for c in other_actionable_cols_in_df
            )
            # The condition is when this sum is zero
            is_noaction_condition = sum_of_other_actionable_cols == 0
        elif hierarchical_actions_df.width > 0:
            # If there are no other actionable columns in the DF, but the DF has columns,
            # check if the entire row is zero by summing all columns present.
            sum_all_cols_in_df = sum(pl.col(c) for c in hierarchical_actions_df.columns)
            is_noaction_condition = sum_all_cols_in_df == 0
        else:
            # If the DataFrame has no columns (width is 0), then the 'sum' is implicitly 0.
            # If NoAction is actionable, and there are no other actionable columns to check,
            # assume an empty/all-zero row maps to NoAction. This handles edge cases like
            # only NoAction being actionable and its column was removed.
            is_noaction_condition = pl.lit(
                True
            )  # Row is effectively all zeros relevant to other actions

        # If the condition is met, map to 'NoAction'. This is the highest priority mapping.
        reverse_expr = (
            pl.when(is_noaction_condition)
            .then(pl.lit(noaction_node))
            .otherwise(reverse_expr)
        )

    # --- Logic for mapping to other actionable nodes ---
    # Iterate through the other actionable nodes whose columns are present in the DataFrame
    # Process them in sorted alphabetical order for deterministic mapping
    mappable_other_actionable_nodes = sorted(
        [
            node
            for node in other_actionable_nodes
            if f"{prefix}{node}" in hierarchical_actions_df.columns
        ]
    )

    for action_name in mappable_other_actionable_nodes:
        col_name = f"{prefix}{action_name}"
        # Add a condition: if this action's column is 1, map to this action name
        # This is chained after the NoAction check
        reverse_expr = (
            pl.when(pl.col(col_name) == 1)
            .then(pl.lit(action_name))
            .otherwise(reverse_expr)
        )

    # Apply the final expression to get the series of chosen actions
    chosen_action_series = hierarchical_actions_df.select(
        reverse_expr.alias("chosen_action")
    )["chosen_action"]

    # Check for rows that couldn't be mapped (still None)
    if chosen_action_series.is_null().any():
        num_unmapped = chosen_action_series.is_null().sum()
        warnings.warn(
            f"{num_unmapped} row(s) could not be mapped back to a chosen action (result is None). Check input DataFrame structure or hierarchy/costs configuration. This may happen if the row's pattern of 0s/1s doesn't match the expected pattern for any single actionable node (e.g., multiple actionable leaves are marked 1, or the row is all zeros when NoAction is not actionable)."
        )

    return chosen_action_series
