import polars as pl


def arg_max_horizontal(*columns: pl.Expr) -> pl.Expr:
    """
    Polars expression to find the column name with the maximum value horizontally.
    Assumes columns are named after the actions they represent.
    This version uses replace_strict for a more direct mapping of index to column name.

    Examples
    --------
    >>> import polars as pl
    >>> test_df = pl.DataFrame({
    ...     "row_id": [1, 2, 3, 4, 5, 6],
    ...     "col_A": [10, 50, 30, 70, 20, 10],
    ...     "col_B": [20, 10, 40, 50, 80, 10],
    ...     "col_C": [30, 20, 10, 60, 40, 50],
    ...     "col_D": [15, 60, 25, 40, 90, 50],
    ... })
    >>>
    >>> # Apply the expression to find the column with the highest value per row
    >>> result = test_df.select(
    ...     arg_max_horizontal(pl.col("col_A"), pl.col("col_B"), pl.col("col_C"), pl.col("col_D"))
    ... )
    >>> print(result)
    shape: (6, 1)
    ┌───────────────┐
    │ chosen_action │
    │ str           │
    ╞═══════════════╡
    │ col_C         │
    │ col_D         │
    │ col_B         │
    │ col_A         │
    │ col_D         │
    │ col_C         │
    └───────────────┘
    """

    return (
        pl.concat_list(columns)
        .list.arg_max()
        .replace_strict({i: col_name for i, col_name in enumerate(columns)})
    )
