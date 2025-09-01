from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Paste the full_feature_engineering_pipeline function here...
def full_feature_engineering_pipeline(
    df: DataFrame,
    impute_config: dict,
    merge_config: dict,
    frequency_config: dict,
    final_cleanup_config: dict = None
) -> DataFrame:
    """
    Applies a full series of feature engineering steps to a DataFrame.
    ... (rest of the function code) ...
    """
    print("🚀 Starting full feature engineering pipeline...")
    processed_df = df

    # --- Steps 1 & 2 (from previous function) ---
    # (Code for null imputation and category merging remains the same)
    numerical_cols = impute_config.get('numerical', [])
    for col_name in numerical_cols:
        if col_name in processed_df.columns:
            processed_df = processed_df.withColumn(
                f"{col_name}_isnull", F.when(F.col(col_name).isNull(), 1).otherwise(0)
            ).fillna(0, subset=[col_name])

    categorical_cols = impute_config.get('categorical', [])
    for col_name in categorical_cols:
        if col_name in processed_df.columns:
            processed_df = processed_df.withColumn(
                f"{col_name}_isnull", F.when(F.col(col_name).isNull(), 1).otherwise(0)
            ).fillna('Other/Unknown', subset=[col_name])

    for col_name, merge_map in merge_config.items():
        if col_name in processed_df.columns:
            mapping_expr = None
            for target_category, source_categories in merge_map.items():
                mapping_expr = (
                    F.when(F.col(col_name).isin(source_categories), target_category)
                    if mapping_expr is None
                    else mapping_expr.when(F.col(col_name).isin(source_categories), target_category)
                )
            if mapping_expr is not None:
                processed_df = processed_df.withColumn(col_name, mapping_expr.otherwise(F.col(col_name)))

    # --- 3. Handle Low-Frequency Category Capping ---
    freq_threshold = frequency_config.get('threshold', 2) # Default to threshold of 2
    freq_cols = frequency_config.get('columns', [])
    for col_name in freq_cols:
        if col_name in processed_df.columns:
            print(f"  - Capping infrequent categories in '{col_name}' (threshold: {freq_threshold})")

            # Find levels that are infrequent
            counts = processed_df.groupBy(col_name).count()
            infrequent_levels = counts.filter(F.col("count") < freq_threshold)\
                                      .select(col_name)\
                                      .rdd.flatMap(lambda x: x)\
                                      .collect()

            if infrequent_levels:
                # Replace the infrequent levels
                processed_df = processed_df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isin(infrequent_levels), 'Other/Unknown')
                     .otherwise(F.col(col_name))
                )

    # --- 4. Final Cleanup: Remove Rows with Rare Categories (Optional) ---
    if final_cleanup_config:
        cleanup_threshold = final_cleanup_config.get('threshold', 2)
        cleanup_cols = final_cleanup_config.get('columns', [])
        for col_name in cleanup_cols:
            if col_name in processed_df.columns:
                print(f"  - Final cleanup on '{col_name}', removing categories with count < {cleanup_threshold}")

                # Recalculate counts on the potentially modified column
                final_counts = processed_df.groupBy(col_name).count()

                # Find categories that are STILL too rare
                categories_to_remove = final_counts.filter(F.col("count") < cleanup_threshold)\
                                                   .select(col_name)\
                                                   .rdd.flatMap(lambda x: x)\
                                                   .collect()

                if categories_to_remove:
                    print(f"    - Found rare categories to remove: {categories_to_remove}")
                    initial_rows = processed_df.count()
                    processed_df = processed_df.filter(~F.col(col_name).isin(categories_to_remove))
                    print(f"    - Removed {initial_rows - processed_df.count()} rows.")

    print("✅ Pipeline finished.")
    return processed_df
