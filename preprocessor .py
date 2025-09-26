from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, DoubleType
from functools import reduce

class SparkFeaturePreprocessor:
    """
    A class for preprocessing a PySpark DataFrame, including imputation,
    category merging, whitelisting, and handling of infrequent categories.
    """
    def __init__(
        self,
        impute_config: dict = None,
        merge_config: dict = None,
        whitelist_config: dict = None,
        frequency_config: dict = None,
        final_cleanup_config: dict = None,
        target_config: dict = None,
    ):
        """
        Initializes the preprocessor with configuration dictionaries.
        """
        self.impute_config = impute_config
        self.merge_config = merge_config
        self.whitelist_config = whitelist_config
        self.frequency_config = frequency_config
        self.final_cleanup_config = final_cleanup_config
        self.target_config = target_config
        
        # Attributes to be learned during fitting
        self.infrequent_levels_ = {}
        self.categories_to_remove_ = {}

    def _impute(self, df: DataFrame) -> DataFrame:
        """
        Imputes missing values for numerical and categorical columns.
        """
        if not self.impute_config:
            return df

        processed_df = df
        
        numerical_cols = self.impute_config.get('numerical', [])
        for col_name in numerical_cols:
            if col_name in processed_df.columns:
                processed_df = processed_df.withColumn(
                    f"{col_name}_isnull", F.when(F.col(col_name).isNull(), 1).otherwise(0)
                ).fillna(0, subset=[col_name])

        categorical_cols = self.impute_config.get('categorical', [])
        for col_name in categorical_cols:
            if col_name in processed_df.columns:
                processed_df = processed_df.withColumn(
                    f"{col_name}_isnull", F.when(F.col(col_name).isNull(), 1).otherwise(0)
                ).fillna('Other/Unknown', subset=[col_name])
                
        return processed_df

    def _merge_categories(self, df: DataFrame) -> DataFrame:
        """
        Merges categorical values based on the merge_config.
        """
        if not self.merge_config:
            return df
        
        processed_df = df
        for col_name, merge_map in self.merge_config.items():
            if col_name in processed_df.columns:
                mapping_expr = reduce(
                    lambda acc, item: acc.when(F.col(col_name).isin(item[1]), item[0]),
                    merge_map.items(),
                    F
                )
                processed_df = processed_df.withColumn(
                    col_name,
                    mapping_expr.otherwise(F.col(col_name))
                )
        return processed_df

    def _apply_whitelist(self, df: DataFrame) -> DataFrame:
        """
        Merges non-whitelisted categories into a single category.
        """
        if not self.whitelist_config:
            return df
        
        processed_df = df
        for col_name, (target_name, labels_to_keep) in self.whitelist_config.items():
            if col_name in processed_df.columns:
                processed_df = processed_df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isin(labels_to_keep), F.col(col_name)).otherwise(target_name)
                )
        return processed_df
    
    def _create_targets(self, df: DataFrame) -> DataFrame:
        """
        Creates binary target columns.
        """
        if not self.target_config:
            return df

        processed_df = df
        for col_name, categories in self.target_config.items():
            for category in categories:
                binary_col_name = f"target_{category.lower().replace(' ', '_')}"
                processed_df = processed_df.withColumn(
                    binary_col_name,
                    F.when(F.col(col_name) == category, 1).otherwise(0).cast("integer")
                )
        return processed_df
    
    def _convert_decimals_to_doubles(self, df: DataFrame) -> DataFrame:
        """
        Converts all DecimalType columns to DoubleType.
        """
        decimal_columns = [
            field.name for field in df.schema.fields if isinstance(field.dataType, DecimalType)
        ]
        
        if not decimal_columns:
            return df
            
        return reduce(
            lambda current_df, col_name: current_df.withColumn(col_name, F.col(col_name).cast(DoubleType())),
            decimal_columns,
            df
        )

    def fit(self, df: DataFrame):
        """
        Learns the infrequent levels and categories to remove from the DataFrame.
        """
        print("Fitting the preprocessor...")
        
        # Fit for low-frequency category capping
        if self.frequency_config:
            freq_threshold = self.frequency_config.get('threshold', 2)
            freq_cols = self.frequency_config.get('columns', [])
            for col_name in freq_cols:
                if col_name in df.columns:
                    counts = df.groupBy(col_name).count()
                    self.infrequent_levels_[col_name] = [
                        row[col_name] for row in counts.filter(F.col("count") < freq_threshold).collect()
                    ]

        # Fit for final cleanup
        if self.final_cleanup_config:
            cleanup_threshold = self.final_cleanup_config.get('threshold', 2)
            cleanup_cols = self.final_cleanup_config.get('columns', [])
            for col_name in cleanup_cols:
                if col_name in df.columns:
                    final_counts = df.groupBy(col_name).count()
                    self.categories_to_remove_[col_name] = [
                        row[col_name] for row in final_counts.filter(F.col("count") < cleanup_threshold).collect()
                    ]
        
        print("Fitting complete.")
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Applies all the learned transformations to the DataFrame.
        """
        print("Transforming the DataFrame...")
        processed_df = df

        # Apply transformations that don't require fitting
        processed_df = self._impute(processed_df)
        processed_df = self._merge_categories(processed_df)
        processed_df = self._apply_whitelist(processed_df)
        
        # Apply transformations based on fitted parameters
        if self.frequency_config:
            for col_name, infrequent_levels in self.infrequent_levels_.items():
                if infrequent_levels:
                    processed_df = processed_df.withColumn(
                        col_name,
                        F.when(F.col(col_name).isin(infrequent_levels), 'Other/Unknown')
                         .otherwise(F.col(col_name))
                    )

        if self.final_cleanup_config:
            for col_name, categories_to_remove in self.categories_to_remove_.items():
                if categories_to_remove:
                    initial_rows = processed_df.count()
                    processed_df = processed_df.filter(~F.col(col_name).isin(categories_to_remove))
                    print(f"Removed {initial_rows - processed_df.count()} rows based on rare categories in '{col_name}'.")

        # Apply final transformations
        processed_df = self._create_targets(processed_df)
        processed_df = self._convert_decimals_to_doubles(processed_df)

        print("Transformation complete.")
        return processed_df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fits the preprocessor and then transforms the DataFrame.
        """
        self.fit(df)
        return self.transform(df)
