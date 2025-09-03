import pickle
import os
import shutil
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.window import Window
from functools import reduce

# Mock MLFlow for demonstration purposes if not installed.
# In a real environment, you would `import mlflow`.
class MockMLFlow:
    """A mock MLFlow class to simulate saving and loading artifacts."""
    def __init__(self):
        self._run_id = None
        self._artifact_path = "/tmp/mlflow_artifacts"
        if os.path.exists(self._artifact_path):
            shutil.rmtree(self._artifact_path)
        os.makedirs(self._artifact_path)

    def start_run(self):
        self._run_id = "run_" + os.urandom(6).hex()
        print(f"\n--- MLFlow: Starting Run ({self._run_id}) ---")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"--- MLFlow: Ending Run ({self._run_id}) ---\n")
        self._run_id = None

    def log_artifact(self, local_path, artifact_path=None):
        dest_path = os.path.join(self._artifact_path, artifact_path or os.path.basename(local_path))
        shutil.copy(local_path, dest_path)
        print(f"MLFlow: Logged '{local_path}' to '{dest_path}'")

    def download_artifacts(self, artifact_path):
        local_dir = f"/tmp/downloaded_{artifact_path}"
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir)
        
        src_path = os.path.join(self._artifact_path, artifact_path)
        shutil.copy(src_path, local_dir)
        print(f"MLFlow: Downloaded '{artifact_path}' to '{local_dir}'")
        return os.path.join(local_dir, artifact_path)

mlflow = MockMLFlow()

class SparkFeaturePreprocessor:
    """
    A stateful preprocessor for PySpark DataFrames to handle categorical
    and numerical features, designed for evolving datasets and MLFlow integration.
    
    Manages two types of transformations:
    1.  Categorical: Label encoding (String -> Integer). Can be configured to either
        extend with new categories or raise an error.
    2.  Numerical: Standard scaling (mean=0, std=1).
    """

    def __init__(self, numerical_cols, categorical_cols, extendable_cols=None):
        """
        Initializes the preprocessor.

        Args:
            numerical_cols (list[str]): List of numerical column names.
            categorical_cols (list[str]): List of categorical column names.
            extendable_cols (list[str], optional): List of categorical columns that can be
                                                  extended with new categories during transform.
                                                  Defaults to [].
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.extendable_cols = extendable_cols or []
        
        # State dictionaries
        self.numerical_scalers = {}  # {col_name: {'mean': float, 'stddev': float}}
        self.categorical_maps = {}   # {col_name: {'mapping': {str: int}, 'next_id': int}}

        # Validate that extendable_cols are a subset of categorical_cols
        if not set(self.extendable_cols).issubset(set(self.categorical_cols)):
            raise ValueError("extendable_cols must be a subset of categorical_cols")

    def fit(self, df: SparkSession.builder.getOrCreate().createDataFrame([], schema="id INT")):
        """
        Learns the scaling parameters and category mappings from the data.
        This is used for a "full retrain".

        Args:
            df (pyspark.sql.DataFrame): The training DataFrame.
        """
        print("Fitting preprocessor...")
        
        # 1. Learn parameters for numerical columns
        if self.numerical_cols:
            agg_exprs = []
            for col_name in self.numerical_cols:
                agg_exprs.append(F.mean(col_name).alias(f"{col_name}_mean"))
                agg_exprs.append(F.stddev(col_name).alias(f"{col_name}_stddev"))
            
            stats_row = df.agg(*agg_exprs).first()
            
            for col_name in self.numerical_cols:
                mean_val = stats_row[f"{col_name}_mean"]
                stddev_val = stats_row[f"{col_name}_stddev"]
                
                self.numerical_scalers[col_name] = {
                    'mean': mean_val,
                    # Handle zero standard deviation to prevent division by zero
                    'stddev': stddev_val if stddev_val > 0 else 1.0
                }
        
        # 2. Learn mappings for categorical columns
        for col_name in self.categorical_cols:
            # Collect distinct categories and create an integer mapping
            distinct_cats = df.select(col_name).distinct().collect()
            mapping = {row[col_name]: i for i, row in enumerate(distinct_cats)}
            
            self.categorical_maps[col_name] = {
                'mapping': mapping,
                'next_id': len(mapping)
            }
        print("Fit complete.")
        return self

    def transform(self, df: SparkSession.builder.getOrCreate().createDataFrame([], schema="id INT")):
        """
        Applies the learned transformations to a DataFrame.
        Handles new categories based on the 'extendable_cols' configuration.

        Args:
            df (pyspark.sql.DataFrame): The DataFrame to transform.

        Returns:
            pyspark.sql.DataFrame: The transformed DataFrame.
        """
        print("Transforming data...")
        
        # 1. Pre-computation: Handle new categories
        self._update_mappings_for_new_categories(df)
        
        transformed_df = df
        
        # 2. Apply numerical scaling
        for col_name, params in self.numerical_scalers.items():
            mean = params['mean']
            stddev = params['stddev']
            transformed_df = transformed_df.withColumn(
                col_name,
                (F.col(col_name) - mean) / stddev
            )
        
        # 3. Apply categorical encoding using efficient map lookups
        for col_name, state in self.categorical_maps.items():
            # Create a PySpark map literal from the Python dictionary
            mapping_expr = F.create_map([F.lit(x) for x in state['mapping'].items()])
            
            # Apply the mapping
            transformed_df = transformed_df.withColumn(
                col_name,
                mapping_expr[F.col(col_name)]
            )
        
        print("Transform complete.")
        return transformed_df

    def _update_mappings_for_new_categories(self, df):
        """
        Scans for new categories and updates mappings or raises errors accordingly.
        This stateful update happens on the driver before the distributed transform.
        """
        spark = SparkSession.builder.getOrCreate()
        
        for col_name in self.categorical_cols:
            if not self.categorical_maps.get(col_name):
                raise RuntimeError(f"Preprocessor not fitted for categorical column '{col_name}'.")

            # Get the new categories present in the DataFrame but not in our mapping
            current_mapping = self.categorical_maps[col_name]['mapping']
            known_cats_df = spark.createDataFrame(
                [(k,) for k in current_mapping.keys()],
                [col_name]
            )
            
            new_cats_df = df.select(col_name).distinct() \
                            .join(known_cats_df, on=col_name, how="left_anti")
            
            new_cats = [row[col_name] for row in new_cats_df.collect()]

            if new_cats:
                if col_name in self.extendable_cols:
                    # Extend the mapping for this column
                    print(f"Found new categories for extendable column '{col_name}': {new_cats}")
                    next_id = self.categorical_maps[col_name]['next_id']
                    for cat in new_cats:
                        if cat not in self.categorical_maps[col_name]['mapping']:
                            print(f"  -> Extending mapping for '{col_name}': '{cat}' -> {next_id}")
                            self.categorical_maps[col_name]['mapping'][cat] = next_id
                            next_id += 1
                    self.categorical_maps[col_name]['next_id'] = next_id
                else:
                    # This column is strict, so raise an error
                    raise ValueError(
                        f"Found unexpected new categories in 'strict' column '{col_name}': {new_cats}. "
                        "This may indicate a data quality issue."
                    )
    
    def save(self, file_path: str):
        """Saves the preprocessor state to a file using pickle."""
        print(f"Saving preprocessor to '{file_path}'...")
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print("Save complete.")

    @classmethod
    def load(cls, file_path: str):
        """Loads a preprocessor state from a file."""
        print(f"Loading preprocessor from '{file_path}'...")
        with open(file_path, 'rb') as f:
            instance = pickle.load(f)
        print("Load complete.")
        return instance


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("SparkFeaturePreprocessorDemo") \
        .master("local[*]") \
        .getOrCreate()

    # --- Configuration ---
    NUMERICAL_COLS = ['customer_age', 'avg_spend']
    CATEGORICAL_COLS = ['banner_id', 'customer_country']
    # banner_id can be extended, customer_country cannot.
    EXTENDABLE_COLS = ['banner_id']
    
    PREPROCESSOR_ARTIFACT_NAME = "feature_preprocessor.pkl"

    # =========================================================================
    # Use Case 1: Full Model Retraining
    # =========================================================================
    print("="*50)
    print(" Use Case 1: Full Model Retraining")
    print("="*50)

    # 1.1. Initial training data
    train_data = [
        ('A', 'USA', 25, 120.50), ('B', 'UK', 45, 88.0),
        ('A', 'UK', 31, 250.0), ('C', 'USA', 22, 94.20),
        ('B', 'USA', 51, 150.75), ('C', 'CAN', 38, 310.0),
    ]
    train_df = spark.createDataFrame(train_data, CATEGORICAL_COLS + NUMERICAL_COLS)
    
    # 1.2. Create and fit the preprocessor
    preprocessor = SparkFeaturePreprocessor(
        numerical_cols=NUMERICAL_COLS,
        categorical_cols=CATEGORICAL_COLS,
        extendable_cols=EXTENDABLE_COLS
    )
    preprocessor.fit(train_df)
    
    # 1.3. Transform the training data
    transformed_train_df = preprocessor.transform(train_df)
    print("\nTransformed training data:")
    transformed_train_df.show()
    
    print("\nLearned Numerical Scalers:")
    print(preprocessor.numerical_scalers)
    print("\nLearned Categorical Mappings:")
    print(preprocessor.categorical_maps)

    # 1.4. Save the preprocessor using MLFlow
    with mlflow.start_run() as run:
        temp_path = PREPROCESSOR_ARTIFACT_NAME
        preprocessor.save(temp_path)
        mlflow.log_artifact(temp_path, artifact_path="preprocessor")
        os.remove(temp_path)

    # =========================================================================
    # Use Case 2: Applying the Model (Inference) with Strict Checking
    # =========================================================================
    print("\n" + "="*50)
    print(" Use Case 2: Applying the Model (Inference)")
    print("="*50)

    # 2.1. New data for inference, including an unknown country ('GER')
    inference_data = [
        ('A', 'USA', 33, 150.0), # Known categories
        ('C', 'GER', 40, 200.0)  # New 'GER' for a strict column
    ]
    inference_df = spark.createDataFrame(inference_data, CATEGORICAL_COLS + NUMERICAL_COLS)

    # 2.2. Load the preprocessor from MLFlow and try to transform
    try:
        # In a real app, this path comes from your MLFlow server/registry
        loaded_artifact_path = mlflow.download_artifacts(f"preprocessor/{PREPROCESSOR_ARTIFACT_NAME}")
        loaded_preprocessor = SparkFeaturePreprocessor.load(loaded_artifact_path)
        
        print("\nAttempting to transform inference data with a new 'strict' category...")
        transformed_inference_df = loaded_preprocessor.transform(inference_df)
        transformed_inference_df.show()
    except ValueError as e:
        print(f"\nSUCCESSFULLY caught expected error: {e}")

    # =========================================================================
    # Use Case 3: Partial Retraining with New Banners
    # =========================================================================
    print("\n" + "="*50)
    print(" Use Case 3: Partial Retraining")
    print("="*50)

    # 3.1. Data for partial retraining, including a new banner 'D'
    partial_retrain_data = [
        ('B', 'UK', 60, 50.0),   # Existing categories
        ('D', 'USA', 29, 180.0), # New banner 'D' for an extendable column
        ('A', 'CAN', 42, 220.0)
    ]
    partial_retrain_df = spark.createDataFrame(partial_retrain_data, CATEGORICAL_COLS + NUMERICAL_COLS)
    
    # 3.2. Load the same preprocessor again
    # The loaded_preprocessor is the same one from before the error
    print("\nTransforming partial retrain data...")
    transformed_partial_df = loaded_preprocessor.transform(partial_retrain_df)

    print("\nTransformed partial retrain data:")
    transformed_partial_df.show()

    print("\nUPDATED Categorical Mappings after partial retrain:")
    print(loaded_preprocessor.categorical_maps)
    
    # 3.3. Save the *updated* preprocessor back to MLFlow for future use
    with mlflow.start_run() as run:
        temp_path = f"updated_{PREPROCESSOR_ARTIFACT_NAME}"
        loaded_preprocessor.save(temp_path)
        mlflow.log_artifact(temp_path, artifact_path="preprocessor")
        os.remove(temp_path)

    spark.stop()
