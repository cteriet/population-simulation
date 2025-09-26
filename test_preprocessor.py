import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    IntegerType, DecimalType, DoubleType
)
from decimal import Decimal

# Assume SparkFeaturePreprocessor is in a file named `preprocessor.py`
from preprocessor import SparkFeaturePreprocessor 

# -----------------
# Pytest Fixture for SparkSession
# -----------------

@pytest.fixture(scope="session")
def spark():
    """
    Creates a SparkSession for testing. A 'session' scope fixture is created 
    only once per test session, making tests run faster.
    """
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-pyspark-local-testing") \
        .getOrCreate()

# -----------------
# Tests for Each Method
# -----------------

def test_impute(spark):
    """
    Tests the _impute method for both numerical and categorical columns.
    """
    # Arrange: Create mock DataFrame and config
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True)
    ])
    data = [(1, 25, "New York"), (2, None, "London"), (3, 30, None)]
    df = spark.createDataFrame(data, schema)
    
    impute_config = {'numerical': ['age'], 'categorical': ['city']}
    preprocessor = SparkFeaturePreprocessor(impute_config=impute_config)

    # Act: Apply the method
    imputed_df = preprocessor._impute(df)

    # Assert: Define expected outcome and verify
    expected_data = [
        (1, 25, "New York", 0, 0),
        (2, 0, "London", 1, 0),
        (3, 30, "Other/Unknown", 0, 1)
    ]
    expected_df = spark.createDataFrame(
        expected_data,
        ["id", "age", "city", "age_isnull", "city_isnull"]
    )
    
    assert sorted(imputed_df.collect()) == sorted(expected_df.collect())

def test_merge_categories(spark):
    """
    Tests the _merge_categories method.
    """
    # Arrange
    data = [("A",), ("B",), ("C",), ("D",)]
    df = spark.createDataFrame(data, ["category"])
    
    merge_config = {
        "category": {
            "Group1": ["A", "B"],
            "Group2": ["C"]
        }
    }
    preprocessor = SparkFeaturePreprocessor(merge_config=merge_config)

    # Act
    merged_df = preprocessor._merge_categories(df)

    # Assert
    expected_data = [("Group1",), ("Group1",), ("Group2",), ("D",)]
    expected_df = spark.createDataFrame(expected_data, ["category"])
    
    assert sorted(merged_df.collect()) == sorted(expected_df.collect())

def test_apply_whitelist(spark):
    """
    Tests the _apply_whitelist method.
    """
    # Arrange
    data = [("USA",), ("Canada",), ("Mexico",), ("UK",)]
    df = spark.createDataFrame(data, ["country"])
    
    whitelist_config = {
        "country": ("Other", ["USA", "Canada"])
    }
    preprocessor = SparkFeaturePreprocessor(whitelist_config=whitelist_config)

    # Act
    whitelisted_df = preprocessor._apply_whitelist(df)

    # Assert
    expected_data = [("USA",), ("Canada",), ("Other",), ("Other",)]
    expected_df = spark.createDataFrame(expected_data, ["country"])

    assert sorted(whitelisted_df.collect()) == sorted(expected_df.collect())

def test_create_targets(spark):
    """
    Tests the _create_targets method.
    """
    # Arrange
    data = [("Approved",), ("Rejected",), ("Pending",)]
    df = spark.createDataFrame(data, ["status"])

    target_config = {"status": ["Approved", "Rejected"]}
    preprocessor = SparkFeaturePreprocessor(target_config=target_config)
    
    # Act
    target_df = preprocessor._create_targets(df)
    
    # Assert
    expected_data = [
        ("Approved", 1, 0),
        ("Rejected", 0, 1),
        ("Pending", 0, 0)
    ]
    expected_df = spark.createDataFrame(
        expected_data, 
        ["status", "target_approved", "target_rejected"]
    )
    
    assert sorted(target_df.collect()) == sorted(expected_df.collect())

def test_convert_decimals_to_doubles(spark):
    """
    Tests the _convert_decimals_to_doubles method.
    """
    # Arrange: Note the explicit DecimalType in the schema
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("price", DecimalType(10, 2)),
        StructField("quantity", IntegerType())
    ])
    data = [(1, Decimal("19.99"), 100), (2, Decimal("4.50"), 200)]
    df = spark.createDataFrame(data, schema)
    
    preprocessor = SparkFeaturePreprocessor() # No config needed
    
    # Act
    converted_df = preprocessor._convert_decimals_to_doubles(df)
    
    # Assert
    # Check that the schema has been updated correctly
    final_schema = {field.name: field.dataType for field in converted_df.schema.fields}
    assert isinstance(final_schema['price'], DoubleType)
    assert isinstance(final_schema['id'], IntegerType) # Ensure other types are unchanged

def test_fit(spark):
    """
    Tests the fit method to ensure it correctly learns the categories.
    """
    # Arrange
    data = [
        ("A", "X"), ("A", "X"), ("A", "X"), 
        ("B", "Y"), ("B", "Y"), 
        ("C", "Z") # C and Z are infrequent
    ]
    df = spark.createDataFrame(data, ["col1", "col2"])
    
    frequency_config = {'threshold': 2, 'columns': ['col1']}
    final_cleanup_config = {'threshold': 3, 'columns': ['col2']}
    
    preprocessor = SparkFeaturePreprocessor(
        frequency_config=frequency_config,
        final_cleanup_config=final_cleanup_config
    )
    
    # Act
    preprocessor.fit(df)
    
    # Assert: Check the internal state of the fitted object
    assert preprocessor.infrequent_levels_ == {"col1": ["C"]}
    assert preprocessor.categories_to_remove_ == {"col2": ["Z", "Y"]}
    
def test_fit_transform_end_to_end(spark):
    """
    Tests the full fit_transform pipeline to ensure all steps work together.
    """
    # Arrange
    data = [
        (1, "A"), (2, "A"), (3, "A"), # 'A' is frequent
        (4, "B"), (5, "B"),           # 'B' is frequent enough to keep but not to remove
        (6, "C")                      # 'C' is infrequent
    ]
    df = spark.createDataFrame(data, ["id", "category"])
    
    frequency_config = {'threshold': 2, 'columns': ['category']}
    final_cleanup_config = {'threshold': 3, 'columns': ['category']}
    
    preprocessor = SparkFeaturePreprocessor(
        frequency_config=frequency_config,
        final_cleanup_config=final_cleanup_config
    )
    
    # Act
    transformed_df = preprocessor.fit_transform(df)
    
    # Assert
    # 1. 'C' should be capped to 'Other/Unknown'.
    # 2. After capping, the counts are A:3, B:2, Other/Unknown:1.
    # 3. The final cleanup (threshold 3) should remove rows where category is B or Other/Unknown.
    # 4. Only rows with category 'A' should remain.
    expected_data = [(1, "A"), (2, "A"), (3, "A")]
    expected_df = spark.createDataFrame(expected_data, ["id", "category"])
    
    assert transformed_df.count() == 3
    assert sorted(transformed_df.collect()) == sorted(expected_df.collect())
