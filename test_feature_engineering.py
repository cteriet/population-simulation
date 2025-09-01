import pytest
from pyspark.sql import SparkSession, Row

# Import the function you want to test
from feature_engineering import full_feature_engineering_pipeline

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for the test session."""
    session = SparkSession.builder \
        .appName("pytest-pyspark-local-testing") \
        .master("local[2]") \
        .getOrCreate()
    yield session
    session.stop()

# --- Test Cases for Each Unit of Functionality ---

def test_null_imputation(spark):
    """Tests if nulls are imputed correctly and indicators are created."""
    # Arrange
    input_data = [Row(id=1, age=25, country='USA'), Row(id=2, age=None, country=None)]
    input_df = spark.createDataFrame(input_data)
    
    config = {
        'impute_config': {'numerical': ['age'], 'categorical': ['country']},
        'merge_config': {},
        'frequency_config': {}
    }
    
    # Act
    result_df = full_feature_engineering_pipeline(input_df, **config)
    result = {row['id']: row for row in result_df.collect()}
    
    # Assert
    assert result[1]['age'] == 25 and result[1]['age_isnull'] == 0
    assert result[1]['country'] == 'USA' and result[1]['country_isnull'] == 0
    
    assert result[2]['age'] == 0 and result[2]['age_isnull'] == 1
    assert result[2]['country'] == 'Other/Unknown' and result[2]['country_isnull'] == 1

def test_category_merging(spark):
    """Tests if categories are merged according to the config."""
    # Arrange
    input_data = [Row(country='USA'), Row(country='Canada'), Row(country='Mexico')]
    input_df = spark.createDataFrame(input_data)
    
    config = {
        'impute_config': {},
        'merge_config': {'country': {'North America': ['USA', 'Canada']}},
        'frequency_config': {}
    }
    
    # Act
    result_df = full_feature_engineering_pipeline(input_df, **config)
    result = [row['country'] for row in result_df.collect()]
    
    # Assert
    expected = ['North America', 'North America', 'Mexico']
    assert sorted(result) == sorted(expected)

def test_frequency_capping(spark):
    """Tests if rare categories are mapped to 'Other/Unknown'."""
    # Arrange
    input_data = [Row(c='A'), Row(c='A'), Row(c='A'), Row(c='B'), Row(c='C')]
    input_df = spark.createDataFrame(input_data)
    
    config = {
        'impute_config': {},
        'merge_config': {},
        'frequency_config': {'columns': ['c'], 'threshold': 2}
    }
    
    # Act
    result_df = full_feature_engineering_pipeline(input_df, **config)
    result = [row['c'] for row in result_df.collect()]
    
    # Assert
    expected = ['A', 'A', 'A', 'Other/Unknown', 'Other/Unknown']
    assert sorted(result) == sorted(expected)

def test_final_cleanup(spark):
    """Tests if rows with very rare categories are removed."""
    # Arrange
    input_data = [Row(c='A'), Row(c='A'), Row(c='A'), Row(c='B')]
    input_df = spark.createDataFrame(input_data)
    
    config = {
        'impute_config': {},
        'merge_config': {},
        'frequency_config': {},
        'final_cleanup_config': {'columns': ['c'], 'threshold': 2}
    }
    
    # Act
    result_df = full_feature_engineering_pipeline(input_df, **config)
    
    # Assert
    assert result_df.count() == 3
    assert result_df.filter("c = 'B'").count() == 0

# --- Full End-to-End Integration Test ---

def test_full_pipeline_integration(spark):
    """Tests the entire pipeline with a complex scenario."""
    # Arrange
    input_data = [
        Row(id=1, age=25, income=50000.0, product_category='Laptop', country='USA'),
        Row(id=2, age=30, income=75000.0, product_category='Phone', country='USA'),
        Row(id=3, age=None, income=120000.0, product_category='TV', country='Canada'),
        Row(id=4, age=45, income=None, product_category='Chair', country='USA'),
        Row(id=5, age=22, income=40000.0, product_category=None, country='Mexico'),
        Row(id=6, age=50, income=250000.0, product_category='Sofa', country='Canada'),
        Row(id=7, age=35, income=95000.0, product_category='Table', country='Mexico')
    ]
    input_df = spark.createDataFrame(input_data)

    config = {
        'impute_config': {'numerical': ['age', 'income'], 'categorical': ['product_category']},
        'merge_config': {'country': {'North America': ['USA', 'Canada']}},
        'frequency_config': {'columns': ['country'], 'threshold': 3},
        'final_cleanup_config': {'columns': ['product_category'], 'threshold': 2}
    }

    # Act
    result_df = full_feature_engineering_pipeline(input_df, **config)
    result_data = sorted([row.asDict() for row in result_df.collect()], key=lambda x: x['id'])

    # Assert
    expected_data = [
        {'id': 1, 'age': 25, 'income': 50000.0, 'product_category': 'Laptop', 'country': 'North America', 'age_isnull': 0, 'income_isnull': 0, 'product_category_isnull': 0},
        {'id': 2, 'age': 30, 'income': 75000.0, 'product_category': 'Phone', 'country': 'North America', 'age_isnull': 0, 'income_isnull': 0, 'product_category_isnull': 0},
        {'id': 3, 'age': 0, 'income': 120000.0, 'product_category': 'TV', 'country': 'North America', 'age_isnull': 1, 'income_isnull': 0, 'product_category_isnull': 0},
        {'id': 4, 'age': 45, 'income': 0.0, 'product_category': 'Chair', 'country': 'North America', 'age_isnull': 0, 'income_isnull': 1, 'product_category_isnull': 0},
        {'id': 6, 'age': 50, 'income': 250000.0, 'product_category': 'Sofa', 'country': 'North America', 'age_isnull': 0, 'income_isnull': 0, 'product_category_isnull': 0},
        {'id': 7, 'age': 35, 'income': 95000.0, 'product_category': 'Table', 'country': 'Other/Unknown', 'age_isnull': 0, 'income_isnull': 0, 'product_category_isnull': 0}
    ]

    assert result_df.count() == 6
    assert result_data == expected_data
