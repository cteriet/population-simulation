import pytest
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from spark_preprocessor import SparkFeaturePreprocessor # Assumes the class is in this file

# Pytest fixture to create a SparkSession for the tests
@pytest.fixture(scope="session")
def spark_session():
    """Creates a SparkSession for the test suite."""
    spark = SparkSession.builder \
        .appName("SparkPreprocessorTests") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture to provide a sample DataFrame and configuration for tests
@pytest.fixture
def sample_data(spark_session):
    """Provides a sample DataFrame and column configuration."""
    data = [
        ('A', 'USA', 25, 120.50),
        ('B', 'UK',  45, 88.0),
        ('A', 'UK',  31, 250.0),
        ('C', 'USA', 22, 94.20),
        ('B', 'USA', 51, 150.75),
        ('C', 'CAN', 38, 310.0),
    ]
    numerical_cols = ['customer_age', 'avg_spend']
    categorical_cols = ['banner_id', 'customer_country']
    schema = StructType([
        StructField('banner_id', StringType()),
        StructField('customer_country', StringType()),
        StructField('customer_age', IntegerType()),
        StructField('avg_spend', DoubleType())
    ])
    df = spark_session.createDataFrame(data, schema)
    return df, numerical_cols, categorical_cols

def test_initialization():
    """Tests that the preprocessor initializes correctly."""
    preprocessor = SparkFeaturePreprocessor(
        numerical_cols=['n1'],
        categorical_cols=['c1', 'c2'],
        extendable_cols=['c1']
    )
    assert preprocessor.numerical_cols == ['n1']
    assert preprocessor.categorical_cols == ['c1', 'c2']
    assert preprocessor.extendable_cols == ['c1']

def test_initialization_invalid_extendable_raises_error():
    """Tests that an error is raised if extendable_cols is not a subset of categorical_cols."""
    with pytest.raises(ValueError, match="extendable_cols must be a subset of categorical_cols"):
        SparkFeaturePreprocessor(
            numerical_cols=['n1'],
            categorical_cols=['c1'],
            extendable_cols=['c1', 'c2'] # 'c2' is not in categorical_cols
        )

def test_fit(sample_data):
    """Tests the fit method to ensure it learns the correct parameters."""
    df, numerical_cols, categorical_cols = sample_data
    preprocessor = SparkFeaturePreprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        extendable_cols=[]
    )
    preprocessor.fit(df)

    # Test numerical scalers
    scalers = preprocessor.numerical_scalers
    assert 'customer_age' in scalers
    assert 'avg_spend' in scalers
    pytest.approx(35.33, 0.01) == scalers['customer_age']['mean']
    pytest.approx(10.84, 0.01) == scalers['customer_age']['stddev']
    pytest.approx(168.91, 0.01) == scalers['avg_spend']['mean']
    pytest.approx(88.01, 0.01) == scalers['avg_spend']['stddev']

    # Test categorical maps (order of mapping is not guaranteed)
    maps = preprocessor.categorical_maps
    assert 'banner_id' in maps
    assert 'customer_country' in maps
    assert set(maps['banner_id']['mapping'].keys()) == {'A', 'B', 'C'}
    assert set(maps['customer_country']['mapping'].keys()) == {'USA', 'UK', 'CAN'}
    assert maps['banner_id']['next_id'] == 3
    assert maps['customer_country']['next_id'] == 3

def test_fit_handles_zero_stddev(spark_session):
    """Tests that fit handles numerical columns with zero standard deviation."""
    data = [('A', 10), ('B', 10), ('C', 10)]
    df = spark_session.createDataFrame(data, ['cat', 'num'])
    preprocessor = SparkFeaturePreprocessor(numerical_cols=['num'], categorical_cols=['cat'])
    preprocessor.fit(df)

    # stddev should default to 1.0 to avoid division by zero
    assert preprocessor.numerical_scalers['num']['stddev'] == 1.0

def test_transform(sample_data):
    """Tests the transform method applies learned transformations correctly."""
    df, numerical_cols, categorical_cols = sample_data
    preprocessor = SparkFeaturePreprocessor(numerical_cols, categorical_cols).fit(df)
    transformed_df = preprocessor.transform(df)

    # Check if columns are transformed and have the correct type
    assert 'customer_age' in transformed_df.columns
    assert 'banner_id' in transformed_df.columns
    assert isinstance(transformed_df.schema['customer_age'].dataType, DoubleType)
    assert isinstance(transformed_df.schema['banner_id'].dataType, IntegerType)
    
    # Simple check on row count
    assert transformed_df.count() == df.count()

def test_transform_extends_categories(spark_session, sample_data):
    """Tests that new categories are added for extendable columns during transform."""
    df, numerical_cols, categorical_cols = sample_data
    
    # 'banner_id' is extendable
    preprocessor = SparkFeaturePreprocessor(numerical_cols, categorical_cols, extendable_cols=['banner_id'])
    preprocessor.fit(df)

    # New data with a new banner 'D'
    new_data = [('D', 'USA', 30, 100.0)]
    new_df = spark_session.createDataFrame(new_data, schema=df.schema)

    # Transform should not fail and should extend the mapping
    transformed_df = preprocessor.transform(new_df)
    
    # Check that the mapping was updated
    assert 'D' in preprocessor.categorical_maps['banner_id']['mapping']
    assert preprocessor.categorical_maps['banner_id']['next_id'] == 4 # A, B, C, now D
    
    # Check if the new banner was encoded correctly
    new_banner_id = preprocessor.categorical_maps['banner_id']['mapping']['D']
    assert transformed_df.select('banner_id').first()[0] == new_banner_id

def test_transform_strict_categories_raises_error(spark_session, sample_data):
    """Tests that an error is raised for new categories in a strict column."""
    df, numerical_cols, categorical_cols = sample_data

    # 'customer_country' is strict by default
    preprocessor = SparkFeaturePreprocessor(numerical_cols, categorical_cols, extendable_cols=['banner_id'])
    preprocessor.fit(df)

    # New data with a new country 'GER'
    new_data = [('A', 'GER', 30, 100.0)]
    new_df = spark_session.createDataFrame(new_data, schema=df.schema)

    # Expect a ValueError because 'customer_country' is not extendable
    with pytest.raises(ValueError, match="Found unexpected new categories in 'strict' column 'customer_country'"):
        preprocessor.transform(new_df)

def test_save_and_load(sample_data, tmp_path):
    """Tests that saving and loading the preprocessor preserves its state."""
    df, numerical_cols, categorical_cols = sample_data
    
    original_preprocessor = SparkFeaturePreprocessor(numerical_cols, categorical_cols)
    original_preprocessor.fit(df)

    # Save to a temporary path provided by pytest
    file_path = os.path.join(tmp_path, "preprocessor.pkl")
    original_preprocessor.save(file_path)
    
    assert os.path.exists(file_path)

    # Load it back
    loaded_preprocessor = SparkFeaturePreprocessor.load(file_path)

    # Check if the state is identical
    assert original_preprocessor.numerical_scalers == loaded_preprocessor.numerical_scalers
    assert original_preprocessor.categorical_maps == loaded_preprocessor.categorical_maps
    assert original_preprocessor.numerical_cols == loaded_preprocessor.numerical_cols
    assert original_preprocessor.categorical_cols == loaded_preprocessor.categorical_cols
    assert original_preprocessor.extendable_cols == loaded_preprocessor.extendable_cols
