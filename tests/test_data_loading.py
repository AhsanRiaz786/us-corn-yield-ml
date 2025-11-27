import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'modeling_dataset_final.csv'

def test_data_file_exists():
    """Test if the processed data file exists."""
    assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"

def test_data_loading():
    """Test if data can be loaded and has expected columns."""
    if not DATA_PATH.exists():
        pytest.skip("Data file not found")
    
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataframe is empty"
    
    expected_cols = ['Year', 'State', 'County', 'Yield_BU_ACRE']
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    assert df['Yield_BU_ACRE'].dtype in [float, int], "Yield column should be numeric"
