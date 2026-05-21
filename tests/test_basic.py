"""
Tests for multi_factor_selection package.
多因子选股系统测试文件。
"""

import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import DataFetcher
from factor_calculator import FactorCalculator
from factor_processor import FactorProcessor
from stock_selector import StockSelector


class TestDataFetcher:
    """Test cases for DataFetcher class."""

    @pytest.fixture
    def fetcher(self):
        """Create a DataFetcher instance for testing."""
        return DataFetcher(use_tushare=True, use_akshare_backup=True)

    def test_init(self, fetcher):
        """Test DataFetcher initialization."""
        assert fetcher is not None
        assert hasattr(fetcher, 'use_tushare')
        assert hasattr(fetcher, 'use_akshare_backup')

    def test_get_stock_list(self, fetcher):
        """Test getting stock list."""
        # This test might require network access
        # Use mock in real tests
        pass

    def test_get_stock_info(self, fetcher):
        """Test getting stock info."""
        # Test with a known stock code
        pass


class TestFactorCalculator:
    """Test cases for FactorCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a FactorCalculator instance for testing."""
        return FactorCalculator()

    def test_init(self, calculator):
        """Test FactorCalculator initialization."""
        assert calculator is not None

    def test_calculate_value_factors(self, calculator):
        """Test value factor calculation."""
        # Create mock data
        data = pd.DataFrame({
            'close': [10.0, 20.0, 30.0],
            'pe': [15.0, 20.0, 25.0],
            'pb': [1.5, 2.0, 2.5],
        })
        result = calculator.calculate_value_factors(data)
        assert result is not None


class TestFactorProcessor:
    """Test cases for FactorProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a FactorProcessor instance for testing."""
        return FactorProcessor()

    def test_init(self, processor):
        """Test FactorProcessor initialization."""
        assert processor is not None

    def test_winsorize(self, processor):
        """Test winsorization."""
        import numpy as np
        data = pd.Series(np.random.normal(0, 1, 100))
        result = processor.winsorize(data, lower=0.01, upper=0.99)
        assert len(result) == len(data)


class TestStockSelector:
    """Test cases for StockSelector class."""

    @pytest.fixture
    def selector(self):
        """Create a StockSelector instance for testing."""
        return StockSelector()

    def test_init(self, selector):
        """Test StockSelector initialization."""
        assert selector is not None

    def test_select_stocks(self, selector):
        """Test stock selection."""
        # Create mock data
        data = pd.DataFrame({
            'code': ['000001', '000002', '000003'],
            'name': ['平安银行', '万科A', '中国平安'],
            'total_score': [0.8, 0.6, 0.9],
        })
        result = selector.select_stocks(data, top_n=2)
        assert len(result) == 2
        assert result.iloc[0]['code'] == '000003'  # Highest score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
