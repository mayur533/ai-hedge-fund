import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtester import Backtester


class TestBacktester:
    """Test suite for the backtesting engine."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = []
        
        for i, date in enumerate(dates):
            # Create realistic price movements
            base_price = 100
            price = base_price + np.sin(i * 0.1) * 10 + np.random.normal(0, 2)
            
            data.append({
                'date': date,
                'open': price - 0.5,
                'high': price + 1,
                'low': price - 1,
                'close': price,
                'volume': 1000000 + np.random.randint(-100000, 100000)
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def backtester(self):
        """Create a backtester instance."""
        return Backtester(
            initial_cash=100000,
            commission_rate=0.001,
            slippage_rate=0.0001
        )

    def test_backtester_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.initial_cash == 100000
        assert backtester.commission_rate == 0.001
        assert backtester.slippage_rate == 0.0001
        assert backtester.current_cash == 100000
        assert backtester.positions == {}
        assert backtester.trades == []

    def test_backtester_buy_signal(self, backtester, sample_price_data):
        """Test executing a buy signal."""
        # Setup
        price_data = sample_price_data.head(10)
        
        # Create buy signal
        signals = pd.DataFrame([{
            'date': '2024-01-01',
            'signal': 'buy',
            'confidence': 0.8,
            'ticker': 'AAPL'
        }])
        
        # Execute trade
        backtester.execute_trades(signals, price_data)
        
        # Verify trade execution
        assert len(backtester.trades) == 1
        assert backtester.trades[0]['action'] == 'buy'
        assert 'AAPL' in backtester.positions
        assert backtester.current_cash < 100000  # Cash reduced by purchase

    def test_backtester_sell_signal(self, backtester, sample_price_data):
        """Test executing a sell signal."""
        # First buy to establish position
        backtester.positions['AAPL'] = {
            'quantity': 100,
            'avg_price': 100.0,
            'total_cost': 10000.0
        }
        
        # Create sell signal
        signals = pd.DataFrame([{
            'date': '2024-01-01',
            'signal': 'sell',
            'confidence': 0.8,
            'ticker': 'AAPL'
        }])
        
        # Execute trade
        backtester.execute_trades(signals, sample_price_data.head(10))
        
        # Verify trade execution
        assert len(backtester.trades) == 1
        assert backtester.trades[0]['action'] == 'sell'
        assert backtester.current_cash > 100000  # Cash increased by sale

    def test_backtester_hold_signal(self, backtester, sample_price_data):
        """Test executing a hold signal."""
        # Create hold signal
        signals = pd.DataFrame([{
            'date': '2024-01-01',
            'signal': 'hold',
            'confidence': 0.5,
            'ticker': 'AAPL'
        }])
        
        initial_cash = backtester.current_cash
        initial_trades = len(backtester.trades)
        
        # Execute trade
        backtester.execute_trades(signals, sample_price_data.head(10))
        
        # Verify no trade executed
        assert len(backtester.trades) == initial_trades
        assert backtester.current_cash == initial_cash

    def test_backtester_portfolio_value_calculation(self, backtester, sample_price_data):
        """Test portfolio value calculation."""
        # Add some positions
        backtester.positions['AAPL'] = {
            'quantity': 100,
            'avg_price': 100.0,
            'total_cost': 10000.0
        }
        
        # Calculate portfolio value
        portfolio_value = backtester.calculate_portfolio_value(sample_price_data.head(1))
        
        # Verify calculation
        assert portfolio_value > backtester.current_cash
        assert isinstance(portfolio_value, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
