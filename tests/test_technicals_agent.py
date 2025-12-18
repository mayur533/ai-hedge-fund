import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import pandas as pd
from datetime import datetime

from src.agents.technicals import technical_analyst_agent
from src.graph.state import AgentState


class TestTechnicalsAgent:
    """Test suite for the technical analyst agent."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock agent state for testing."""
        return {
            "data": {
                "end_date": "2024-01-01",
                "tickers": ["AAPL", "GOOGL"],
                "analyst_signals": {}
            },
            "metadata": {
                "show_reasoning": False
            }
        }

    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data for testing."""
        # Create a list of mock price objects
        prices = []
        for i in range(50):  # 50 days of data
            price = Mock()
            price.time = f"2024-{(i % 12) + 1}-{(i % 28) + 1}T00:00:00Z"
            price.open = 100.0 + i
            price.close = 101.0 + i
            price.high = 102.0 + i
            price.low = 99.0 + i
            price.volume = 1000000 + i * 1000
            prices.append(price)
        return prices

    @patch('src.agents.technicals.get_prices')
    @patch('src.agents.technicals.get_api_key_from_state')
    @patch('src.agents.technicals.progress')
    def test_technical_analyst_success(self, mock_progress, mock_get_api_key, mock_get_prices, mock_agent_state, mock_price_data):
        """Test successful technical analysis."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_prices.return_value = mock_price_data
        
        # Call the function
        result = technical_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        assert len(result["messages"]) == 1
        
        # Verify API calls
        mock_get_prices.assert_called()
        mock_get_api_key.assert_called_once()
        
        # Verify progress updates were called
        assert mock_progress.update_status.call_count > 0

    @patch('src.agents.technicals.get_prices')
    @patch('src.agents.technicals.get_api_key_from_state')
    @patch('src.agents.technicals.progress')
    def test_technical_analyst_no_price_data(self, mock_progress, mock_get_api_key, mock_get_prices, mock_agent_state):
        """Test handling when no price data is available."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_prices.return_value = []
        
        # Call the function
        result = technical_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        
        # Verify the analysis contains empty results for failed ticker
        analyst_signals = result["data"]["analyst_signals"]["technical_analyst_agent"]
        assert "AAPL" not in analyst_signals  # Should be skipped due to no data

    @patch('src.agents.technicals.get_prices')
    @patch('src.agents.technicals.get_api_key_from_state')
    @patch('src.agents.technicals.progress')
    @patch('src.agents.technicals.show_agent_reasoning')
    def test_technical_analyst_with_reasoning(self, mock_show_reasoning, mock_progress, mock_get_api_key, mock_get_prices, mock_agent_state, mock_price_data):
        """Test technical analysis with reasoning enabled."""
        # Enable reasoning
        mock_agent_state["metadata"]["show_reasoning"] = True
        
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_prices.return_value = mock_price_data
        
        # Call the function
        result = technical_analyst_agent(mock_agent_state)
        
        # Verify reasoning was displayed
        mock_show_reasoning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
