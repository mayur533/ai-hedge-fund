import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from src.agents.fundamentals import fundamentals_analyst_agent
from src.graph.state import AgentState


class TestFundamentalsAgent:
    """Test suite for the fundamentals analyst agent."""

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
    def mock_financial_metrics(self):
        """Create mock financial metrics data."""
        mock_metrics = Mock()
        mock_metrics.return_on_equity = 0.20
        mock_metrics.net_margin = 0.25
        mock_metrics.operating_margin = 0.18
        mock_metrics.revenue_growth = 0.15
        mock_metrics.earnings_growth = 0.12
        mock_metrics.book_value_growth = 0.10
        mock_metrics.current_ratio = 2.0
        mock_metrics.debt_to_equity = 0.3
        mock_metrics.free_cash_flow_per_share = 5.0
        mock_metrics.earnings_per_share = 6.0
        mock_metrics.price_to_earnings_ratio = 20.0
        mock_metrics.price_to_book_ratio = 2.5
        mock_metrics.price_to_sales_ratio = 4.0
        return mock_metrics

    @patch('src.agents.fundamentals.get_financial_metrics')
    @patch('src.agents.fundamentals.get_api_key_from_state')
    @patch('src.agents.fundamentals.progress')
    def test_fundamentals_analyst_success(self, mock_progress, mock_get_api_key, mock_get_metrics, mock_agent_state, mock_financial_metrics):
        """Test successful fundamentals analysis."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_metrics.return_value = [mock_financial_metrics]
        
        # Call the function
        result = fundamentals_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        assert len(result["messages"]) == 1
        
        # Verify API calls
        mock_get_metrics.assert_called()
        mock_get_api_key.assert_called_once()
        
        # Verify progress updates were called
        assert mock_progress.update_status.call_count > 0

    @patch('src.agents.fundamentals.get_financial_metrics')
    @patch('src.agents.fundamentals.get_api_key_from_state')
    @patch('src.agents.fundamentals.progress')
    def test_fundamentals_analyst_no_metrics(self, mock_progress, mock_get_api_key, mock_get_metrics, mock_agent_state):
        """Test handling when no financial metrics are available."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_metrics.return_value = []
        
        # Call the function
        result = fundamentals_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        
        # Verify the analysis contains empty results for failed ticker
        analyst_signals = result["data"]["analyst_signals"]["fundamentals_analyst_agent"]
        assert "AAPL" not in analyst_signals  # Should be skipped due to no metrics

    @patch('src.agents.fundamentals.get_financial_metrics')
    @patch('src.agents.fundamentals.get_api_key_from_state')
    @patch('src.agents.fundamentals.progress')
    @patch('src.agents.fundamentals.show_agent_reasoning')
    def test_fundamentals_analyst_with_reasoning(self, mock_show_reasoning, mock_progress, mock_get_api_key, mock_get_metrics, mock_agent_state, mock_financial_metrics):
        """Test fundamentals analysis with reasoning enabled."""
        # Enable reasoning
        mock_agent_state["metadata"]["show_reasoning"] = True
        
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_metrics.return_value = [mock_financial_metrics]
        
        # Call the function
        result = fundamentals_analyst_agent(mock_agent_state)
        
        # Verify reasoning was displayed
        mock_show_reasoning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
