import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from src.agents.portfolio_manager import portfolio_manager_agent
from src.graph.state import AgentState


class TestPortfolioManager:
    """Test suite for the portfolio manager agent."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock agent state for testing."""
        return {
            "data": {
                "end_date": "2024-01-01",
                "tickers": ["AAPL", "GOOGL", "MSFT"],
                "analyst_signals": {
                    "fundamentals_analyst_agent": {
                        "AAPL": {"signal": "bullish", "confidence": 75, "reasoning": {}},
                        "GOOGL": {"signal": "bearish", "confidence": 60, "reasoning": {}},
                        "MSFT": {"signal": "neutral", "confidence": 50, "reasoning": {}}
                    },
                    "technical_analyst_agent": {
                        "AAPL": {"signal": "bullish", "confidence": 80, "reasoning": {}},
                        "GOOGL": {"signal": "neutral", "confidence": 40, "reasoning": {}},
                        "MSFT": {"signal": "bullish", "confidence": 70, "reasoning": {}}
                    }
                }
            },
            "metadata": {
                "show_reasoning": False
            }
        }

    @patch('src.agents.portfolio_manager.progress')
    def test_portfolio_manager_success(self, mock_progress, mock_agent_state):
        """Test successful portfolio management analysis."""
        # Call the function
        result = portfolio_manager_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        assert len(result["messages"]) == 1
        
        # Verify progress updates were called
        assert mock_progress.update_status.call_count > 0
        
        # Extract portfolio decisions
        portfolio_decisions = json.loads(result["messages"][0].content)
        
        # Verify all tickers have portfolio decisions
        assert "AAPL" in portfolio_decisions
        assert "GOOGL" in portfolio_decisions
        assert "MSFT" in portfolio_decisions
        
        # Verify decision structure
        for ticker, decision in portfolio_decisions.items():
            assert "action" in decision  # buy, sell, hold
            assert "confidence" in decision
            assert "position_size" in decision
            assert "reasoning" in decision

    @patch('src.agents.portfolio_manager.progress')
    def test_portfolio_manager_with_reasoning(self, mock_progress, mock_agent_state):
        """Test portfolio management with reasoning enabled."""
        # Enable reasoning
        mock_agent_state["metadata"]["show_reasoning"] = True
        
        # Call the function
        result = portfolio_manager_agent(mock_agent_state)
        
        # Verify reasoning was displayed (would be called in actual implementation)

    @patch('src.agents.portfolio_manager.progress')
    def test_portfolio_manager_missing_analyst_signals(self, mock_progress):
        """Test portfolio manager with missing analyst signals."""
        # Create state with missing analyst signals
        incomplete_state = {
            "data": {
                "end_date": "2024-01-01",
                "tickers": ["AAPL"],
                "analyst_signals": {}  # No analyst signals
            },
            "metadata": {
                "show_reasoning": False
            }
        }
        
        # Call the function
        result = portfolio_manager_agent(incomplete_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        
        # Extract portfolio decisions
        portfolio_decisions = json.loads(result["messages"][0].content)
        
        # Should handle missing signals gracefully
        assert "AAPL" in portfolio_decisions
        assert portfolio_decisions["AAPL"]["action"] == "hold"  # Default action


if __name__ == "__main__":
    pytest.main([__file__])
