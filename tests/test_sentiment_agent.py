import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from src.agents.sentiment import sentiment_analyst_agent
from src.graph.state import AgentState


class TestSentimentAgent:
    """Test suite for the sentiment analyst agent."""

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
    def mock_news_data(self):
        """Create mock news data for testing."""
        return [
            {
                "title": "Apple Reports Strong Q4 Earnings",
                "content": "Apple Inc. reported better than expected quarterly earnings driven by strong iPhone sales.",
                "sentiment": "positive",
                "confidence": 0.85,
                "source": "Reuters",
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "title": "Google Faces Regulatory Challenges",
                "content": "Alphabet's Google subsidiary faces new regulatory scrutiny in European markets.",
                "sentiment": "negative",
                "confidence": 0.75,
                "source": "Bloomberg",
                "timestamp": "2024-01-01T09:00:00Z"
            },
            {
                "title": "Tech Stocks Mixed in Early Trading",
                "content": "Technology stocks showed mixed performance in early market trading.",
                "sentiment": "neutral",
                "confidence": 0.60,
                "source": "CNBC",
                "timestamp": "2024-01-01T08:00:00Z"
            }
        ]

    @patch('src.agents.sentiment.get_news_sentiment')
    @patch('src.agents.sentiment.get_api_key_from_state')
    @patch('src.agents.sentiment.progress')
    def test_sentiment_analyst_success(self, mock_progress, mock_get_api_key, mock_get_news, mock_agent_state, mock_news_data):
        """Test successful sentiment analysis."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_news.return_value = mock_news_data
        
        # Call the function
        result = sentiment_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        assert len(result["messages"]) == 1
        
        # Verify API calls
        mock_get_news.assert_called()
        mock_get_api_key.assert_called_once()
        
        # Verify progress updates were called
        assert mock_progress.update_status.call_count > 0

    @patch('src.agents.sentiment.get_news_sentiment')
    @patch('src.agents.sentiment.get_api_key_from_state')
    @patch('src.agents.sentiment.progress')
    def test_sentiment_analyst_no_news_data(self, mock_progress, mock_get_api_key, mock_get_news, mock_agent_state):
        """Test handling when no news data is available."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_get_news.return_value = []
        
        # Call the function
        result = sentiment_analyst_agent(mock_agent_state)
        
        # Verify the result structure
        assert "messages" in result
        assert "data" in result
        
        # Verify the analysis contains empty results for failed ticker
        analyst_signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]
        assert "AAPL" not in analyst_signals  # Should be skipped due to no data

    @patch('src.agents.sentiment.get_news_sentiment')
    @patch('src.agents.sentiment.get_api_key_from_state')
    @patch('src.agents.sentiment.progress')
    def test_positive_sentiment_analysis(self, mock_progress, mock_get_api_key, mock_get_news, mock_agent_state):
        """Test positive sentiment analysis."""
        # Create positive news data
        positive_news = [
            {
                "title": "Apple Stock Surges on Positive Outlook",
                "content": "Apple shares rose significantly after positive analyst ratings.",
                "sentiment": "positive",
                "confidence": 0.90,
                "source": "WSJ",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]
        
        mock_get_api_key.return_value = "test-api-key"
        mock_get_news.return_value = positive_news
        
        # Call the function
        result = sentiment_analyst_agent(mock_agent_state)
        
        # Extract analysis
        analysis = json.loads(result["messages"][0].content)
        aapl_analysis = analysis["AAPL"]
        
        # Verify bullish sentiment signal
        assert aapl_analysis["signal"] in ["bullish", "strong_bullish"]
        assert aapl_analysis["confidence"] > 70


if __name__ == "__main__":
    pytest.main([__file__])
