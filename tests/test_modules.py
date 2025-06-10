"""Unit tests for LangChain multi-model examples"""

import pytest
from unittest.mock import Mock, patch
from src.config import ModelConfig
from src.tool_usage import get_weather, calculate, get_time

class TestConfig:
    """Test configuration loading"""
    
    def test_config_initialization(self):
        """Test that config initializes correctly"""
        with patch.dict('os.environ', {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test-key'
        }):
            config = ModelConfig()
            assert config.provider == 'openai'
            assert config.openai_api_key == 'test-key'
    
    def test_invalid_provider(self):
        """Test handling of invalid provider"""
        config = ModelConfig()
        with pytest.raises(ValueError):
            config.get_model('invalid_provider')

class TestTools:
    """Test tool functions"""
    
    def test_weather_tool(self):
        """Test weather tool returns expected format"""
        result = get_weather.invoke({'city': 'London'})
        assert 'London' in result
        assert '°C' in result
    
    def test_calculate_tool(self):
        """Test calculation tool"""
        result = calculate.invoke({'expression': '2 + 2'})
        assert 'Result: 4' in result
        
        # Test error handling
        result = calculate.invoke({'expression': 'invalid'})
        assert 'Error' in result
    
    def test_time_tool(self):
        """Test time tool"""
        result = get_time.invoke({'timezone': 'UTC'})
        assert 'Current time' in result
        assert 'UTC' in result

class TestStructuredOutputs:
    """Test Pydantic model validation"""
    
    def test_person_model(self):
        """Test Person model validation"""
        from src.structured_outputs import Person
        
        person = Person(
            name="John Doe",
            age=30,
            occupation="Engineer",
            skills=["Python", "AI"]
        )
        
        assert person.name == "John Doe"
        assert person.age == 30
        assert len(person.skills) == 2
    
    def test_company_model(self):
        """Test CompanyInfo model validation"""
        from src.structured_outputs import CompanyInfo
        
        company = CompanyInfo(
            name="TechCorp",
            industry="Technology",
            founded_year=2020,
            headquarters="San Francisco",
            key_products=["AI Platform"]
        )
        
        assert company.name == "TechCorp"
        assert company.founded_year == 2020
        assert company.employee_count is None  # Optional field

class TestResearchAssistant:
    """Test research assistant components"""
    
    def test_research_query_model(self):
        """Test ResearchQuery model"""
        from src.research_assistant import ResearchQuery
        
        query = ResearchQuery(
            topic="AI Safety",
            depth="deep",
            focus_areas=["alignment", "robustness"]
        )
        
        assert query.topic == "AI Safety"
        assert query.depth == "deep"
        assert len(query.focus_areas) == 2
    
    def test_research_result_model(self):
        """Test ResearchResult model"""
        from src.research_assistant import ResearchResult
        from datetime import datetime
        
        result = ResearchResult(
            topic="AI Safety",
            summary="Research summary",
            key_findings=["Finding 1", "Finding 2"],
            sources_consulted=["source1", "source2"],
            confidence_score=0.85,
            model_used="openai",
            timestamp=datetime.now().isoformat()
        )
        
        assert result.topic == "AI Safety"
        assert result.confidence_score == 0.85
        assert len(result.key_findings) == 2

@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functionality"""
    
    async def test_basic_chat_demo(self):
        """Test basic chat demonstration"""
        from src.basic_chat import demonstrate_basic_chat
        
        # Mock models
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(content="Test response")
        mock_model.astream.return_value = [Mock(content="Chunk")]
        
        models = {"test": mock_model}
        
        # Should not raise exceptions
        await demonstrate_basic_chat(models)
        assert mock_model.invoke.called
