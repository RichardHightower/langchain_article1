"""Multi-model research assistant example"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

# Data models
class ResearchQuery(BaseModel):
    topic: str = Field(description="Research topic")
    depth: str = Field(
        description="Research depth: quick, moderate, deep",
        default="moderate"
    )
    focus_areas: List[str] = Field(
        description="Specific areas to focus on",
        default_factory=list
    )

class ResearchResult(BaseModel):
    topic: str
    summary: str
    key_findings: List[str]
    sources_consulted: List[str]
    confidence_score: float = Field(
        description="Confidence in findings (0-1)",
        ge=0, le=1
    )
    model_used: str
    timestamp: str

# Research tools
@tool
def search_papers(topic: str, limit: int = 5) -> str:
    """Search for academic papers on a topic.
    
    Args:
        topic: Research topic
        limit: Maximum number of results
        
    Returns:
        List of relevant papers
    """
    papers = [
        f"'{topic}' - Comprehensive Review (2024)",
        f"Advances in {topic}: Recent Developments (2023)",
        f"Understanding {topic}: A Practical Guide (2024)",
        f"{topic} Applications in Industry (2023)",
        f"Future of {topic}: Predictions and Trends (2024)"
    ]
    return f"Found {len(papers[:limit])} papers:\n" + "\n".join(papers[:limit])

@tool
def get_statistics(topic: str) -> str:
    """Get statistics related to a topic.
    
    Args:
        topic: Topic to get statistics for
        
    Returns:
        Relevant statistics
    """
    # Simulated statistics
    import random
    growth = random.randint(10, 50)
    adoption = random.randint(30, 80)
    investment = random.randint(1, 10)
    
    return f"""Statistics for {topic}:
    - Annual growth rate: {growth}%
    - Industry adoption: {adoption}%
    - Investment (billions): ${investment}B
    - Research papers published (2023): {random.randint(1000, 5000)}"""

@tool
def get_expert_opinion(topic: str) -> str:
    """Get simulated expert opinion on a topic.
    
    Args:
        topic: Topic for expert opinion
        
    Returns:
        Expert perspective
    """
    opinions = {
        "ai": "AI will transform every industry, but ethical considerations are crucial.",
        "blockchain": "Blockchain's real value lies in transparency and decentralization.",
        "quantum": "Quantum computing will revolutionize cryptography and drug discovery.",
        "default": f"{topic} shows promising potential but requires careful implementation."
    }
    
    for key, opinion in opinions.items():
        if key in topic.lower():
            return f"Expert opinion on {topic}: {opinion}"
    
    return f"Expert opinion on {topic}: {opinions['default']}"

class MultiModelResearchAssistant:
    """Research assistant that leverages multiple models"""
    
    def __init__(self, models: Dict[str, BaseChatModel]):
        self.models = models
        self.tools = [search_papers, get_statistics, get_expert_opinion]
        
        # Initialize tool-capable models
        self.tool_models = {}
        for name, model in models.items():
            try:
                self.tool_models[name] = model.bind_tools(self.tools)
            except:
                pass
        
        # Research prompts
        self.prompts = {
            "quick": "Provide a brief overview of {topic}. Focus on: {focus_areas}",
            "moderate": """Research {topic} comprehensively.
            Focus areas: {focus_areas}
            Use available tools to gather information.
            Provide a balanced analysis.""",
            "deep": """Conduct thorough research on {topic}.
            Focus areas: {focus_areas}
            Use all available tools to gather data, statistics, and expert opinions.
            Provide a detailed analysis with multiple perspectives."""
        }
        
        # Results parser
        self.result_parser = PydanticOutputParser(pydantic_object=ResearchResult)
    
    async def research(self, query: ResearchQuery, model_name: Optional[str] = None) -> ResearchResult:
        """Perform research using specified or best available model"""
        
        # Select model
        if model_name and model_name in self.tool_models:
            model = self.tool_models[model_name]
            used_model = model_name
        elif self.tool_models:
            # Prefer models in order: openai, anthropic, others
            for preferred in ["openai", "anthropic"]:
                if preferred in self.tool_models:
                    model = self.tool_models[preferred]
                    used_model = preferred
                    break
            else:
                model = list(self.tool_models.values())[0]
                used_model = list(self.tool_models.keys())[0]
        else:
            raise ValueError("No tool-capable models available")
        
        # Create research prompt
        prompt = ChatPromptTemplate.from_template(self.prompts[query.depth])
        
        # Format focus areas
        focus_areas_str = ", ".join(query.focus_areas) if query.focus_areas else "general overview"
        
        messages = prompt.format_messages(
            topic=query.topic,
            focus_areas=focus_areas_str
        )
        
        # Execute research
        response = await model.ainvoke(messages)
        messages.append(response)
        
        # Track sources consulted
        sources = []
        
        # Handle tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                sources.append(f"{tool_call['name']}({tool_call['args']})")
                
                # Execute tool
                tool_fn = globals()[tool_call['name']]
                result = tool_fn.invoke(tool_call['args'])
                
                # Add result
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id']
                ))
            
            # Get final response with tool results
            response = await model.ainvoke(messages)
        
        # Extract key findings
        content = response.content
        lines = content.split('\n')
        key_findings = [
            line.strip().lstrip('- •')
            for line in lines
            if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))
        ][:5]
        
        # Create result
        return ResearchResult(
            topic=query.topic,
            summary=content[:500] + "..." if len(content) > 500 else content,
            key_findings=key_findings or ["Research completed"],
            sources_consulted=sources or ["Direct model knowledge"],
            confidence_score=0.85 if sources else 0.75,
            model_used=used_model,
            timestamp=datetime.now().isoformat()
        )
    
    async def compare_models(self, query: ResearchQuery) -> Dict[str, ResearchResult]:
        """Run the same research across all available models"""
        
        results = {}
        
        for model_name in self.tool_models:
            try:
                result = await self.research(query, model_name)
                results[model_name] = result
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        return results

async def demonstrate_research_assistant(models: Dict[str, BaseChatModel]):
    """Demonstrate the multi-model research assistant"""
    
    assistant = MultiModelResearchAssistant(models)
    
    if not assistant.tool_models:
        print("No tool-capable models available for research assistant!")
        print("Note: This demo requires models with function calling support.\n")
        return
    
    print(f"Research assistant initialized with models: {list(assistant.tool_models.keys())}\n")
    
    # Example 1: Quick research
    print("=== Quick Research Example ===\n")
    
    quick_query = ResearchQuery(
        topic="Large Language Models",
        depth="quick",
        focus_areas=["applications", "limitations"]
    )
    
    try:
        result = await assistant.research(quick_query)
        print(f"Topic: {result.topic}")
        print(f"Model used: {result.model_used}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"\nSummary: {result.summary[:200]}...")
        print(f"\nKey findings:")
        for finding in result.key_findings[:3]:
            print(f"- {finding}")
        print(f"\nSources: {', '.join(result.sources_consulted)}")
    except Exception as e:
        print(f"Quick research error: {e}")
    
    # Example 2: Deep research
    print("\n\n=== Deep Research Example ===\n")
    
    deep_query = ResearchQuery(
        topic="AI Safety and Alignment",
        depth="deep",
        focus_areas=["current challenges", "proposed solutions", "research directions"]
    )
    
    try:
        result = await assistant.research(deep_query)
        print(f"Topic: {result.topic}")
        print(f"Model used: {result.model_used}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Timestamp: {result.timestamp}")
        print(f"\nDetailed findings:")
        for i, finding in enumerate(result.key_findings, 1):
            print(f"{i}. {finding}")
        print(f"\nSources consulted: {len(result.sources_consulted)}")
        for source in result.sources_consulted:
            print(f"  - {source}")
    except Exception as e:
        print(f"Deep research error: {e}")
    
    # Example 3: Model comparison
    if len(assistant.tool_models) > 1:
        print("\n\n=== Model Comparison ===\n")
        
        compare_query = ResearchQuery(
            topic="Quantum Computing Applications",
            depth="moderate",
            focus_areas=["cryptography", "drug discovery"]
        )
        
        print(f"Comparing research on: {compare_query.topic}")
        print("This may take a moment...\n")
        
        try:
            comparison_results = await assistant.compare_models(compare_query)
            
            for model_name, result in comparison_results.items():
                print(f"\n{model_name.upper()} Results:")
                print(f"- Confidence: {result.confidence_score:.2f}")
                print(f"- Sources used: {len(result.sources_consulted)}")
                print(f"- Key findings: {len(result.key_findings)}")
                print(f"- First finding: {result.key_findings[0] if result.key_findings else 'None'}")
            
            # Find best result
            best_model = max(comparison_results.items(), 
                           key=lambda x: x[1].confidence_score)
            print(f"\nBest result from: {best_model[0]} "
                  f"(confidence: {best_model[1].confidence_score:.2f})")
            
        except Exception as e:
            print(f"Comparison error: {e}")
    
    print("\n✅ Research assistant demonstration complete!")
