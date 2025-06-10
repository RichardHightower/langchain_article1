"""Tool usage and function calling examples"""

import random
import json
from datetime import datetime
from typing import Dict
import math
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: City name to get weather for
        
    Returns:
        Weather description and temperature
    """
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    temp = random.randint(15, 30)
    return f"{city}: {random.choice(conditions)}, {temp}°C"

@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        allowed_names['abs'] = abs
        allowed_names['round'] = round
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone.
    
    Args:
        timezone: Timezone name (default UTC)
        
    Returns:
        Current time as string
    """
    # Simplified - returns local time
    return f"Current time in {timezone}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def search_knowledge(query: str) -> str:
    """Search internal knowledge base.
    
    Args:
        query: Search query
        
    Returns:
        Relevant information
    """
    # Simulated knowledge base
    knowledge = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "python": "Python is a high-level programming language known for its simplicity.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines."
    }
    
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    
    return f"No information found for '{query}'"

async def demonstrate_tool_usage(models: Dict[str, BaseChatModel]):
    """Demonstrate tool usage with different models"""
    
    tools = [get_weather, calculate, get_time, search_knowledge]
    
    print("=== Tool Availability Check ===\n")
    
    # Check which models support tools
    tool_capable_models = {}
    
    for name, model in models.items():
        try:
            model_with_tools = model.bind_tools(tools)
            tool_capable_models[name] = model_with_tools
            print(f"✓ {name}: Tool support available")
        except Exception as e:
            print(f"✗ {name}: No tool support - {type(e).__name__}")
    
    if not tool_capable_models:
        print("\nNo models with tool support available!")
        print("Note: Ollama models typically don't support native function calling.")
        print("Workaround: Use structured prompts to simulate tool usage.\n")
        return
    
    print("\n=== Simple Tool Usage ===\n")
    
    # Test simple tool calls
    test_queries = [
        "What's the weather in Paris?",
        "Calculate the area of a circle with radius 5 (use pi)",
        "What time is it in Tokyo?",
        "Tell me about LangChain"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        
        for name, model in tool_capable_models.items():
            print(f"\n{name}:")
            
            try:
                # Initial response
                response = await model.ainvoke(query)
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"Tool calls requested: {len(response.tool_calls)}")
                    
                    # Execute tools
                    messages = [HumanMessage(content=query), response]
                    
                    for tool_call in response.tool_calls:
                        print(f"- Calling {tool_call['name']} with {tool_call['args']}")
                        
                        # Execute tool
                        tool_fn = globals()[tool_call['name']]
                        result = tool_fn.invoke(tool_call['args'])
                        
                        # Add result
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call['id']
                        ))
                    
                    # Get final response
                    final_response = await model.ainvoke(messages)
                    print(f"Final answer: {final_response.content}")
                else:
                    print(f"Direct response: {response.content[:100]}...")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "-" * 50 + "\n")
    
    print("\n=== Multi-Tool Usage ===\n")
    
    complex_query = "What's the weather in London and New York, and what's 15% of 847?"
    
    for name, model in tool_capable_models.items():
        print(f"{name}:")
        
        try:
            response = await model.ainvoke(complex_query)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Tools requested: {[tc['name'] for tc in response.tool_calls]}")
                
                messages = [HumanMessage(content=complex_query), response]
                
                # Execute all tools
                for tool_call in response.tool_calls:
                    tool_fn = globals()[tool_call['name']]
                    result = tool_fn.invoke(tool_call['args'])
                    
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call['id']
                    ))
                
                # Final response
                final = await model.ainvoke(messages)
                print(f"Answer: {final.content}\n")
                
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Demonstrate fallback for non-tool models
    if len(tool_capable_models) < len(models):
        print("\n=== Fallback Strategy for Non-Tool Models ===\n")
        
        # Create a structured prompt that simulates tool usage
        fallback_prompt = """You are an AI assistant. When asked about weather, time, or calculations,
        provide a response in this JSON format:
        {
            "tool_needed": "tool_name",
            "parameters": {"param": "value"},
            "explanation": "what you would do with this tool"
        }
        
        Available tools:
        - get_weather: Get weather for a city
        - calculate: Perform calculations
        - get_time: Get current time
        - search_knowledge: Search information
        
        Query: What's the weather in Berlin?"""
        
        # Try with a non-tool model
        non_tool_models = {k: v for k, v in models.items() if k not in tool_capable_models}
        
        if non_tool_models:
            model_name, model = list(non_tool_models.items())[0]
            print(f"Using {model_name} with structured prompt fallback:\n")
            
            try:
                response = await model.ainvoke(fallback_prompt)
                print(f"Response: {response.content}")
                
                # Try to parse the response
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                    if json_match:
                        tool_request = json.loads(json_match.group())
                        print(f"\nParsed tool request: {tool_request}")
                        
                        # Simulate tool execution
                        if tool_request.get('tool_needed') == 'get_weather':
                            city = tool_request.get('parameters', {}).get('city', 'Berlin')
                            result = get_weather.invoke({'city': city})
                            print(f"Tool result: {result}")
                except:
                    print("Could not parse structured response")
                    
            except Exception as e:
                print(f"Fallback error: {e}")

if __name__ == "__main__":
    import asyncio
    from src.config import setup_and_get_models
    
    async def main():
        models = setup_and_get_models()
        if models:
            print("\n" + "=" * 60)
            print("  Tool Usage")
            print("=" * 60 + "\n")
            await demonstrate_tool_usage(models)
            print("\n✅ Tool usage demonstration complete!")
    
    asyncio.run(main())
