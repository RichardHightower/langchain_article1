"""Main entry point - runs all examples sequentially"""

import asyncio
from src.config import config
from src.basic_chat import demonstrate_basic_chat
from src.lcel_pipelines import demonstrate_lcel_pipelines
from src.structured_outputs import demonstrate_structured_outputs
from src.tool_usage import demonstrate_tool_usage
from src.research_assistant import demonstrate_research_assistant

def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

async def main():
    """Run all demonstrations"""
    print("LangChain Multi-Model Examples")
    print("==============================")
    print(f"Primary provider: {config.provider}")
    
    # Get available models
    models = config.get_all_models()
    if not models:
        print("No models available! Please check your configuration.")
        return
        
    print(f"\nAvailable models: {list(models.keys())}")
    
    # Run demonstrations
    print_section("1. Basic Chat Interactions")
    await demonstrate_basic_chat(models)
    
    print_section("2. LCEL Pipelines")
    await demonstrate_lcel_pipelines(models)
    
    print_section("3. Structured Outputs")
    await demonstrate_structured_outputs(models)
    
    print_section("4. Tool Usage")
    await demonstrate_tool_usage(models)
    
    print_section("5. Research Assistant")
    await demonstrate_research_assistant(models)
    
    print("\n✅ All demonstrations complete!")

if __name__ == "__main__":
    asyncio.run(main())
