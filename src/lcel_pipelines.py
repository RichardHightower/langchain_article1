"""LCEL (LangChain Expression Language) pipeline examples"""

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


async def demonstrate_lcel_pipelines(models: Dict[str, BaseChatModel]):
    """Demonstrate LCEL pipeline patterns"""

    # Basic pipeline
    print("=== Basic LCEL Pipeline ===\n")

    prompt = ChatPromptTemplate.from_template(
        "Tell me a {adjective} fact about {topic}"
    )
    output_parser = StrOutputParser()

    # Create chains for each model
    for name, model in models.items():
        chain = prompt | model | output_parser

        try:
            result = await chain.ainvoke(
                {"adjective": "interesting", "topic": "artificial intelligence"}
            )
            print(f"{name}: {result}\n")
        except Exception as e:
            print(f"{name} error: {e}\n")

    # Parallel execution
    print("\n=== Parallel Execution ===\n")

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in 10 words or less: {text}"
    )

    # Create parallel chain with all models
    parallel_chains = {}
    for name, model in models.items():
        parallel_chains[name] = summary_prompt | model | output_parser
    parallel_chains["original"] = RunnablePassthrough()

    parallel = RunnableParallel(parallel_chains)

    test_text = {
        "text": "LangChain is a framework for developing applications "
        "powered by language models. It provides modular components "
        "and tools for building AI applications."
    }

    try:
        results = await parallel.ainvoke(test_text)
        print("Original text:", results["original"]["text"])
        print("\nSummaries:")
        for name, summary in results.items():
            if name != "original":
                print(f"- {name}: {summary}")
    except Exception as e:
        print(f"Parallel execution error: {e}")

    # Conditional routing
    print("\n\n=== Conditional Routing ===\n")

    def route_by_length(x):
        """Route based on input length"""
        # Handle both dict input and formatted prompt input
        if isinstance(x, dict):
            text = x.get("text", "")
        else:
            # If it's already a formatted prompt, extract text from it
            text = str(x) if hasattr(x, '__str__') else ""
        
        if len(text) < 50:
            return list(models.values())[0]  # First available model
        else:
            # Use the most capable model for longer texts
            if "openai" in models:
                return models["openai"]
            elif "anthropic" in models:
                return models["anthropic"]
            else:
                return list(models.values())[0]

    router = RunnableLambda(route_by_length)

    routing_prompt = ChatPromptTemplate.from_template(
        "Process this text appropriately: {text}"
    )

    print(routing_prompt)

    # Create a simpler routing chain that routes first, then formats
    def create_routed_chain(test_input):
        selected_model = route_by_length(test_input)
        return routing_prompt | selected_model | output_parser

    routed_chain = RunnableLambda(create_routed_chain)

    # Test with different length inputs
    test_inputs = [
        {"text": "Hello"},
        {
            "text": "This is a much longer text that contains multiple sentences "
            "and should be routed to a more capable model for processing."
        },
    ]

    for test_input in test_inputs:
        try:
            chain = create_routed_chain(test_input)
            result = await chain.ainvoke(test_input)
            print(f"Input length: {len(test_input['text'])}")
            print(f"Result: {result[:100]}...\n")
        except Exception as e:
            print(f"Routing error: {e}\n")


if __name__ == "__main__":
    import asyncio

    from src.config import setup_and_get_models

    async def main():
        models = setup_and_get_models()
        if models:
            print("\n" + "=" * 60)
            print("  LCEL Pipelines")
            print("=" * 60 + "\n")
            await demonstrate_lcel_pipelines(models)
            print("\n✅ LCEL pipelines demonstration complete!")

    asyncio.run(main())
