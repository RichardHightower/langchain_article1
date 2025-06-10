"""Basic chat interactions with multiple models"""

from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

async def demonstrate_basic_chat(models: Dict[str, BaseChatModel]):
    """Demonstrate basic chat with all available models"""
    
    # Test message
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Explain what LangChain is in one sentence.")
    ]
    
    print("Testing basic chat with all models...\n")
    
    # Synchronous example
    for name, model in models.items():
        print(f"--- {name.upper()} Response ---")
        try:
            response = model.invoke(messages)
            print(f"{response.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Temperature comparison
    print("\n=== Temperature Comparison ===")
    prompt = "Write a creative haiku about programming"
    
    for temp in [0.1, 0.7, 1.2]:
        print(f"\nTemperature: {temp}")
        
        for name, model in models.items():
            # Clone model with new temperature
            try:
                if hasattr(model, 'temperature'):
                    # Use the config to recreate models with new temperature
                    from src.config import config
                    model_with_temp = config.get_model(name, temperature=temp)
                else:
                    model_with_temp = model
                    
                print(f"\n{name}:")
                response = model_with_temp.invoke(prompt)
                print(response.content)
            except Exception as e:
                print(f"Error with {name}: {e}")
    
    # Streaming example
    print("\n=== Streaming Example ===")
    stream_prompt = "Count from 1 to 5 slowly, with a description for each number."
    
    for name, model in models.items():
        print(f"\n{name} streaming:")
        try:
            async for chunk in model.astream(stream_prompt):
                print(chunk.content, end="", flush=True)
            print()  # New line after streaming
        except Exception as e:
            print(f"Streaming not supported or error: {e}")
