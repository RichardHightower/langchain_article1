"""Structured output examples with Pydantic models"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel

# Define structured models
class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or occupation")
    skills: List[str] = Field(description="List of key skills")

class CompanyInfo(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    founded_year: int = Field(description="Year the company was founded")
    headquarters: str = Field(description="Location of headquarters")
    key_products: List[str] = Field(description="Main products or services")
    employee_count: Optional[int] = Field(description="Approximate number of employees", default=None)

async def demonstrate_structured_outputs(models: Dict[str, BaseChatModel]):
    """Demonstrate structured output parsing"""
    
    print("=== Person Information Extraction ===\n")
    
    # Create parser
    person_parser = PydanticOutputParser(pydantic_object=Person)
    
    # Create prompt with format instructions
    person_prompt = ChatPromptTemplate.from_template(
        "Extract person information from the following text:\n"
        "{text}\n\n"
        "{format_instructions}"
    )
    
    test_text = """
    Sarah Johnson is a 32-year-old data scientist working at TechCorp. 
    She specializes in machine learning, data visualization, and Python programming. 
    Sarah also has strong skills in statistics and cloud computing.
    """
    
    # Test with each model
    for name, model in models.items():
        print(f"--- {name} ---")
        try:
            chain = person_prompt | model | person_parser
            
            result = await chain.ainvoke({
                "text": test_text,
                "format_instructions": person_parser.get_format_instructions()
            })
            
            print(f"Parsed result: {result}")
            print(f"Name: {result.name}")
            print(f"Skills: {', '.join(result.skills)}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n=== Company Information Extraction ===\n")
    
    # Company parser
    company_parser = PydanticOutputParser(pydantic_object=CompanyInfo)
    
    company_prompt = ChatPromptTemplate.from_template(
        "Extract company information from this text:\n"
        "{text}\n\n"
        "{format_instructions}"
    )
    
    company_text = """
    OpenAI is an AI research company founded in 2015. Based in San Francisco, 
    the company is known for developing GPT models, DALL-E image generation, 
    and the ChatGPT conversational AI. They employ hundreds of researchers 
    and engineers working on artificial general intelligence.
    """
    
    # Test with native structured output (where available)
    for name, model in models.items():
        print(f"--- {name} ---")
        
        # Try native structured output first
        try:
            if hasattr(model, 'with_structured_output'):
                structured_model = model.with_structured_output(CompanyInfo)
                simple_prompt = ChatPromptTemplate.from_template(
                    "Extract company information from: {text}"
                )
                chain = simple_prompt | structured_model
                
                result = await chain.ainvoke({"text": company_text})
                print(f"Native structured output: {result}")
                print(f"Products: {', '.join(result.key_products[:2])}...\n")
                continue
        except Exception:
            pass
        
        # Fall back to parser-based approach
        try:
            chain = company_prompt | model | company_parser
            
            result = await chain.ainvoke({
                "text": company_text,
                "format_instructions": company_parser.get_format_instructions()
            })
            
            print(f"Parser-based result: {result}")
            print(f"Founded: {result.founded_year}")
            print(f"Industry: {result.industry}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Multiple extractions
    print("\n=== Batch Extraction ===\n")
    
    batch_texts = [
        "John Doe, 28, software engineer skilled in Java and cloud architecture",
        "Jane Smith, 35, product manager with expertise in agile and data analysis",
        "Bob Wilson, 42, DevOps engineer specializing in Kubernetes and CI/CD"
    ]
    
    # Pick the best available model for batch processing
    best_model = models.get("openai") or models.get("anthropic") or list(models.values())[0]
    
    chain = person_prompt | best_model | person_parser
    
    print(f"Using {[k for k, v in models.items() if v == best_model][0]} for batch processing\n")
    
    extracted_people = []
    for text in batch_texts:
        try:
            person = await chain.ainvoke({
                "text": text,
                "format_instructions": person_parser.get_format_instructions()
            })
            extracted_people.append(person)
            print(f"✓ Extracted: {person.name} - {person.occupation}")
        except Exception as e:
            print(f"✗ Failed to extract from: {text[:30]}...")
    
    print(f"\nSuccessfully extracted {len(extracted_people)} people")

if __name__ == "__main__":
    import asyncio
    from src.config import setup_and_get_models
    
    async def main():
        models = setup_and_get_models()
        if models:
            print("\n" + "=" * 60)
            print("  Structured Outputs")
            print("=" * 60 + "\n")
            await demonstrate_structured_outputs(models)
            print("\n✅ Structured outputs demonstration complete!")
    
    asyncio.run(main())
