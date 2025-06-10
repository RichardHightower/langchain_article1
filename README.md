# LangChain Multi-Model: Building Intelligent AI Applications

This project contains working examples for the article "Building Intelligent AI Applications with LangChain: A Developer's Journey from Chat to Production"

## Overview

Learn how to build production-ready AI applications using LangChain with multiple model providers:

- Multi-model support (OpenAI, Anthropic Claude, Ollama)
- LangChain Expression Language (LCEL) patterns
- Structured outputs and tool usage
- Building intelligent agents and assistants
- Production deployment strategies

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API key for OpenAI or Anthropic (Claude) OR Ollama installed locally

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and configure your LLM provider:
    
    ```bash
    cp .env.example .env
    ```
    
3. Edit `.env` to select your provider and model:
    - For OpenAI: Set `LLM_PROVIDER=openai` and add your API key
    - For Claude: Set `LLM_PROVIDER=anthropic` and add your API key
    - For Ollama: Set `LLM_PROVIDER=ollama` (install Ollama and pull gemma:7b model first)
4. Run the setup task:
    
    ```bash
    task setup
    ```

## Supported LLM Providers

### OpenAI

- Model: gpt-4-turbo-preview
- Requires: OpenAI API key

### Anthropic (Claude)

- Model: claude-3-opus-20240229
- Requires: Anthropic API key

### Ollama (Local)

- Model: gemma:7b
- Requires: Ollama installed and gemma:7b model pulled
- Install: `brew install ollama` (macOS) or see [ollama.ai](https://ollama.ai/)
- Pull model: `ollama pull gemma:7b`

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # LLM configuration
│   ├── main.py                # Entry point with all examples
│   ├── basic_chat.py          # Basic chat interactions
│   ├── lcel_pipelines.py      # LCEL pipeline examples
│   ├── structured_outputs.py  # Structured output parsing
│   ├── tool_usage.py          # Tool/function calling
│   └── research_assistant.py  # Multi-model research system
├── tests/
│   └── test_modules.py        # Unit tests
├── .env.example               # Environment template
├── Taskfile.yml               # Task automation
└── pyproject.toml             # Poetry configuration
```

## Key Concepts Demonstrated

1. **Multi-Model Architecture**: Seamlessly switch between OpenAI, Claude, and local models
2. **LCEL Pipelines**: Build composable AI workflows with the pipe operator
3. **Structured Outputs**: Use Pydantic schemas for reliable, validated responses
4. **Tool Integration**: Enable LLMs to interact with external systems
5. **Production Patterns**: Cost optimization, error handling, and monitoring

## Running Examples

Run all examples:

```bash
task run
```

Or run individual modules:

```bash
task run-basic           # Basic chat examples
task run-lcel           # LCEL pipeline demos
task run-structured     # Structured output examples
task run-tools          # Tool usage demonstrations
task run-research       # Research assistant
```

Direct Python execution:

```bash
poetry run python src/main.py
poetry run python src/basic_chat.py
poetry run python src/lcel_pipelines.py
```

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run the main example script
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Virtual Environment Setup Instructions

### Prerequisites

1. Install pyenv (if not already installed):
    
    ```bash
    # macOS
    brew install pyenv
    
    # Linux
    curl https://pyenv.run | bash
    ```
    
2. Add pyenv to your shell:
    
    ```bash
    # Add to ~/.zshrc or ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    
    # Reload shell
    source ~/.zshrc
    ```

### Setup Steps

1. **Install Python 3.12.9**:
    
    ```bash
    pyenv install 3.12.9
    ```
    
2. **Navigate to your project directory**:
    
    ```bash
    cd /path/to/langchain-multimodel
    ```
    
3. **Set local Python version**:
    
    ```bash
    pyenv local 3.12.9
    ```
    
4. **Install Poetry** (if not installed):
    
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    
5. **Install project dependencies**:
    
    ```bash
    poetry install
    ```
    
6. **Activate the virtual environment**:
    
    ```bash
    poetry config virtualenvs.in-project true
    source .venv/bin/activate
    ```

### Alternative: If you have Go Task installed

Simply run:

```bash
brew install go-task
task setup
```

### Configure your LLM provider

1. **Copy the example env file**:
    
    ```bash
    cp .env.example .env
    ```
    
2. **Edit .env and set your provider**:
    
    ```bash
    # For OpenAI
    LLM_PROVIDER=openai
    OPENAI_API_KEY=your-key-here
    OPENAI_MODEL=gpt-4-turbo-preview
    
    # For Anthropic/Claude
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=your-key-here
    ANTHROPIC_MODEL=claude-3-opus-20240229
    
    # For Ollama (local)
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=gemma:7b
    # Make sure Ollama is running: ollama serve
    # Pull the model: ollama pull gemma:7b
    ```

### Verify setup

```bash
# Check Python version
python --version  # Should show 3.12.9

# Test imports
python -c "import langchain; print('LangChain installed successfully')"
```

### Run the example

```bash
poetry run python src/main.py
```

Note: main.py runs all examples sequentially with descriptive text and delimiters of output.

## Example Output

The examples demonstrate:

1. **Basic Chat**: Multi-model conversations with temperature control
2. **LCEL Pipelines**: Composable chains with parallel execution
3. **Structured Outputs**: Type-safe data extraction with Pydantic
4. **Tool Usage**: Weather, calculations, and custom functions
5. **Research Assistant**: Intelligent multi-model research system

## Troubleshooting

- **Ollama connection error**: Make sure Ollama is running (`ollama serve`)
- **API key errors**: Check your `.env` file has the correct keys
- **Model not found**: For Ollama, ensure you've pulled the model (`ollama pull gemma:7b`)
- **Import errors**: Run `poetry install` to ensure all dependencies are installed

## Learn More

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai)

## Security Best Practices

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Set spending limits with your API providers
- Validate and sanitize all LLM outputs before execution
- Monitor usage with LangSmith for production deployments
