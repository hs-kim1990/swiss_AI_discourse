# LLM Document Classification Framework

A Python framework for classifying JSON documents using local LLM models via vLLM server.

## Features

- 🚀 **vLLM Integration**: Connect to local LLM models (Llama, Mistral, etc.) via vLLM server
- 📁 **Batch Processing**: Efficiently process multiple documents in batches
- 🎯 **Flexible Classification**: Support for custom categories and prompts
- 📊 **Result Tracking**: Automatic saving and summary statistics
- ⚙️ **Configurable**: Easy configuration via Python or JSON files
- 🔄 **Multiple APIs**: Support for both chat completions and completions endpoints

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Start your vLLM server:
```bash
# Example: Start vLLM server with Llama model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

## Quick Start

### Basic Usage

```python
from llm_module import Config, ClassificationPipeline

# Configure the classifier
config = Config(
    vllm_base_url="http://localhost:8000",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    input_folder="./documents",
    output_folder="./results",
    categories=["positive", "negative", "neutral"],
    batch_size=10
)

# Run classification pipeline
pipeline = ClassificationPipeline(config)
results = pipeline.run(
    text_fields=["title", "content"],  # Fields to extract from JSON
    save_results=True
)

# View summary
print(results['summary'])
```

### Using Configuration File

Create a `config.json`:
```json
{
  "vllm_base_url": "http://localhost:8000",
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "input_folder": "./documents",
  "output_folder": "./results",
  "categories": ["technology", "health", "business"],
  "temperature": 0.1,
  "max_tokens": 512,
  "batch_size": 10,
  "system_prompt": "You are an expert document classifier."
}
```

Load and use:
```python
from llm_module import Config, ClassificationPipeline

config = Config.from_json("config.json")
pipeline = ClassificationPipeline(config)
results = pipeline.run()
```

## Architecture

### Components

1. **Config** (`config.py`): Configuration management
   - vLLM server settings
   - Model parameters
   - Classification categories
   - Custom prompts

2. **DocumentLoader** (`document_loader.py`): Document handling
   - Load JSON documents from folder
   - Batch processing
   - Text extraction from specific fields

3. **LLMClassifier** (`llm_classifier.py`): Classification logic
   - Connect to vLLM server
   - Build prompts
   - Classify documents
   - Save results

4. **ClassificationPipeline** (`pipeline.py`): End-to-end workflow
   - Orchestrate the entire process
   - Handle batching
   - Generate summaries

## Advanced Usage

### Custom Processing

```python
from llm_module import Config, DocumentLoader, LLMClassifier

config = Config(
    vllm_base_url="http://localhost:8000",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    input_folder="./documents",
    categories=["urgent", "normal", "low_priority"]
)

# Initialize components
loader = DocumentLoader(config.input_folder)
classifier = LLMClassifier(config)

# Process documents with custom logic
documents = loader.load_all_documents()
for doc in documents:
    text = loader.extract_text_content(doc, text_fields=["subject", "message"])
    result = classifier.classify_document(doc)
    print(f"{doc['file_name']}: {result['classification']['category']}")
```

### Classify Single Document

```python
pipeline = ClassificationPipeline(config)
result = pipeline.run_single_document(
    document_path="./documents/sample.json",
    text_fields=["title", "body"]
)
print(f"Category: {result['classification']['category']}")
```

### Using Completions API (Alternative)

```python
# Use completions endpoint instead of chat
results = pipeline.run(use_chat=False)
```

## Document Format

Expected JSON document structure:
```json
{
  "title": "Document title",
  "content": "Main content here...",
  "author": "Author name",
  "date": "2024-01-01"
}
```

You can specify which fields to use for classification:
```python
results = pipeline.run(text_fields=["title", "content"])
```

## Output Format

Classification results are saved as JSON:
```json
[
  {
    "file_name": "doc1.json",
    "file_path": "/path/to/doc1.json",
    "classification": {
      "category": "positive",
      "raw_response": "positive",
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "timestamp": "2024-01-01T12:00:00"
    },
    "original_content": {...}
  }
]
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_base_url` | str | `http://localhost:8000` | vLLM server URL |
| `model_name` | str | `meta-llama/Llama-2-7b-chat-hf` | Model identifier |
| `categories` | List[str] | `[]` | Classification categories |
| `temperature` | float | `0.1` | Sampling temperature |
| `max_tokens` | int | `512` | Maximum response tokens |
| `top_p` | float | `0.9` | Nucleus sampling parameter |
| `batch_size` | int | `10` | Documents per batch |
| `system_prompt` | str | Default classifier prompt | System prompt for LLM |

## Supported Models

Any model supported by vLLM:
- Llama 2, Llama 3
- Mistral, Mixtral
- Vicuna
- Qwen
- And more...

## Error Handling

The framework includes robust error handling:
- Connection testing before classification
- Invalid category matching
- Document loading failures
- API request retries (can be customized)

## Examples

See `example_usage.py` for complete examples:
- Basic classification
- Using config files
- Custom processing
- Single document classification

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
