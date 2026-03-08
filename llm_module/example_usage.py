"""
Example usage of the LLM classification framework
"""
from llm_module import Config, DocumentLoader, LLMClassifier, ClassificationPipeline


def example_basic_usage():
    """Basic usage example"""
    
    # Create configuration
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_folder="./documents",
        output_folder="./results",
        categories=["positive", "negative", "neutral"],
        temperature=0.1,
        batch_size=10
    )
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(config)
    
    # Run classification
    results = pipeline.run(
        text_fields=["title", "content"],  # Specify which fields to use
        use_chat=True,
        save_results=True
    )
    
    # Print summary
    print("\n=== Classification Summary ===")
    print(f"Total documents: {results['summary']['total_documents']}")
    print(f"Success rate: {results['summary']['success_rate']:.2%}")
    print(f"Category distribution: {results['summary']['category_distribution']}")


def example_with_config_file():
    """Example using configuration file"""
    
    # Load config from file
    config = Config.from_json("config.json")
    
    # Run pipeline
    pipeline = ClassificationPipeline(config)
    results = pipeline.run()
    
    return results


def example_custom_classification():
    """Example with custom processing"""
    
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_folder="./documents",
        output_folder="./results",
        categories=["technology", "health", "business", "sports", "politics"],
        system_prompt="You are an expert document classifier specializing in news articles."
    )
    
    # Initialize components separately
    loader = DocumentLoader(config.input_folder)
    classifier = LLMClassifier(config)
    
    # Test connection
    if not classifier.test_connection():
        print("Failed to connect to vLLM server")
        return
    
    # Load documents
    documents = loader.load_all_documents()
    
    # Classify with custom processing
    results = []
    for doc in documents:
        # Extract specific fields
        text = loader.extract_text_content(doc, text_fields=["headline", "body"])
        
        # Classify
        result = classifier.classify_document(doc, text_fields=["headline", "body"])
        results.append(result)
        
        print(f"Classified {doc['file_name']}: {result['classification']['category']}")
    
    # Save results
    classifier.save_results(results)
    
    # Generate and print summary
    summary = classifier.generate_summary(results)
    print("\nSummary:", summary)


def example_single_document():
    """Example classifying a single document"""
    
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_folder="./documents",
        categories=["urgent", "normal", "low_priority"]
    )
    
    pipeline = ClassificationPipeline(config)
    
    # Classify single document
    result = pipeline.run_single_document(
        document_path="./documents/sample.json",
        text_fields=["subject", "message"]
    )
    
    print(f"Category: {result['classification']['category']}")
    print(f"Raw response: {result['classification']['raw_response']}")


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()
