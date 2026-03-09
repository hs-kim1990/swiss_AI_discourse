"""
Example usage of the LLM classification framework
"""
from llm_module import Config, DocumentLoader, LLMClassifier


def example_basic_usage():
    """Basic usage example"""
    
    # Create configuration
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_csv_file="./data/swiss_ai_discourse_articles_de.csv",
        output_folder="./results",
        categories=["positive", "negative", "neutral"],
        temperature=0.1,
        batch_size=10
    )
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(config)
    
    # Run classification
    results = pipeline.run(
        text_fields=["head", "content"],  # Use CSV column names
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
        input_csv_file="./data/politics/cleaned_trial_data.csv",
        output_folder="./results",
        categories=["technology", "health", "business", "sports", "politics"],
        system_prompt="You are an expert document classifier specializing in news articles."
    )
    
    # Initialize components separately
    loader = DocumentLoader(config.input_csv_file)
    classifier = LLMClassifier(config)
    
    # Test connection
    if not classifier.test_connection():
        print("Failed to connect to vLLM server")
        return
    
    # Load documents (CSV rows)
    documents = loader.load_all_documents()
    
    # Classify with custom processing
    results = []
    for doc in documents:
        # Extract specific fields
        text = loader.extract_text_content(doc, text_fields=["head", "content"])
        
        # Classify
        result = classifier.classify_document(doc, text_fields=["head", "content"])
        results.append(result)
        
        doc_id = doc.get('id', 'unknown')
        print(f"Classified row {doc['row_index']} (ID: {doc_id}): {result['classification']['category']}")
    
    # Save results
    classifier.save_results(results)
    
    # Generate and print summary
    summary = classifier.generate_summary(results)
    print("\nSummary:", summary)


def example_single_document():
    """Example classifying a single row from CSV"""
    
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_csv_file="./data/swiss_ai_discourse_articles_de.csv",
        categories=["urgent", "normal", "low_priority"]
    )
    
    pipeline = ClassificationPipeline(config)
    
    # Classify single row from CSV
    result = pipeline.run_single_document(
        document_path="./data/swiss_ai_discourse_articles_de.csv",
        row_index=0,  # First row
        text_fields=["head", "content"]
    )
    
    print(f"Category: {result['classification']['category']}")
    print(f"Raw response: {result['classification']['raw_response']}")


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()
