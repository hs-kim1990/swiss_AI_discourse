"""
main entry point for actual running of the classification pipeline
"""


from llm_module import Config, DocumentLoader, LLMClassifier

def main():
    """Main function to run the classification pipeline"""
    
    config = Config(
        vllm_base_url="http://localhost:8000",
        model_name="llama",
        input_csv_file="./data/politics/cleaned_trial_data.csv",
        output_folder="./results",
        categories=["Asylum", "Integration", "Economy", "Politics", "Security"],
        system_prompt="You are an expert document summarizer and classifier specializing in news articles."
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
    
    # get 100 random items from list object documents
    # import random
    # documents = random.sample(documents, min(100, len(documents)))

    # Classify with custom processing
    results = []
    for doc in documents:
        # Extract specific fields
        text = loader.extract_text_content(doc, text_fields=["head", "content"])
        
        # Classify - use_chat=False to use completions API instead
        result = classifier.classify_document(doc, text_fields=["head", "content"], use_chat=True)
        results.append(result)
        
        doc_id = doc.get('id', 'unknown')
        # Handle None result from failed API calls
        if result['classification'] is not None:
            print(f"head: {doc.get('head')}")
            print(f"Classified row {doc['row_index']} (ID: {doc_id}): {result['classification']}")
            print(f"Summary: {result['summary']}")
        else:
            print(f"Failed to classify row {doc['row_index']} (ID: {doc_id})")
    
    # Save results
    classifier.save_results(results)
    
    # Generate and print summary
    summary = classifier.generate_summary(results)
    print("\nSummary:", summary)

    
    # Print summary
    print("\n=== Classification Summary ===")
    print(f"Total documents: {summary['total_documents']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Category distribution: {summary['category_distribution']}")


if __name__ == "__main__":
    main()