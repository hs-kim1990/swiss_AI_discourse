"""
main entry point for actual running of the classification pipeline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from llm_module import Config, DocumentLoader, LLMClassifier


DEFAULT_CATEGORIES = ["Labor Market Integration", "Asylum & Protection", "Social Cohesion"]


def parse_categories(categories_input: str):
    """Parse categories from JSON array string or comma-separated string."""
    if not categories_input:
        return DEFAULT_CATEGORIES

    parsed_categories = None
    try:
        loaded = json.loads(categories_input)
        if isinstance(loaded, list):
            parsed_categories = [str(item).strip() for item in loaded if str(item).strip()]
    except json.JSONDecodeError:
        parsed_categories = None

    if parsed_categories is None:
        parsed_categories = [item.strip() for item in categories_input.split(",") if item.strip()]

    if not parsed_categories:
        raise ValueError("Categories cannot be empty after parsing.")

    return parsed_categories


def parse_args():
    """Parse CLI arguments for classifier execution."""
    parser = argparse.ArgumentParser(
        description="Run LLM classification pipeline with configurable inputs."
    )
    parser.add_argument(
        "--url",
        type=int,
        default=8000,
        help="vLLM server port number (default: 8000). Example: 9000",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama",
        help="Model name on vLLM server (default: llama).",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/politics/cleaned_trial_data.csv",
        help="Input CSV file path.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON filename written under default output folder (`./results`).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help=(
            "Categories as comma-separated string or JSON array string. "
            "Example CSV: 'Asylum,Integration,Economy' "
            "Example JSON: '[\"Asylum\",\"Integration\",\"Economy\"]'"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "classify", "summarize", "verify"],
        help="Processing mode passed to classifier.classify_document().",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default = None,
        help =(
            "Sample extractions for testing."
        )
    )
    return parser.parse_args()

def main():
    """Main function to run the classification pipeline"""

    args = parse_args()

    categories = parse_categories(args.categories)

    config = Config(
        vllm_base_url=f"http://localhost:{args.url}",
        model_name=args.model_name,
        input_csv_file=args.input_file,
        output_folder="./results",
        categories=categories,
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
    
    # get 100 random items from list object documents if args contains variable sample.
    if args.sample:
        import random
        documents = random.sample(documents, min(args.sample, len(documents)))

    # Classify with custom processing
    results = []
    for doc in documents:
        result = classifier.classify_document(doc, text_fields=["head", "content"], mode=args.mode)
        results.append(result)

        doc_id = doc.get('id', 'unknown')
        classification = result.get("classification")
        summary_text = result.get("summary")
        verification = result.get("verification")

        if result:
            print(f"========================== head: {doc.get('head')} ==========================")
            if args.mode in ["all", "classify"] and classification is not None:
                print(f"Classified row {doc['row_index']} (ID: {doc_id}): {classification}")            
            if args.mode in ["all", "summarize"] and summary_text is not None:
                print(f"Summarized row {doc['row_index']} (ID: {doc_id}): {summary_text}")
            if args.mode in ["all", "verify"] and verification is not None:
                print(f"Verified row {doc['row_index']} (ID: {doc_id}): {verification}")
        else:
            print(f"Failed to process row {doc['row_index']} (ID: {doc_id})")

    output_path = None
    if args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = Path(args.output_file)
        output_path = (
            Path(config.output_folder)
            / f"{output_filename.stem}_{timestamp}{output_filename.suffix}"
        )

    # Save results
    classifier.save_results(results, output_path=str(output_path) if output_path else None)

    # Generate and print summary
    summary = classifier.generate_summary(results)

    
    # Print summary
    print("\n=== Classification Summary ===")
    print(f"Total documents: {summary['total_documents']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Category distribution: {summary['category_distribution']}")
    print(f"Verification rate: {summary['verified_articles']}")
    print(f"Verification rate: {summary['verification_rate']:.2%}")
    other_topics = summary.get("other_topics", {})
    top_other_topics = sorted(other_topics.items(), key=lambda item: item[1], reverse=True)[:10]
    print("Top 10 other topics (reverse sorted):")
    for topic, count in top_other_topics:
        print(f"- {topic}: {count}")
    


if __name__ == "__main__":
    main()