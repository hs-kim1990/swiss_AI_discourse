"""
main entry point for actual running of the classification pipeline
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from llm_module import Config, DocumentLoader, LLMClassifier
from llm_module.llm_classifier import UTILIZE_CATEGORIES


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
        type=str,
        default="8000",
        help=(
            "vLLM server port(s), comma-separated for multi-server failover "
            "(default: 8000). Example: 8000,9000"
        ),
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
        default=None,
        help=(
            "Categories as comma-separated string or JSON array string. "
            "Defaults to seed topics in explore mode and predefined taxonomy in utilize mode. "
            "Example CSV: 'Asylum,Integration,Economy' "
            "Example JSON: '[\"Asylum\",\"Integration\",\"Economy\"]'"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "classify", "summarize", "verify", "sentiment"],
        help="Processing mode passed to classifier.classify_document().",
    )
    parser.add_argument(
        "--topic_mode",
        type=str,
        default="explore",
        choices=["explore", "utilize"],
        help=(
            "Topic modeling behavior. "
            "'explore': uses seed categories and expands when no match found (default). "
            "'utilize': assigns to predefined 10-category taxonomy; also runs a subtopic step."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N documents for testing.",
    )
    return parser.parse_args()


def main():
    """Main function to run the classification pipeline"""

    args = parse_args()

    # Resolve categories
    if args.categories:
        categories = parse_categories(args.categories)
    elif args.topic_mode == "utilize":
        # In utilize mode, default to the full predefined taxonomy
        categories = list(UTILIZE_CATEGORIES)
    else:
        categories = list(DEFAULT_CATEGORIES)

    # Build server list from comma-separated ports
    ports = [p.strip() for p in args.url.split(",") if p.strip()]
    vllm_servers = [f"http://localhost:{p}" for p in ports]

    config = Config(
        vllm_base_url=vllm_servers[0],
        vllm_servers=vllm_servers,
        model_name=args.model_name,
        input_csv_file=args.input_file,
        output_folder="./results",
        categories=categories,
        topic_modeling_mode=args.topic_mode,
        system_prompt="You are an expert document summarizer and classifier specializing in news articles.",
    )

    # Initialize components
    loader = DocumentLoader(config.input_csv_file)
    classifier = LLMClassifier(config)

    # Test connection (logs per-server status)
    if not classifier.test_connection():
        print("Failed to connect to any vLLM server")
        return

    # Load documents
    documents = loader.load_all_documents()

    if args.sample:
        documents = random.sample(documents, min(args.sample, len(documents)))

    # Classify
    results = []
    for doc in documents:
        result = classifier.classify_document(doc, text_fields=["head", "content"], mode=args.mode)
        results.append(result)

        doc_id = doc.get("id", "unknown")
        if result.get("error"):
            print(f"FAILED row {doc['row_index']} (ID: {doc_id}): {result['error']}")
            continue

        classification = result.get("classification")
        summary_text = result.get("summary")
        verification = result.get("verification")
        sentiment = result.get("sentiment")
        subtopic = result.get("subtopic")

        print(f"========================== head: {doc.get('head')} ==========================")
        if args.mode in ("all", "classify") and classification is not None:
            print(f"Classified row {doc['row_index']} (ID: {doc_id}): {classification}")
        if subtopic is not None:
            print(f"Subtopic row {doc['row_index']} (ID: {doc_id}): {subtopic}")
        if args.mode in ("all", "summarize") and summary_text is not None:
            print(f"Summarized row {doc['row_index']} (ID: {doc_id}): {summary_text}")
        if args.mode in ("all", "verify") and verification is not None:
            print(f"Verified row {doc['row_index']} (ID: {doc_id}): {verification}")
        if args.mode in ("all", "sentiment") and sentiment is not None:
            print(f"Sentiment for row {doc['row_index']} (ID: {doc_id}): {sentiment}")

    # Resolve output path
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

    # Print summary
    summary = classifier.generate_summary(results)

    print("\n=== Classification Summary ===")
    print(f"Total documents: {summary['total_documents']}")
    print(f"Failed (all servers down): {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Category distribution: {summary['category_distribution']}")
    print(f"Verified articles: {summary['verified_articles']}")
    print(f"Verification rate: {summary['verification_rate']:.2%}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")

    if args.topic_mode == "utilize":
        print(f"Subtopic distribution: {summary.get('subtopic_distribution', {})}")
    else:
        other_topics = summary.get("other_topics", {})
        top_other = sorted(other_topics.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 other topics (explore mode):")
        for topic, count in top_other:
            print(f"  - {topic}: {count}")


if __name__ == "__main__":
    main()
