"""
Main entry point for the classification pipeline.

Two topic-modeling modes:
  --topic_mode explore  (default)
      Seed-and-expand: the classifier is given an initial topic list and may
      create new topics when none of the existing ones fit.

  --topic_mode utilize
      Fixed taxonomy: articles are assigned to one of the 10 predefined
      categories (from subgroup_taxonomy.json), then a second LLM step
      assigns a subgroup within that category.
      Default taxonomy file: ./results/subgroup_taxonomy.json
      Override with --taxonomy_file <path>.

Parallelisation:
  --vllm_urls "http://host1:port,http://host2:port"
      Distributes documents across servers via a shared queue. If a server
      crashes mid-run its remaining items are picked up by healthy workers
      automatically (fallback without data loss).

Sampling:
  --sample N
      Randomly sample N documents before processing (works in both single-
      server and multi-server mode).
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from llm_module import Config, DocumentLoader, LLMClassifier

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EXPLORE_CATEGORIES = [
    "Labor Market Integration",
    "Asylum & Protection",
    "Social Cohesion",
]

DEFAULT_UTILIZE_CATEGORIES = [
    "Swiss Domestic Politics & Governance",
    "Economy, Labor & Finance",
    "Immigration & Asylum Policy",
    "Swiss-EU Relations & Bilateral Agreements",
    "Refugee & Migrant Integration",
    "Defense, Security & Military",
    "Environment, Energy & Infrastructure",
    "Healthcare, Social Welfare & Education",
    "International Relations & Geopolitics",
    "Demographic Change & Population",
]

DEFAULT_TAXONOMY_FILE = "./results/subgroup_taxonomy.json"
DEFAULT_SUBTOPIC_MAP_FILE = "./results/category_subtopic_map.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_categories(categories_input: str) -> List[str]:
    """Parse categories from JSON array string or comma-separated string."""
    if not categories_input:
        return DEFAULT_EXPLORE_CATEGORIES

    parsed = None
    try:
        loaded = json.loads(categories_input)
        if isinstance(loaded, list):
            parsed = [str(item).strip() for item in loaded if str(item).strip()]
    except json.JSONDecodeError:
        pass

    if parsed is None:
        parsed = [item.strip() for item in categories_input.split(",") if item.strip()]

    if not parsed:
        raise ValueError("Categories cannot be empty after parsing.")
    return parsed


def load_taxonomy(taxonomy_file: str) -> Optional[Dict[str, Any]]:
    """Load subgroup taxonomy JSON. Returns None and logs a warning on failure."""
    path = Path(taxonomy_file)
    if not path.exists():
        logger.warning(f"Taxonomy file not found: {taxonomy_file}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_subtopic_map(subtopic_map_file: str):
    """Load flat category->subgroup mapping JSON. Returns None on missing file."""
    path = Path(subtopic_map_file)
    if not path.exists():
        logger.warning(f"Subtopic map file not found: {subtopic_map_file}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def categories_from_taxonomy(taxonomy: Optional[Dict[str, Any]]) -> List[str]:
    """Extract top-level category names from taxonomy dict."""
    if taxonomy and "taxonomy" in taxonomy:
        return list(taxonomy["taxonomy"].keys())
    return list(DEFAULT_UTILIZE_CATEGORIES)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server / model
    parser.add_argument("--url", type=int, default=8000,
                        help="Single vLLM server port (default: 8000). Ignored when --vllm_urls is set.")
    parser.add_argument("--model_name", type=str, default="llama",
                        help="Model name on vLLM server (default: llama).")

    # Input / output
    parser.add_argument("--input_file", type=str,
                        default="./data/politics/cleaned_trial_data.csv",
                        help="Input CSV/TSV file path.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON filename (written under ./results/).")

    # Categories (only relevant in explore mode; ignored in utilize mode)
    parser.add_argument("--categories", type=str,
                        default=",".join(DEFAULT_EXPLORE_CATEGORIES),
                        help=(
                            "Seed topic list for explore mode. "
                            "Comma-separated or JSON array. "
                            "Ignored when --topic_mode utilize is set."
                        ))

    # Pipeline mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "classify", "summarize", "verify", "sentiment",
                                 "supporting", "denying"],
                        help=(
                            "Processing steps to run (default: all). "
                            "'supporting': extract ideas supporting migration/integration. "
                            "'denying': extract ideas opposing migration/integration. "
                            "Both run automatically in 'all' mode for verified migration articles."
                        ))

    # Topic modeling mode
    parser.add_argument("--topic_mode", type=str, default="explore",
                        choices=["explore", "utilize"],
                        help=(
                            "Topic modeling mode. "
                            "'explore': seed-and-expand (default). "
                            "'utilize': fixed predefined taxonomy + subtopic step."
                        ))

    # Taxonomy file (utilize mode)
    parser.add_argument("--taxonomy_file", type=str, default=DEFAULT_TAXONOMY_FILE,
                        help=(
                            f"Path to subgroup_taxonomy.json for utilize mode "
                            f"(default: {DEFAULT_TAXONOMY_FILE})."
                        ))

    # Subtopic map file (utilize mode)
    parser.add_argument("--subtopic_map_file", type=str, default=DEFAULT_SUBTOPIC_MAP_FILE,
                        help=(
                            "Path to category_subtopic_map.json for subtopic classification "
                            "in utilize mode (default: ./results/category_subtopic_map.json)."
                        ))

    # Parallel execution
    parser.add_argument("--vllm_urls", type=str, default=None,
                        help=(
                            "Comma-separated vLLM base URLs for parallel execution. "
                            "Example: 'http://localhost:8090,http://localhost:8091'"
                        ))
    parser.add_argument("--workers", type=int, default=None,
                        help="Max worker threads for parallel run (defaults to number of URLs).")

    # Sampling
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N documents before processing.")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve categories and taxonomy ───────────────────────────────────────
    taxonomy: Optional[Dict[str, Any]] = None

    subtopic_map = None

    if args.topic_mode == "utilize":
        taxonomy = load_taxonomy(args.taxonomy_file)
        categories = categories_from_taxonomy(taxonomy)
        subtopic_map = load_subtopic_map(args.subtopic_map_file)
        if subtopic_map:
            print(f"[utilize] Loaded subtopic map from '{args.subtopic_map_file}' "
                  f"({len(subtopic_map)} categories mapped to subgroups).")
        else:
            print("[utilize] Subtopic map file not found -- subtopic assignment disabled.")
        if taxonomy:
            print(f"[utilize] Loaded taxonomy from '{args.taxonomy_file}' "
                  f"({len(categories)} categories).")
        else:
            print(f"[utilize] Taxonomy not found -- using {len(categories)} hard-coded categories.")
    else:
        categories = parse_categories(args.categories)

    # ── Build Config ──────────────────────────────────────────────────────────
    config = Config(
        vllm_base_url=f"http://localhost:{args.url}",
        model_name=args.model_name,
        input_csv_file=args.input_file,
        output_folder="./results",
        categories=categories,
        topic_mode=args.topic_mode,
        subtopic_taxonomy=taxonomy,
        subtopic_map=subtopic_map,
        system_prompt=(
            "You are an expert document summarizer and classifier "
            "specializing in Swiss news articles."
        ),
    )

    # ── Resolve output path ───────────────────────────────────────────────────
    output_path: Optional[str] = None
    if args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(args.output_file).stem
        suffix = Path(args.output_file).suffix or ".json"
        output_path = str(Path(config.output_folder) / f"{stem}_{timestamp}{suffix}")

    # ── Load documents (once, before parallel split or single run) ────────────
    loader = DocumentLoader(config.input_csv_file)
    documents = loader.load_all_documents()

    # Detect corpus language from the first document (whole file is same language)
    _LANG_NAMES = {"de": "German", "fr": "French", "it": "Italian"}
    doc_language = "de"
    if documents:
        detected = (documents[0].get("language") or "de").lower().strip()
        doc_language = detected if detected in _LANG_NAMES else "de"
        print(f"Detected document language: {doc_language} ({_LANG_NAMES.get(doc_language, doc_language)})")
    config.document_language = doc_language

    if args.sample:
        documents = random.sample(documents, min(args.sample, len(documents)))
        print(f"Sampled {len(documents)} documents.")

    # ── Parallel mode ─────────────────────────────────────────────────────────
    if args.vllm_urls:
        urls = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]
        if not urls:
            print("No valid vLLM URLs provided to --vllm_urls")
            return

        from llm_module.parallel_runner import parallel_classify

        results, summary = parallel_classify(
            config,
            vllm_base_urls=urls,
            documents=documents,          # pass pre-sampled docs
            text_fields=["head", "content"],
            mode=args.mode,
            max_workers=args.workers,
            output_path=output_path,
        )

        _print_summary(summary, args.topic_mode)
        return

    # ── Single-server mode ────────────────────────────────────────────────────
    classifier = LLMClassifier(config)
    if not classifier.test_connection():
        print("Failed to connect to vLLM server")
        return

    results = []
    iterator = tqdm(documents, desc="Processing documents") if tqdm is not None else documents
    for i, doc in enumerate(iterator):
        result = classifier.classify_document(
            doc, text_fields=["head", "content"], mode=args.mode
        )
        results.append(result)

        processed = i + 1
        if processed % 100 == 0:
            try:
                interim = classifier.generate_summary(results)
                print(f"\n--- Interim summary after {processed} documents ---")
                print(f"Processed: {interim['total_documents']}")
                print(f"Success rate: {interim['success_rate']:.2%}")
                print(f"Category distribution: {interim['category_distribution']}")
                print(f"Verification rate: {interim.get('verification_rate', 0):.2%}")
                print(f"Sentiment distribution: {interim.get('sentiment_distribution', {})}\n")
            except Exception as e:
                print(f"Failed to generate interim summary at {processed} documents: {e}")

    classifier.save_results(results, output_path=output_path)
    summary = classifier.generate_summary(results)
    _print_summary(summary, args.topic_mode)


def _print_summary(summary: Dict[str, Any], topic_mode: str) -> None:
    print("\n=== Classification Summary ===")
    print(f"Total documents:      {summary['total_documents']}")
    print(f"Success rate:         {summary.get('success_rate', 0):.2%}")
    print(f"Category distribution:{summary.get('category_distribution', {})}")
    if topic_mode == "utilize" and "subtopic_distribution" in summary:
        print(f"Subtopic distribution:{summary['subtopic_distribution']}")
    print(f"Verified articles:    {summary.get('verified_articles', 'n/a')}")
    print(f"Verification rate:    {summary.get('verification_rate', 0):.2%}")
    print(f"Sentiment distribution:{summary.get('sentiment_distribution', {})}")
    other_topics = summary.get("other_topics", {})
    if other_topics:
        top10 = sorted(other_topics.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 non-migration topics:")
        for topic, count in top10:
            print(f"  - {topic}: {count}")


if __name__ == "__main__":
    main()
