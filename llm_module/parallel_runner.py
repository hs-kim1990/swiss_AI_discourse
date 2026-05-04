"""
Parallel runner for classifying documents across multiple vLLM servers.

This module splits the input documents into chunks and runs multiple
instances of `LLMClassifier` in parallel (one per vLLM server URL).

Usage sample:
from llm_module.config import Config
from llm_module.parallel_runner import parallel_classify

config = Config()  # make sure config.input_csv_file and other fields set
vllm_urls = ["http://localhost:8090", "http://localhost:8091"]
results, summary = parallel_classify(config, vllm_urls, mode='all')

The function returns (results_list, summary_dict) and writes a JSON file
in `config.output_folder` by default.
"""
import copy
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config import Config
from .document_loader import DocumentLoader
from .llm_classifier import LLMClassifier

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _chunk_documents(documents: List[Dict[str, Any]], n_chunks: int) -> List[List[Dict[str, Any]]]:
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    total = len(documents)
    if total == 0:
        return [[] for _ in range(n_chunks)]
    chunk_size = math.ceil(total / n_chunks)
    chunks = [documents[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    return chunks


def _generate_summary_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    category_counts: Dict[str, int] = {}
    sentiment_counts: Dict[str, int] = {}
    successful = 0

    for r in results:
        classification = r.get("classification")
        sentiment = r.get("sentiment")
        if classification is not None:
            successful += 1
            category_counts[classification] = category_counts.get(classification, 0) + 1
        if sentiment is not None:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    return {
        "total_documents": total,
        "successfully_classified": successful,
        "failed": total - successful,
        "category_distribution": category_counts,
        "success_rate": successful / total if total > 0 else 0,
        "sentiment_distribution": sentiment_counts,
    }


def parallel_classify(
    config: Config,
    vllm_base_urls: List[str],
    text_fields: Optional[List[str]] = None,
    mode: str = "all",
    max_workers: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run classification in parallel across multiple vLLM servers.

    Args:
        config: Config object (must set `input_csv_file` and `output_folder`).
        vllm_base_urls: List of vLLM server base URLs (e.g., http://host:port)
        text_fields: Which fields to extract for text (passed to DocumentLoader.extract_text_content)
        mode: Mode for classification - one of ["all","classify","summarize","verify","sentiment"]
        max_workers: Max threads to use. Defaults to number of URLs.
        output_path: Optional file path to save JSON results. If None, creates timestamped file in `config.output_folder`.

    Returns:
        (combined_results_list, summary_dict)
    """
    if not vllm_base_urls:
        raise ValueError("Provide at least one vLLM base URL to parallelize across")

    loader = DocumentLoader(config.input_csv_file)
    documents = loader.load_all_documents()
    if not documents:
        logger.warning("No documents loaded; nothing to classify")
        return [], {}

    n_workers = len(vllm_base_urls)
    chunks = _chunk_documents(documents, n_workers)

    if max_workers is None:
        max_workers = n_workers

    lock = threading.Lock()
    combined_results: List[Dict[str, Any]] = []
    processed_counter = {"count": 0}

    total_docs = len(documents)
    progress = tqdm(total=total_docs, desc="Processing documents") if tqdm is not None else None

    def _worker(url: str, docs_chunk: List[Dict[str, Any]]) -> None:
        if not docs_chunk:
            return
        cfg_copy = copy.deepcopy(config)
        # override vllm base URL for this worker
        setattr(cfg_copy, "vllm_base_url", url)
        classifier = LLMClassifier(cfg_copy)
        logger.info(f"Worker starting for {url} with {len(docs_chunk)} docs")

        local_results: List[Dict[str, Any]] = []
        for doc in docs_chunk:
            try:
                res = classifier.classify_document(doc, text_fields=text_fields, mode=mode)
            except Exception as e:
                logger.error(f"Failed to classify doc row {doc.get('row_index')}: {e}")
                res = {
                    "id": doc.get("id"),
                    "source_file": doc.get("source_file"),
                    "row_index": doc.get("row_index"),
                }

            with lock:
                combined_results.append(res)
                processed_counter["count"] += 1
                processed = processed_counter["count"]

            if progress:
                progress.update(1)

            # Every 100 processed docs, print an interim summary
            if processed % 100 == 0:
                try:
                    with lock:
                        interim = _generate_summary_from_results(list(combined_results))
                    print(f"\n--- Interim summary after {processed} documents ---")
                    print(f"Processed: {interim['total_documents']}")
                    print(f"Success rate: {interim['success_rate']:.2%}")
                    print(f"Category distribution: {interim['category_distribution']}")
                    print(f"Sentiment distribution: {interim.get('sentiment_distribution', {})}\n")
                except Exception as e:
                    logger.error(f"Failed to generate interim summary: {e}")

        logger.info(f"Worker finished for {url}")

    # Submit one worker per URL
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, url in enumerate(vllm_base_urls):
            chunk = chunks[i]
            futures.append(ex.submit(_worker, url, chunk))

        # Wait for all to finish
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Worker raised exception: {e}")

    if progress:
        progress.close()

    # Save combined results
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_folder / f"classification_results_parallel_{timestamp}.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Combined results saved to {output_path}")

    summary = _generate_summary_from_results(combined_results)

    # Save summary too
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    return combined_results, summary


if __name__ == "__main__":
    # Minimal CLI example (non-exhaustive). Adjust to your environment.
    import argparse
    parser = argparse.ArgumentParser(description="Parallel classify documents across vLLM servers")
    parser.add_argument("--config", required=False, help="Path to config module or config file (not implemented)")
    parser.add_argument("--urls", required=True, help="Comma-separated vLLM base URLs")
    parser.add_argument("--mode", default="all", help="Mode for classification")
    args = parser.parse_args()

    # NOTE: We cannot auto-load a Config file here generically. The user should integrate the
    # `parallel_classify` function into their `run.py` where a `Config` object is available.
    print("This script provides `parallel_classify(config, urls, ...)`. Import and call it from your run script.")
