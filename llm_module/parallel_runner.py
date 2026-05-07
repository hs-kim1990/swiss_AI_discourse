"""
Parallel runner for classifying documents across multiple vLLM servers.

Documents are distributed via a shared queue so that if one server crashes
mid-run its unprocessed items are automatically picked up by the remaining
healthy workers (fallback without data loss).

Usage:
    from llm_module.config import Config
    from llm_module.parallel_runner import parallel_classify

    config = Config(...)
    results, summary = parallel_classify(
        config,
        vllm_base_urls=["http://localhost:8090", "http://localhost:8091"],
        documents=documents,   # pre-loaded / pre-sampled list
        mode="all",
    )
"""
import copy
import json
import logging
import queue
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

# Number of consecutive all-None results before we suspect a server crash
_NULL_STREAK_THRESHOLD = 5
# Number of additional connection test attempts before declaring a server dead
_MAX_CONN_RETRIES = 3
# Max per-document retry attempts across all servers before giving up
_MAX_DOC_RETRIES = 2


def _generate_summary_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    category_counts: Dict[str, int] = {}
    subtopic_counts: Dict[str, int] = {}
    sentiment_counts: Dict[str, int] = {}
    successful = 0

    for r in results:
        classification = r.get("classification")
        sentiment = r.get("sentiment")
        subtopic = r.get("subtopic")
        if classification is not None:
            successful += 1
            category_counts[classification] = category_counts.get(classification, 0) + 1
        if sentiment is not None:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        if subtopic is not None:
            subtopic_counts[subtopic] = subtopic_counts.get(subtopic, 0) + 1

    summary = {
        "total_documents": total,
        "successfully_classified": successful,
        "failed": total - successful,
        "category_distribution": category_counts,
        "success_rate": successful / total if total > 0 else 0,
        "sentiment_distribution": sentiment_counts,
    }
    if subtopic_counts:
        summary["subtopic_distribution"] = subtopic_counts
    return summary


def parallel_classify(
    config: Config,
    vllm_base_urls: List[str],
    documents: Optional[List[Dict[str, Any]]] = None,
    text_fields: Optional[List[str]] = None,
    mode: str = "all",
    max_workers: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run classification in parallel across multiple vLLM servers.

    Args:
        config: Config object (must set `input_csv_file` and `output_folder`).
        vllm_base_urls: List of vLLM server base URLs, e.g. ["http://host:port"].
        documents: Pre-loaded (and optionally pre-sampled) document list.
                   If None, documents are loaded from config.input_csv_file.
        text_fields: Which fields to extract for text (passed to DocumentLoader).
        mode: One of ["all","classify","summarize","verify","sentiment"].
        max_workers: Max threads. Defaults to number of URLs.
        output_path: Optional explicit output path for the JSON results file.

    Returns:
        (combined_results_list, summary_dict)

    Fallback behaviour:
        Each worker pulls documents from a shared queue. If a server is detected
        as unavailable (consecutive null results confirmed by a connection test),
        that worker exits without consuming further items. The remaining items
        stay in the queue and are processed by healthy workers.
        Any documents still unprocessed after all workers finish are collected
        and added as stub records with an "error" field so nothing is lost.
    """
    if not vllm_base_urls:
        raise ValueError("Provide at least one vLLM base URL")

    # ── Load documents if not provided ────────────────────────────────────────
    if documents is None:
        loader = DocumentLoader(config.input_csv_file)
        documents = loader.load_all_documents()

    if not documents:
        logger.warning("No documents to classify")
        return [], {}

    # ── Shared work queue ─────────────────────────────────────────────────────
    # Each item is (doc, attempt) where attempt starts at 0.
    work_queue: queue.Queue = queue.Queue()
    for doc in documents:
        work_queue.put((doc, 0))

    total_docs = len(documents)
    if max_workers is None:
        max_workers = len(vllm_base_urls)

    # ── Shared state ──────────────────────────────────────────────────────────
    lock = threading.Lock()
    combined_results: List[Dict[str, Any]] = []
    processed_counter = {"count": 0}
    dead_servers: set = set()

    progress = tqdm(total=total_docs, desc="Processing documents") if tqdm is not None else None

    # ── Worker function ───────────────────────────────────────────────────────
    def _worker(url: str) -> None:
        cfg_copy = copy.deepcopy(config)
        cfg_copy.vllm_base_url = url
        classifier = LLMClassifier(cfg_copy)

        # Initial connection check — skip immediately if server is unreachable
        if not classifier.test_connection():
            logger.error(f"[{url}] Server unreachable at startup. Worker skipped.")
            with lock:
                dead_servers.add(url)
            return

        null_streak = 0  # consecutive all-None results

        while True:
            try:
                doc, attempt = work_queue.get_nowait()
            except queue.Empty:
                break

            res = classifier.classify_document(doc, text_fields=text_fields, mode=mode)

            # ── Detect server failure via null result streak ───────────────
            result_values = [
                res.get("classification"),
                res.get("verification"),
                res.get("summary"),
                res.get("sentiment"),
            ]
            all_none = all(v is None for v in result_values)

            if all_none:
                null_streak += 1
                logger.warning(
                    f"[{url}] Null result #{null_streak} for doc {doc.get('row_index')} "
                    f"(attempt {attempt + 1}/{_MAX_DOC_RETRIES + 1})"
                )
                if null_streak >= _NULL_STREAK_THRESHOLD:
                    # Confirm crash with a connection test
                    conn_ok = False
                    for conn_attempt in range(_MAX_CONN_RETRIES):
                        if classifier.test_connection():
                            conn_ok = True
                            break
                        logger.warning(
                            f"[{url}] Connection test failed (attempt {conn_attempt + 1}/{_MAX_CONN_RETRIES})"
                        )
                    if not conn_ok:
                        logger.error(
                            f"[{url}] Server confirmed down after {null_streak} null results. "
                            f"Returning doc {doc.get('row_index')} (attempt {attempt + 1}) to queue. "
                            f"~{work_queue.qsize()} docs remain for other workers."
                        )
                        work_queue.put((doc, attempt))  # preserve attempt count for other workers
                        with lock:
                            dead_servers.add(url)
                        return
                    else:
                        null_streak = 0  # server alive — the Nones were genuine

                # Per-doc retry: re-queue if under the retry limit
                if attempt < _MAX_DOC_RETRIES:
                    work_queue.put((doc, attempt + 1))
                    continue  # discard this result; do not record yet
                else:
                    logger.error(
                        f"[{url}] Doc {doc.get('row_index')} failed after "
                        f"{_MAX_DOC_RETRIES + 1} attempts. Recording as error."
                    )
                    res["error"] = "all_retries_exhausted"
            else:
                null_streak = 0

            with lock:
                combined_results.append(res)
                processed_counter["count"] += 1
                count = processed_counter["count"]

            if progress:
                progress.update(1)

            if count % 100 == 0:
                try:
                    with lock:
                        interim = _generate_summary_from_results(list(combined_results))
                    print(f"\n--- Interim summary after {count} documents ---")
                    print(f"Processed: {interim['total_documents']}")
                    print(f"Success rate: {interim['success_rate']:.2%}")
                    print(f"Category distribution: {interim['category_distribution']}")
                    print(f"Sentiment distribution: {interim.get('sentiment_distribution', {})}\n")
                except Exception as e:
                    logger.error(f"Failed to generate interim summary: {e}")

        logger.info(f"[{url}] Worker finished normally.")

    # ── Launch workers ────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, url): url for url in vllm_base_urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"[{url}] Worker raised unhandled exception: {e}")

    if progress:
        progress.close()

    # ── Collect any remaining unprocessed docs (all servers failed) ───────────
    unprocessed = []
    while not work_queue.empty():
        try:
            doc, _attempt = work_queue.get_nowait()
            unprocessed.append(doc)
        except queue.Empty:
            break

    if unprocessed:
        logger.error(
            f"{len(unprocessed)} document(s) could not be processed "
            f"(no healthy vLLM server available). Adding as error stubs."
        )
        for doc in unprocessed:
            combined_results.append(
                {
                    "id": doc.get("id"),
                    "source_file": doc.get("source_file"),
                    "row_index": doc.get("row_index"),
                    "head": doc.get("head"),
                    "error": "All vLLM servers unavailable",
                }
            )

    if dead_servers:
        logger.warning(f"Servers declared dead during run: {dead_servers}")

    # ── Persist results ───────────────────────────────────────────────────────
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_folder / f"classification_results_parallel_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Combined results saved to {output_path}")

    summary = _generate_summary_from_results(combined_results)
    summary_path = Path(output_path).with_name(Path(output_path).stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    return combined_results, summary
