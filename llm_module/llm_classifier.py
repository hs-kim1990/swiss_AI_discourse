"""
LLM-based document classifier using vLLM server
"""
import logging
import re
import requests
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from .config import Config
from .document_loader import DocumentLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Predefined taxonomy for "utilize" topic_modeling_mode.
# Maps each major category to its ordered list of subgroups.
PREDEFINED_TAXONOMY: Dict[str, List[str]] = {
    "Swiss Domestic Politics & Governance": [
        "Government & Parliamentary Affairs",
        "Media, Transparency & Accountability",
        "Local & Regional Governance",
        "Culture, Society & Identity",
        "Civil Rights, Equality & Justice",
        "Elections & Party Politics",
    ],
    "Economy, Labor & Finance": [
        "Banking, Finance & Taxation",
        "Housing & Urban Development",
        "Other Economic Topics",
        "Agriculture, Industry & Business",
        "Trade & International Commerce",
        "Labor Market & Wages",
        "Social Benefits, Pensions & Welfare",
    ],
    "Immigration & Asylum Policy": [
        "Asylum Procedures & Reform",
        "Swiss Migration Policy",
        "EU & International Asylum Policy",
        "Migration Control & Border Policy",
        "Other Migration Topics",
        "Refugee Housing & Facilities",
    ],
    "Swiss-EU Relations & Bilateral Agreements": [
        "Core Switzerland-EU Framework",
        "Brexit & European Realignment",
        "EU Trade & Economic Cooperation",
        "EU Security & Defense",
        "EU Institutional Affairs",
        "Other EU Topics",
    ],
    "Refugee & Migrant Integration": [
        "Refugee Experiences & Stories",
        "Integration Programs & Policy",
        "Other Integration Topics",
        "Cultural & Social Integration",
        "Employment & Economic Integration",
        "Education & Youth Integration",
    ],
    "Defense, Security & Military": [
        "Military Policy & Procurement",
        "Security Policy & Intelligence",
        "Crime Prevention & Law Enforcement",
        "Disaster Response & Humanitarian",
        "Other Defense & Security Topics",
        "Defense Industry & Arms Exports",
    ],
    "Environment, Energy & Infrastructure": [
        "Energy Policy & Nuclear",
        "Climate & Environmental Protection",
        "Transportation & Infrastructure",
        "Other Environment & Infrastructure Topics",
        "Urban Planning & Development",
    ],
    "Healthcare, Social Welfare & Education": [
        "Healthcare Policy & Reform",
        "Education Policy & Reform",
        "Social Benefits & Family Policy",
        "Other Health & Welfare Topics",
        "End-of-Life, Mental Health & Disability",
    ],
    "International Relations & Geopolitics": [
        "Switzerland-Ukraine & Eastern Europe",
        "Swiss Neutrality & Foreign Aid",
        "Middle East & Africa Policy",
        "Switzerland-US Relations",
        "Switzerland-Asia Relations",
        "Other International Topics",
    ],
    "Demographic Change & Population": [
        "Aging Society & Population Trends",
        "Population & Immigration Demographics",
        "Fertility & Pronatalism",
    ],
}

UTILIZE_CATEGORIES = list(PREDEFINED_TAXONOMY.keys())


class LLMClassifier:
    """Classify documents using LLM via vLLM server"""

    def __init__(self, config: Config):
        self.config = config
        self.config.validate()

        # Server list for failover — populated from config.vllm_servers
        self.servers: List[str] = list(config.vllm_servers)

        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.correct_topics = 0
        self.other_topics = {}

    # ------------------------------------------------------------------
    # Connection / health
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test connection to all configured vLLM servers; returns True if at least one is reachable."""
        available = []
        for server_url in self.servers:
            try:
                response = requests.get(f"{server_url}/v1/models", headers=self.headers, timeout=5)
                response.raise_for_status()
                available.append(server_url)
                logger.info(f"Connected to server: {server_url}")
            except Exception as e:
                logger.warning(f"Server {server_url} unavailable: {e}")

        if not available:
            logger.error("No vLLM servers reachable")
            return False

        if len(available) < len(self.servers):
            down = set(self.servers) - set(available)
            logger.warning(f"Degraded mode — unreachable servers: {down}")

        return True

    # ------------------------------------------------------------------
    # Server failover
    # ------------------------------------------------------------------

    def _post_with_fallback(self, endpoint: str, payload: dict) -> requests.Response:
        """POST to the given endpoint path, cycling through all servers on failure.

        Args:
            endpoint: API path, e.g. "/v1/chat/completions"
            payload: JSON-serialisable request body

        Returns:
            Successful requests.Response

        Raises:
            requests.exceptions.RequestException: if every server fails
        """
        last_exc: Optional[Exception] = None
        for server_url in self.servers:
            url = f"{server_url}{endpoint}"
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Server {server_url} failed ({type(e).__name__}): {e}. Trying next server.")
                last_exc = e

        raise last_exc  # all servers exhausted

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        document_text: str,
        usage: str = "classification",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt for the requested usage type.

        Args:
            document_text: Article text
            usage: "classification" | "summary" | "sentiment" | "verification" | "subtopic"
            context: Extra data needed for some usages.
                     For "subtopic": {"category": str, "subgroups": List[str]}

        Returns:
            Formatted prompt string
        """
        categories_str = ", ".join(self.config.categories)

        if usage == "classification":
            if self.config.topic_modeling_mode == "utilize":
                # Strict assignment — no new categories allowed
                prompt = f"""### Task
Classify the following news article into exactly ONE of the predefined major categories.

### Rules
1. Choose ONLY from the list below — do NOT create new categories.
2. Return ONLY the category name, exactly as written in the list.
3. No explanations, no quotes, no preamble.

### Categories
{chr(10).join(f'- {c}' for c in self.config.categories)}

### Article
{document_text}

### Category:"""
            else:
                # explore mode: prefer existing seeds, expand when nothing fits
                prompt = f"""### Task
Classify the following article about Swiss Migration into exactly ONE specific topic.

### Rules
1. **Prefer Existing Topics**: Choose from the [LIST] below if applicable.
2. **Create New if Necessary**: If none fit, create a new, specific topic (max 3 words).
3. **Be Specific**: Avoid broad terms like "Politics," "Migration," or "Switzerland."
4. **Output**: Return ONLY the topic name. No preamble, no quotes, no "The topic is:".

### [LIST] of Existing Topics: {categories_str}

### Article Text
{document_text}

### Classification:"""

            return prompt

        if usage == "subtopic":
            # Only used in utilize mode; context must contain category and subgroups
            ctx = context or {}
            category = ctx.get("category", "")
            subgroups = ctx.get("subgroups", [])
            subgroups_str = "\n".join(f"- {sg}" for sg in subgroups)
            prompt = f"""### Task
The article below has been classified under the major category "{category}".
Now assign it to the single most appropriate subgroup from the list below.

### Rules
1. Choose ONLY from the subgroup list.
2. Return ONLY the subgroup name, exactly as written.
3. No explanations or extra text.

### Subgroups
{subgroups_str}

### Article
{document_text}

### Subgroup:"""
            return prompt

        if usage == "summary":
            prompt = f"""Summarize the following document in one sentence:

- Give a concise summary of the main topic and key points.
- Do NOT include any classification or category information, just a neutral summary.
- Focus on the content of the document, not on metadata or publication details.
- Summarize in english, regardless of the document language.

Document:
{document_text}

Provide a concise summary in english."""
            return prompt

        if usage == "sentiment":
            prompt = f"""Analyze the overall sentiment of the following news article.

- Return exactly one label: positive, negative, or neutral.
- Base the sentiment on the article's overall tone toward its main topic.
- Do not provide explanation or additional text.

Document:
{document_text}

Sentiment:"""
            return prompt

        # verification (default fallback)
        existing_topics_str = ", ".join(self.other_topics.keys())
        prompt = f"""
### ROLE
You are a Swiss News Classifier. Your goal is to separate articles about **Swiss Immigration/Migration** from all other news.

### THE "YES" RULE (Swiss Immigration & Migration)
You MUST respond with exactly "yes" if the article covers any of the following in the context of Switzerland:
1. **Political Debates on Migration:** Statements by Swiss parties (SVP, Centre/Pfister, FDP, etc.) regarding population growth, the "10-million Switzerland" initiative, or limiting movement.
2. **Asylum & Refugees:** Status S, Ukrainian refugees, Dublin transfers, or solidarity mechanisms with the EU.
3. **Foreign Labor:** Cross-border workers (frontaliers), work permits, or labor shortages filled by foreigners.
4. **Integration:** Language laws, Swiss citizenship/naturalization, or social integration.

### THE "NO" RULE (Non-Migration News)
Respond "no, [Label]" ONLY if the article is NOT about migration.
**CRITICAL:** Do not use the word "Politics" as a label. It is too broad. Be specific (e.g., "Transportation", "Taxation", "Swiss Sports", "International Conflict").

### SELECTION PRIORITY FOR "NO" LABELS:
1. Look at this list: [{existing_topics_str}]
2. If the topic fits one of those (and isn't "Politics"), use it.
3. Otherwise, create a new 1-3 word English label.

### OUTPUT FORMAT:
- If Immigration/Migration: `yes`
- If NOT Immigration/Migration: `no, [Specific Label]`

---
DOCUMENT FOR ANALYSIS:
{document_text}

CLASSIFICATION (English):
"""
        return prompt

    # ------------------------------------------------------------------
    # Core LLM calls
    # ------------------------------------------------------------------

    def _call_chat(self, prompt: str) -> Optional[str]:
        """Send a single chat-completions request with server failover.

        Returns the stripped response text, or None on total failure.
        """
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        try:
            response = self._post_with_fallback("/v1/chat/completions", payload)
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"All servers failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None

    def classify_with_chat(self, document_text: str, mode: str) -> Optional[Dict[str, Any]]:
        """Run the full classification pipeline for a single document.

        Args:
            document_text: Extracted article text
            mode: Processing mode ("all", "classify", "summarize", "verify", "sentiment")

        Returns:
            Dict with keys: verification, category, raw_response, sentiment, summary, subtopic (utilize mode)
        """
        result: Dict[str, Any] = {
            "model": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        verification_is_yes: Optional[bool] = None

        # ---- verification step ----
        if mode in ("all", "verify"):
            verification_text = self._call_chat(self.build_prompt(document_text, usage="verification"))
            if verification_text is None:
                return None

            result["verification"] = verification_text
            cleaned = re.sub(r"\*\*", "", verification_text.strip()).strip()
            lowered = cleaned.lower()
            verification_is_yes = lowered == "yes"

            if verification_is_yes:
                self.correct_topics += 1
            elif lowered.startswith("no"):
                topic_match = re.search(
                    r"^\s*no\s*,?\s*\[?([^\]\n]+)\]?", cleaned, flags=re.IGNORECASE
                )
                topic = topic_match.group(1).strip() if topic_match else "unknown"
                self.other_topics[topic] = self.other_topics.get(topic, 0) + 1

        # ---- classification step ----
        should_classify = mode == "classify" or (mode == "all" and verification_is_yes is True)

        if should_classify:
            classification_text = self._call_chat(self.build_prompt(document_text, usage="classification"))
            if classification_text is None:
                return None

            result["category"] = classification_text
            result["raw_response"] = classification_text

            if self.config.topic_modeling_mode == "explore":
                # In explore mode, grow the category list with new topics
                if classification_text not in self.config.categories:
                    self.config.categories.append(classification_text)
            # In utilize mode, the list is fixed — no update needed

            # ---- subtopic step (utilize mode only) ----
            if self.config.topic_modeling_mode == "utilize":
                subgroups = PREDEFINED_TAXONOMY.get(classification_text)
                if subgroups:
                    subtopic_text = self._call_chat(
                        self.build_prompt(
                            document_text,
                            usage="subtopic",
                            context={"category": classification_text, "subgroups": subgroups},
                        )
                    )
                    result["subtopic"] = subtopic_text if subtopic_text is not None else "unknown"
                else:
                    result["subtopic"] = "unknown"

        elif mode == "all":
            result["category"] = "NOT TOPIC"
            result["raw_response"] = "NOT TOPIC"

        # ---- sentiment step ----
        should_sentiment = mode == "sentiment" or (mode == "all" and verification_is_yes is True)

        if should_sentiment:
            sentiment_text = self._call_chat(self.build_prompt(document_text, usage="sentiment"))
            if sentiment_text is None:
                return None
            result["sentiment"] = sentiment_text
        elif mode == "all":
            result["sentiment"] = "NOT TOPIC"

        # ---- summarization step ----
        if mode in ("all", "summarize"):
            summary_text = self._call_chat(self.build_prompt(document_text, usage="summary"))
            if summary_text is None:
                return None
            result["summary"] = summary_text

        return result

    def classify_with_completions(self, document_text: str) -> Optional[Dict[str, Any]]:
        """Classify using the legacy completions endpoint (alternative to chat completions)."""
        prompt = self.build_prompt(document_text)
        full_prompt = f"{self.config.system_prompt}\n\n{prompt}"

        payload = {
            "model": self.config.model_name,
            "prompt": full_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        try:
            response = self._post_with_fallback("/v1/completions", payload)
            data = response.json()
            predicted_category = data["choices"][0]["text"].strip()

            if predicted_category not in self.config.categories:
                logger.warning(f"Model returned invalid category: {predicted_category}")
                predicted_category = self._match_category(predicted_category)

            return {
                "category": predicted_category,
                "raw_response": data["choices"][0]["text"],
                "model": self.config.model_name,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"All servers failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None

    def _match_category(self, predicted: str) -> str:
        """Fuzzy-match predicted string to the nearest configured category."""
        predicted_lower = predicted.lower()
        for category in self.config.categories:
            if category.lower() in predicted_lower or predicted_lower in category.lower():
                return category
        return "unknown"

    # ------------------------------------------------------------------
    # Document-level entry points
    # ------------------------------------------------------------------

    def classify_document(
        self,
        document: Dict[str, Any],
        text_fields: List[str] = None,
        mode: str = "all",
    ) -> Dict[str, Any]:
        """Classify a single document (CSV row).

        Args:
            document: Row dictionary from DocumentLoader
            text_fields: Fields to concatenate for classification text
            mode: Processing mode ("all", "classify", "summarize", "verify", "sentiment")

        Returns:
            Result dict with document metadata and classification outputs
        """
        loader = DocumentLoader(self.config.input_csv_file)
        document_text = loader.extract_text_content(document, text_fields)

        classification = self.classify_with_chat(document_text, mode=mode)

        result = {
            "id": document.get("id"),
            "source_file": document.get("source_file"),
            "row_index": document.get("row_index"),
            "pubtime": document.get("pubtime"),
            "medium_code": document.get("medium_code"),
            "language": document.get("language"),
            "head": document.get("head"),
        }

        if classification is None:
            # Document failed even after server fallover — mark as failed
            result["error"] = "all_servers_failed"
            return result

        if mode == "all":
            result["classification"] = classification.get("category")
            result["raw_response"] = classification.get("raw_response")
            result["verification"] = classification.get("verification")
            result["sentiment"] = classification.get("sentiment")
            result["summary"] = classification.get("summary")
            if self.config.topic_modeling_mode == "utilize":
                result["subtopic"] = classification.get("subtopic")
        elif mode == "classify":
            result["classification"] = classification.get("category")
            result["raw_response"] = classification.get("raw_response")
            if self.config.topic_modeling_mode == "utilize":
                result["subtopic"] = classification.get("subtopic")
        elif mode == "summarize":
            result["summary"] = classification.get("summary")
        elif mode == "verify":
            result["verification"] = classification.get("verification")
        elif mode == "sentiment":
            result["sentiment"] = classification.get("sentiment")

        return result

    def classify_batch(
        self,
        documents: List[Dict[str, Any]],
        text_fields: List[str] = None,
        mode: str = "all",
    ) -> List[Dict[str, Any]]:
        """Classify a batch of documents."""
        results = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"row_{i}")
            source = doc.get("source_file", "unknown")
            logger.info(f"Classifying document {i+1}/{len(documents)}: {source} - ID: {doc_id}")
            result = self.classify_document(doc, text_fields, mode=mode)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Persistence & reporting
    # ------------------------------------------------------------------

    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
        """Save classification results to JSON file."""
        if output_path is None:
            output_folder = Path(self.config.output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_folder / f"classification_results_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from classification results."""
        total = len(results)
        category_counts: Dict[str, int] = {}
        subtopic_counts: Dict[str, int] = {}
        sentiment_counts: Dict[str, int] = {}
        failed = 0
        successful = 0

        for result in results:
            if result.get("error"):
                failed += 1
                continue

            classification = result.get("classification")
            sentiment = result.get("sentiment")
            subtopic = result.get("subtopic")

            if classification is not None:
                successful += 1
                category_counts[classification] = category_counts.get(classification, 0) + 1
            if sentiment is not None:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            if subtopic is not None:
                subtopic_counts[subtopic] = subtopic_counts.get(subtopic, 0) + 1

        summary: Dict[str, Any] = {
            "total_documents": total,
            "successfully_classified": successful,
            "failed": failed,
            "category_distribution": category_counts,
            "success_rate": successful / total if total > 0 else 0,
            "verified_articles": self.correct_topics,
            "verification_rate": self.correct_topics / successful if successful > 0 else 0,
            "other_topics": self.other_topics,
            "sentiment_distribution": sentiment_counts,
        }
        if self.config.topic_modeling_mode == "utilize":
            summary["subtopic_distribution"] = subtopic_counts

        return summary
