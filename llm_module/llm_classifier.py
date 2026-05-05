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

_MAX_RETRIES = 3        # max re-attempts before falling back
_FALLBACK_LABEL = "Others"  # assigned when all retries are exhausted
_EXPLORE_MAX_WORDS = 7  # explore mode: topics longer than this trigger a retry
_LANGUAGE_NAMES = {"de": "German", "fr": "French", "it": "Italian"}


def _strip_number_prefix(text: str) -> str:
    """Strip leading numbered-list labels such as '5. ' or '3) ' from LLM responses."""
    return re.sub(r"^\d+[\.)\s]\s*", "", text).strip()


def _strip_llm_preamble(text: str) -> str:
    """Strip boilerplate opener lines echoed from the prompt."""
    return re.sub(
        r"^(?:here is|based on|the following)[^:\n]*:\s*\n\s*",
        "", text, flags=re.IGNORECASE
    ).strip()


class LLMClassifier:
    """Classify documents using LLM via vLLM server"""

    def __init__(self, config: Config):
        self.config = config
        self.config.validate()

        self.completions_url = f"{config.vllm_base_url}/v1/completions"
        self.chat_completions_url = f"{config.vllm_base_url}/v1/chat/completions"

        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.correct_topics = 0
        self.other_topics = {}

    def test_connection(self) -> bool:
        """Test connection to vLLM server"""
        try:
            models_url = f"{self.config.vllm_base_url}/v1/models"
            response = requests.get(models_url, headers=self.headers, timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to vLLM server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            return False

    def build_prompt(self, document_text: str, usage: str = "classification") -> str:
        """
        Build a prompt for the requested usage.

        For 'classification':
          - explore mode: seed-and-expand (original behaviour)
          - utilize mode: pick exactly one of the predefined categories
        """
        categories_str = ", ".join(self.config.categories)

        if usage == "classification":
            if self.config.topic_mode == "utilize":
                numbered = "\n".join(
                    f"{i + 1}. {c}" for i, c in enumerate(self.config.categories)
                )
                lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
                prompt = f"""### Task
Classify the following {lang_name} Swiss news article into exactly ONE of the predefined categories below.

### Categories
{numbered}

### Rules
1. You MUST choose exactly one category from the list above.
2. Do NOT invent or create new categories.
3. Return ONLY the exact category name as written above. No preamble, no quotes, no explanation.

### Article Text
{document_text}

### Category:"""
            else:
                # explore mode — seed-and-expand (original)
                lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
                prompt = f"""### Task
Classify the following {lang_name}-language article about Swiss Migration into exactly ONE specific topic.

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

        if usage == "summary":
            lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
            return f"""You are a news analyst for a migration policy research project called the 10M Initiative.
Read the following {lang_name} news article and write a concise English summary in 3-5 sentences.
Focus on: what happened, who is involved, and why it matters for migration policy.
Start directly with the summary. Do NOT begin with phrases like "Here is a summary", "This article", or any other preamble.

Article:
{document_text}

Summary:"""

        if usage == "sentiment":
            return f"""Analyze the overall sentiment of the following news article.

- Return exactly one label: positive, negative, or neutral.
- Base the sentiment on the article's overall tone toward its main topic.
- Do not provide explanation or additional text.

Document:
{document_text}

Sentiment:"""

        if usage == "verification":
            existing_topics_str = ", ".join(self.other_topics.keys())
            lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
            return f"""
### ROLE
You are a research assistant for the Swiss "10 Million Initiative" media analysis project,
analyzing {lang_name}-language Swiss news articles.
Decide whether this article is part of the Swiss public debate on immigration, population growth,
or the societal impacts of a large and growing foreign-born population in Switzerland.

### SAY "yes" — if the article touches ANY of the following in the Swiss context:
1. **10M Initiative & population cap**: The "Zuwanderungsinitiative", "10-million Switzerland" debate, party proposals to limit immigration or population size, referendums on migration limits.
2. **Immigration & asylum policy**: Federal/cantonal immigration laws, asylum procedures, residency permits, deportations, border control, Dublin/Schengen agreements.
3. **Swiss-EU free movement**: Free movement of persons, bilateral agreements with the EU, effects of EU enlargement on Swiss population or labor.
4. **Labor market & economic impacts**: Foreign workers, frontaliers/cross-border commuters, wage competition, labor shortages filled by immigrants, effects on pensions or social insurance.
5. **Housing, schools & public services**: Housing shortages, rising rents, school or hospital overcrowding — when attributed to or linked with population growth or immigration.
6. **Social cohesion, integration & identity**: Language barriers, naturalization, cultural enrichment or tensions, discrimination, crime statistics by nationality, Swiss national identity debates.
7. **Foreign population data & demographics**: Statistics on foreign-born share, nationality breakdowns, naturalization rates, regional settlement patterns, population projections.

### SAY "no, [Label]" — if the article is NOT about immigration or foreigners in Switzerland:
**CRITICAL:** Do NOT use "Politics" as a label — it is too broad. Be specific (e.g., "Swiss Taxation", "Transport Policy", "Sports", "Environment", "International Conflict").

### LABEL SELECTION PRIORITY:
1. Reuse a label from this list if it fits: [{existing_topics_str}]
2. If none fit, create a new 1-3 word English label.

### OUTPUT FORMAT:
- Relevant to immigration/population debate: `yes`
- Not relevant: `no, [Specific Label]`

---
DOCUMENT FOR ANALYSIS:
{document_text}

CLASSIFICATION (English):
"""


        return ""

    def build_supporting_prompt(
        self, document_text: str, category: str = "", subtopic: str = ""
    ) -> str:
        """Build prompt to extract ideas that support migration/integration."""
        lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
        topic_ctx = ""
        if category and category not in ("NOT TOPIC", _FALLBACK_LABEL, ""):
            topic_ctx = f"\nTopic: {category}"
            if subtopic and subtopic not in ("NOT TOPIC", _FALLBACK_LABEL, "unknown", ""):
                topic_ctx += f"\nSubtopic: {subtopic}"
        return f"""### ROLE
        You are a news analyst for the 10M Initiative migration research project.
        Read the following {lang_name} news article and identify ideas or evidence that SUPPORT {topic_ctx}.
 
        Summarize the key supporting arguments found in this article regarding {topic_ctx}. 
        Summarize in English in 2-3 sentences. Start directly with the content. Do NOT begin with preamble phrases.
        
        If no supporting ideas are present, respond simply as "No supporting arguments found."

    Article:
    {document_text}

    Supporting Arguments:"""

    def build_denying_prompt(
        self, document_text: str, category: str = "", subtopic: str = ""
    ) -> str:
        """Build prompt to extract ideas that oppose or criticize migration/integration."""
        lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
        topic_ctx = ""
        if category and category not in ("NOT TOPIC", _FALLBACK_LABEL, ""):
            topic_ctx = f"\nTopic: {category}"
            if subtopic and subtopic not in ("NOT TOPIC", _FALLBACK_LABEL, "unknown", ""):
                topic_ctx += f"\nSubtopic: {subtopic}"
        return f"""You are a news analyst for the 10M Initiative migration research project.
Read the following {lang_name} news article and identify ideas or evidence that OPPOSE or CRITICIZE {topic_ctx}.

        Summarize the key opposing arguments found in this article regarding {topic_ctx}.
        Summarize in English in 2-3 sentences. Start directly with the content. Do NOT begin with preamble phrases.
        
        If no opposing ideas are present, respond simply as "No opposing arguments found."

Article:
{document_text}

Opposing Arguments:"""

    def build_subtopic_prompt(
        self, document_text: str, category: str, subgroups: List[str]
    ) -> str:
        """Build prompt for subtopic assignment within a predefined category (utilize mode)."""
        subgroups_str = "\n".join(f"- {sg}" for sg in subgroups)
        lang_name = _LANGUAGE_NAMES.get(self.config.document_language, "German")
        return f"""### Task
The following {lang_name} Swiss news article has been classified under the category "{category}".
Now assign it to exactly ONE subgroup within that category.

### Available Subgroups
{subgroups_str}

### Rules
1. Choose the BEST matching subgroup from the list above.
2. Return ONLY the subgroup name exactly as written. No explanation.

### Article Text
{document_text}

### Subgroup:"""

    def _classify_subtopic(self, document_text: str, category: str) -> Optional[str]:
        """
        Run subtopic classification for a document already assigned to `category`.
        Only used in utilize mode.

        Subgroup candidates come from (in priority order):
          1. subtopic_map  -- flat {category: [subgroup, ...]} from category_subtopic_map.json
          2. subtopic_taxonomy -- nested full taxonomy JSON

        Retries up to _MAX_RETRIES times when the response does not match any known subgroup.
        Returns _FALLBACK_LABEL ("Others") after all retries are exhausted.
        """
        subgroups: List[str] = []

        if self.config.subtopic_map:
            subgroups = self.config.subtopic_map.get(category, [])
        elif self.config.subtopic_taxonomy:
            taxonomy = self.config.subtopic_taxonomy.get("taxonomy", {})
            subgroups = list(taxonomy.get(category, {}).get("subgroups", {}).keys())

        if not subgroups:
            logger.warning(f"No subgroups found for category '{category}' -- skipping subtopic step.")
            return None

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": self.build_subtopic_prompt(document_text, category, subgroups)},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        def _call() -> Optional[str]:
            try:
                resp = requests.post(
                    self.chat_completions_url, headers=self.headers, json=payload, timeout=60
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                return _strip_number_prefix(raw)
            except Exception as exc:
                logger.error(f"Subtopic API call failed: {exc}")
                return None

        def _normalize(raw: str) -> str:
            """Exact match first, then substring fuzzy match."""
            if raw in subgroups:
                return raw
            for sg in subgroups:
                if sg.lower() in raw.lower() or raw.lower() in sg.lower():
                    return sg
            return raw  # unchanged -- caller checks membership

        raw = _normalize(_call() or "")

        for attempt in range(1, _MAX_RETRIES + 1):
            if raw in subgroups:
                break
            logger.warning(
                f"Subtopic '{raw}' not in known subgroups for '{category}' "
                f"(attempt {attempt}/{_MAX_RETRIES}), retrying..."
            )
            raw = _normalize(_call() or "")

        if raw not in subgroups:
            logger.warning(
                f"Subtopic assignment failed after {_MAX_RETRIES} retries for '{category}'. "
                f"Assigning '{_FALLBACK_LABEL}'."
            )
            raw = _FALLBACK_LABEL

        return raw

    def classify_with_chat(self, document_text: str, mode: str) -> Optional[Dict[str, Any]]:
        """
        Run the full classification pipeline for a document.

        Modes: "all", "classify", "summarize", "verify", "sentiment"

        topic_mode="explore": seed-and-expand categories (original behaviour).
        topic_mode="utilize": fixed predefined categories; adds subtopic step after classification.
        """
        result = {
            "model": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        verification_is_yes = None

        # ── Verification step ──────────────────────────────────────────────────
        if mode in ("all", "verify"):
            verification_prompt = self.build_prompt(document_text, usage="verification")
            verification_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": verification_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            try:
                resp = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=verification_payload,
                    timeout=60,
                )
                resp.raise_for_status()
                verification_text = resp.json()["choices"][0]["message"]["content"].strip()
                result["verification"] = verification_text

                cleaned = re.sub(r"\*\*", "", verification_text).strip()
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

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (verification): {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response (verification): {e}")
                return None

        # ── Classification step ────────────────────────────────────────────────
        should_classify = mode == "classify" or (mode == "all" and verification_is_yes is True)

        if should_classify:
            classification_prompt = self.build_prompt(document_text, usage="classification")
            classification_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": classification_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            def _call_classification() -> Optional[str]:
                try:
                    resp = requests.post(
                        self.chat_completions_url,
                        headers=self.headers,
                        json=classification_payload,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    return _strip_number_prefix(raw)
                except requests.exceptions.RequestException as exc:
                    logger.error(f"API request failed (classification): {exc}")
                    return None
                except (KeyError, json.JSONDecodeError) as exc:
                    logger.error(f"Failed to parse API response (classification): {exc}")
                    return None

            def _is_valid_category(text: Optional[str]) -> bool:
                if not text:
                    return False
                if self.config.topic_mode == "utilize":
                    return text in self.config.categories
                # explore: reject if response is too verbose (> _EXPLORE_MAX_WORDS words)
                return len(text.split()) <= _EXPLORE_MAX_WORDS

            predicted_category = _call_classification()
            for attempt in range(1, _MAX_RETRIES + 1):
                if _is_valid_category(predicted_category):
                    break
                logger.warning(
                    f"Classification result '{predicted_category}' invalid "
                    f"(attempt {attempt}/{_MAX_RETRIES}), retrying..."
                )
                predicted_category = _call_classification()

            if not _is_valid_category(predicted_category):
                logger.warning(
                    f"Classification failed after {_MAX_RETRIES} retries. "
                    f"Assigning '{_FALLBACK_LABEL}'."
                )
                predicted_category = _FALLBACK_LABEL

            result["category"] = predicted_category
            result["raw_response"] = predicted_category

            # explore: grow category list only for valid new topics (not the fallback)
            if self.config.topic_mode == "explore" and predicted_category != _FALLBACK_LABEL:
                if predicted_category not in self.config.categories:
                    self.config.categories.append(predicted_category)

            # ── Subtopic step (utilize mode only) ─────────────────────────────
            if self.config.topic_mode == "utilize":
                if result.get("category") == _FALLBACK_LABEL:
                    result["subtopic"] = _FALLBACK_LABEL  # skip LLM when topic already failed
                else:
                    subtopic = self._classify_subtopic(document_text, result.get("category", ""))
                    result["subtopic"] = subtopic

        elif mode == "all":
            result["category"] = "NOT TOPIC"
            result["raw_response"] = "NOT TOPIC"
            if self.config.topic_mode == "utilize":
                result["subtopic"] = None

        # ── Sentiment step ─────────────────────────────────────────────────────
        should_sentiment = mode == "sentiment" or (mode == "all" and verification_is_yes is True)

        if should_sentiment:
            sentiment_prompt = self.build_prompt(document_text, usage="sentiment")
            sentiment_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": sentiment_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            try:
                resp = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=sentiment_payload,
                    timeout=60,
                )
                resp.raise_for_status()
                result["sentiment"] = resp.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (sentiment): {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response (sentiment): {e}")
                return None
        elif mode == "all":
            result["sentiment"] = "NOT TOPIC"

        # ── Summary step ───────────────────────────────────────────────────────
        if mode in ("all", "summarize"):
            summary_prompt = self.build_prompt(document_text, usage="summary")
            summary_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": summary_prompt},
                    {"role": "assistant", "content": ""},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            try:
                resp = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=summary_payload,
                    timeout=60,
                )
                resp.raise_for_status()
                raw_summary = resp.json()["choices"][0]["message"]["content"].strip()
                result["summary"] = _strip_llm_preamble(raw_summary)
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (summary): {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response (summary): {e}")
                return None

        # ── Supporting stance step ─────────────────────────────────────────────
        should_supporting = mode == "supporting" or (mode == "all" and verification_is_yes is True)

        if should_supporting:
            category = result.get("category", "") or ""
            subtopic = result.get("subtopic", "") or ""
            supporting_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": self.build_supporting_prompt(document_text, category, subtopic)},
                    {"role": "assistant", "content": ""},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            try:
                resp = requests.post(
                    self.chat_completions_url, headers=self.headers, json=supporting_payload, timeout=60
                )
                resp.raise_for_status()
                raw_supporting = resp.json()["choices"][0]["message"]["content"].strip()
                result["supporting"] = _strip_llm_preamble(raw_supporting)
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (supporting): {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response (supporting): {e}")
                return None
        elif mode == "all":
            result["supporting"] = "NOT TOPIC"

        # ── Denying stance step ────────────────────────────────────────────────
        should_denying = mode == "denying" or (mode == "all" and verification_is_yes is True)

        if should_denying:
            category = result.get("category", "") or ""
            subtopic = result.get("subtopic", "") or ""
            denying_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": self.build_denying_prompt(document_text, category, subtopic)},
                    {"role": "assistant", "content": ""},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            try:
                resp = requests.post(
                    self.chat_completions_url, headers=self.headers, json=denying_payload, timeout=60
                )
                resp.raise_for_status()
                raw_denying = resp.json()["choices"][0]["message"]["content"].strip()
                result["denying"] = _strip_llm_preamble(raw_denying)
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (denying): {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response (denying): {e}")
                return None
        elif mode == "all":
            result["denying"] = "NOT TOPIC"

        return result

    def classify_with_completions(self, document_text: str) -> Optional[Dict[str, Any]]:
        """Classify document using completions API (alternative method)"""
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
            response = requests.post(
                self.completions_url, headers=self.headers, json=payload, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            predicted_category = result["choices"][0]["text"].strip()

            if predicted_category not in self.config.categories:
                logger.warning(f"Model returned invalid category: {predicted_category}")
                predicted_category = self._match_category(predicted_category)

            return {
                "category": predicted_category,
                "raw_response": result["choices"][0]["text"],
                "model": self.config.model_name,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None

    def _match_category(self, predicted: str) -> str:
        """Fuzzy-match predicted string to a known category."""
        predicted_lower = predicted.lower()
        for category in self.config.categories:
            if category.lower() in predicted_lower or predicted_lower in category.lower():
                return category
        return "unknown"

    def classify_document(
        self,
        document: Dict[str, Any],
        text_fields: List[str] = None,
        mode: str = "all",
    ) -> Dict[str, Any]:
        """Classify a single document (CSV row)."""
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

        if mode == "all":
            result["classification"] = classification.get("category") if classification else None
            result["raw_response"] = classification.get("raw_response") if classification else None
            result["verification"] = classification.get("verification") if classification else None
            result["sentiment"] = classification.get("sentiment") if classification else None
            result["summary"] = classification.get("summary") if classification else None
            result["supporting"] = classification.get("supporting") if classification else None
            result["denying"] = classification.get("denying") if classification else None
            if self.config.topic_mode == "utilize":
                result["subtopic"] = classification.get("subtopic") if classification else None
        elif mode == "classify":
            result["classification"] = classification.get("category") if classification else None
            result["raw_response"] = classification.get("raw_response") if classification else None
            if self.config.topic_mode == "utilize":
                result["subtopic"] = classification.get("subtopic") if classification else None
        elif mode == "summarize":
            result["summary"] = classification.get("summary") if classification else None
        elif mode == "verify":
            result["verification"] = classification.get("verification") if classification else None
        elif mode == "sentiment":
            result["sentiment"] = classification.get("sentiment") if classification else None
        elif mode == "supporting":
            result["supporting"] = classification.get("supporting") if classification else None
        elif mode == "denying":
            result["denying"] = classification.get("denying") if classification else None

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
        successful = 0

        for result in results:
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

        summary = {
            "total_documents": total,
            "successfully_classified": successful,
            "failed": total - successful,
            "category_distribution": category_counts,
            "success_rate": successful / total if total > 0 else 0,
            "verified_articles": self.correct_topics,
            "verification_rate": self.correct_topics / successful if successful > 0 else 0,
            "other_topics": self.other_topics,
            "sentiment_distribution": sentiment_counts,
        }

        if self.config.topic_mode == "utilize":
            summary["subtopic_distribution"] = subtopic_counts

        return summary
