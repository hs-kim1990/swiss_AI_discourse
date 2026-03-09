"""
LLM-based document classifier using vLLM server
"""
import logging
from statistics import mode
import requests
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from .config import Config
from .document_loader import DocumentLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClassifier:
    """Classify documents using LLM via vLLM server"""
    
    def __init__(self, config: Config):
        """
        Initialize classifier
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.config.validate()
        
        # vLLM OpenAI-compatible API endpoints
        self.completions_url = f"{config.vllm_base_url}/v1/completions"
        self.chat_completions_url = f"{config.vllm_base_url}/v1/chat/completions"
        
        self.headers = {
            "Content-Type": "application/json"
        }
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"
    
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
        Build classification prompt
        
        Args:
            document_text: Text content of document
            
        Returns:
            Formatted prompt
        """
        categories_str = ", ".join(self.config.categories)
        
        if usage == "classification":
            prompt = f"""Classify the following document into ONE of these categories: {categories_str}
                - Respond with ONLY the category name, no explanations or additional text.
                - If the document does not fit any category, respond with "unknown".

                Document:
                {document_text}

                Respond with ONLY the category name, nothing else."""
            
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
        
        if usage == "verification":
            prompt = f"""Verify the document is an article referring to the topic "Immigrants in Switzerland" and that the classification is correct.:

                - Based on the content of the document, confirm if the classification is correct.
                - Respond with "yes" if the classification is appropriate, or "no" if it is not.

                Document:
                {document_text}

                Is the classification correct? Respond with ONLY "yes" or "no"."""
            
            return prompt
    
    def classify_with_chat(self, document_text: str, mode: str) -> Optional[Dict[str, Any]]:
        """
        Classify document using chat completions API
        
        Args:
            document_text: Text content to classify
            
        Returns:
            Classification result with category and confidence
        """
        result = {"model": self.config.model_name,
            "timestamp": datetime.now().isoformat()}

        if mode == "all" or mode == "classify":
            classification_prompt = self.build_prompt(document_text, usage = "classification")
            
            classification_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": classification_prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }

            try:
                classification_response = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=classification_payload,
                    timeout=60
                )
                classification_response.raise_for_status()
                result = classification_response.json()
            
                predicted_category = result["choices"][0]["message"]["content"].strip()

                result["category"] = predicted_category
                result["raw_response"] = result["choices"][0]["message"]["content"]
            
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response: {e}")
                return None


        if mode == "all" or mode == "summarize":
            summarization_prompt = self.build_prompt(document_text, usage = "summary")
            summarization_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": summarization_prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }

            try:
                summarization_response = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=summarization_payload,
                    timeout=60
                )
                summarization_response.raise_for_status()
                summary_result = summarization_response.json()
                summary_text = summary_result["choices"][0]["message"]["content"].strip()

                result["summary"] = summary_text
            
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response: {e}")
                return None

        if mode == "all" or mode == "verify":
            verification_prompt = self.build_prompt(document_text, usage = "verification")
            verification_payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": verification_prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }
            try:
                verification_response = requests.post(
                    self.chat_completions_url,
                    headers=self.headers,
                    json=verification_payload,
                    timeout=60
                )
                verification_response.raise_for_status()
                verification_result = verification_response.json()
                verification_text = verification_result["choices"][0]["message"]["content"].strip()

                result["verification"] = verification_text
            
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                return None
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse API response: {e}")
                return None

        return result
    
    def classify_with_completions(self, document_text: str) -> Optional[Dict[str, Any]]:
        """
        Classify document using completions API (alternative method)
        
        Args:
            document_text: Text content to classify
            
        Returns:
            Classification result
        """
        prompt = self.build_prompt(document_text)
        full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.config.model_name,
            "prompt": full_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }
        
        try:
            response = requests.post(
                self.completions_url,
                headers=self.headers,
                json=payload,
                timeout=60
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
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None
    
    def _match_category(self, predicted: str) -> str:
        """
        Try to match predicted category to valid categories
        
        Args:
            predicted: Predicted category string
            
        Returns:
            Matched category or "unknown"
        """
        predicted_lower = predicted.lower()
        for category in self.config.categories:
            if category.lower() in predicted_lower or predicted_lower in category.lower():
                return category
        return "unknown"
    
    # select mode from ["all", "classify", "summarize", "verify"] to determine which classification method to use
    def classify_document(self, document: Dict[str, Any], text_fields: List[str] = None, mode: str = "all") -> Dict[str, Any]:
        """
        Classify a single document (CSV row)
        
        Args:
            document: Row dictionary from DocumentLoader (CSV row)
            text_fields: Fields to extract for classification
            use_chat: Use chat completions API (True) or completions API (False)
            
        Returns:
            Document with classification results
        """
        loader = DocumentLoader(self.config.input_csv_file)
        document_text = loader.extract_text_content(document, text_fields)
        
        classification = self.classify_with_chat(document_text, mode=mode)

        
        # Build result with CSV row metadata
        result = {
            "id": document.get("id"),
            "source_file": document.get("source_file"),
            "row_index": document.get("row_index"),
            "pubtime": document.get("pubtime"),
            "medium_code": document.get("medium_code"),
            "language": document.get("language"),
            "head": document.get("head"),
        }

        if mode == 'all':
            # append to result the classification and summary results
            result["classification"] = classification.get("category") if classification else None
            result["raw_response"] = classification.get("raw_response") if classification else None
            result["summary"] = classification.get("summary") if classification else None
        elif mode == 'classify':
            result["classification"] = classification.get("category") if classification else None
            result["raw_response"] = classification.get("raw_response") if classification else None
        elif mode == 'summarize':
            result["summary"] = classification.get("summary") if classification else None
        elif mode == 'verify':
            result["verification"] = classification.get("verification") if classification else None
        
        return result
    
    def classify_batch(self, documents: List[Dict[str, Any]], text_fields: List[str] = None, use_chat: bool = True) -> List[Dict[str, Any]]:
        """
        Classify a batch of documents (CSV rows)
        
        Args:
            documents: List of row dictionaries from CSV
            text_fields: Fields to extract for classification
            use_chat: Use chat completions API
            
        Returns:
            List of classified documents
        """
        results = []
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', f"row_{i}")
            source = doc.get('source_file', 'unknown')
            logger.info(f"Classifying document {i+1}/{len(documents)}: {source} - ID: {doc_id}")
            result = self.classify_document(doc, text_fields, use_chat)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
        """
        Save classification results to JSON file
        
        Args:
            results: Classification results
            output_path: Output file path (if None, uses config output_folder)
        """
        if output_path is None:
            output_folder = Path(self.config.output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_folder / f"classification_results_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from classification results
        
        Args:
            results: Classification results
            
        Returns:
            Summary dictionary
        """
        total = len(results)
        category_counts = {}
        successful = 0
        correct_article = 0
        
        for result in results:
            if result["classification"] is not None:
                successful += 1
                category = result["classification"]
                category_counts[category] = category_counts.get(category, 0) + 1
            if result["verification"] is not None and result["verification"].lower() == "yes":
                correct_article += 1
        
        summary = {
            "total_documents": total,
            "successfully_classified": successful,
            "failed": total - successful,
            "category_distribution": category_counts,
            "success_rate": successful / total if total > 0 else 0,
            "verified_articles": correct_article,
            "verification_rate": correct_article / successful if successful > 0 else 0
        }
        
        return summary
