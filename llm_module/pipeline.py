"""
Complete classification pipeline
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .config import Config
from .document_loader import DocumentLoader
from .llm_classifier import LLMClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """End-to-end document classification pipeline"""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.loader = DocumentLoader(config.input_folder)
        self.classifier = LLMClassifier(config)
    
    def run(self, text_fields: Optional[List[str]] = None, use_chat: bool = True, save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete classification pipeline
        
        Args:
            text_fields: Document fields to use for classification
            use_chat: Use chat completions API
            save_results: Save results to file
            
        Returns:
            Pipeline results with summary
        """
        logger.info("Starting classification pipeline")
        
        # Test connection
        if not self.classifier.test_connection():
            raise ConnectionError("Cannot connect to vLLM server")
        
        # Load documents
        logger.info(f"Loading documents from {self.config.input_folder}")
        all_results = []
        
        # Process in batches
        for batch_num, batch in enumerate(self.loader.load_documents_batch(self.config.batch_size), 1):
            logger.info(f"Processing batch {batch_num} ({len(batch)} documents)")
            batch_results = self.classifier.classify_batch(batch, text_fields, use_chat)
            all_results.extend(batch_results)
        
        # Generate summary
        summary = self.classifier.generate_summary(all_results)
        logger.info(f"Classification complete. Success rate: {summary['success_rate']:.2%}")
        logger.info(f"Category distribution: {summary['category_distribution']}")
        
        # Save results
        if save_results:
            self.classifier.save_results(all_results)
        
        return {
            "summary": summary,
            "results": all_results
        }
    
    def run_single_document(self, document_path: str, text_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classify a single document
        
        Args:
            document_path: Path to document
            text_fields: Fields to extract
            
        Returns:
            Classification result
        """
        doc = self.loader.load_document(Path(document_path))
        if doc is None:
            raise ValueError(f"Failed to load document: {document_path}")
        
        result = self.classifier.classify_document(doc, text_fields)
        return result
