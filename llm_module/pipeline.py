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
        self.loader = DocumentLoader(config.input_csv_file)
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
        logger.info(f"Loading documents from {self.config.input_csv_file}")
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
    
    def run_single_document(self, document_path: str, row_index: int = 0, text_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classify a single row from a CSV document
        
        Args:
            document_path: Path to CSV file
            row_index: Index of the row to classify (default: 0)
            text_fields: Fields to extract
            
        Returns:
            Classification result
        """
        # Create a temporary loader for this specific file
        temp_loader = DocumentLoader(document_path)
        df = temp_loader.load_document()
        if df is None:
            raise ValueError(f"Failed to load document: {document_path}")
        
        if row_index >= len(df):
            raise ValueError(f"Row index {row_index} out of bounds for document with {len(df)} rows")
        
        # Convert row to dictionary
        row = df.iloc[row_index]
        doc = row.to_dict()
        doc['source_file'] = Path(document_path).name
        doc['row_index'] = row_index
        
        result = self.classifier.classify_document(doc, text_fields)
        return result
