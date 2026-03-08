"""
Document loader module for reading JSON documents from a folder
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import glob


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and preprocess JSON documents from a folder"""
    
    def __init__(self, folder_path: str, file_pattern: str = "*.json"):
        """
        Initialize document loader
        
        Args:
            folder_path: Path to folder containing JSON documents
            file_pattern: Glob pattern for matching files (default: "*.json")
        """
        self.folder_path = Path(folder_path)
        self.file_pattern = file_pattern
        
        if not self.folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
    
    def get_document_paths(self) -> List[Path]:
        """Get list of all document paths matching the pattern"""
        pattern = str(self.folder_path / self.file_pattern)
        paths = [Path(p) for p in glob.glob(pattern)]
        logger.info(f"Found {len(paths)} documents in {self.folder_path}")
        return paths
    
    def load_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single JSON document
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Document as dictionary or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "content": document
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all documents from the folder
        
        Returns:
            List of documents with metadata
        """
        paths = self.get_document_paths()
        documents = []
        
        for path in paths:
            doc = self.load_document(path)
            if doc is not None:
                documents.append(doc)
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_documents_batch(self, batch_size: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Load documents in batches for memory efficiency
        
        Args:
            batch_size: Number of documents per batch
            
        Yields:
            Batches of documents
        """
        paths = self.get_document_paths()
        batch = []
        
        for path in paths:
            doc = self.load_document(path)
            if doc is not None:
                batch.append(doc)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining documents
        if batch:
            yield batch
    
    def extract_text_content(self, document: Dict[str, Any], text_fields: List[str] = None) -> str:
        """
        Extract text content from document for classification
        
        Args:
            document: Document dictionary
            text_fields: List of fields to extract (if None, converts entire content to string)
            
        Returns:
            Extracted text content
        """
        content = document.get("content", {})
        
        if text_fields:
            texts = []
            for field in text_fields:
                if field in content:
                    texts.append(str(content[field]))
            return " ".join(texts)
        else:
            # Convert entire content to string
            return json.dumps(content, ensure_ascii=False)
