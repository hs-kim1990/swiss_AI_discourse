"""
Document loader module for reading CSV documents
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and preprocess CSV documents"""
    
    def __init__(self, csv_file_path: str, required_columns: List[str] = None):
        """
        Initialize document loader
        
        Args:
            csv_file_path: Path to CSV file
            required_columns: List of required columns to load from CSV
        """
        self.csv_file_path = Path(csv_file_path)
        self.required_columns = required_columns or ["id", "pubtime", "medium_code", 
                                                      "language", "head", "content"]
        
        if not self.csv_file_path.exists():
            raise ValueError(f"CSV file not found: {csv_file_path}")
        
        if not self.csv_file_path.is_file():
            raise ValueError(f"Path is not a file: {csv_file_path}")
        
        if self.csv_file_path.suffix.lower() != '.csv':
            raise ValueError(f"File is not a CSV: {csv_file_path}")
    
    def load_document(self) -> Optional[pd.DataFrame]:
        """
        Load the CSV document
        
        Returns:
            DataFrame with required columns or None if loading fails
        """
        try:
            df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {self.csv_file_path}: {missing_cols}")
                # Use only available columns
                available_cols = [col for col in self.required_columns if col in df.columns]
                if not available_cols:
                    logger.error(f"No required columns found in {self.csv_file_path}")
                    return None
                df = df[available_cols]
            else:
                df = df[self.required_columns]
            
            logger.info(f"Loaded {len(df)} rows from {self.csv_file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load document {self.csv_file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all rows from the CSV file and convert to list of row dictionaries
        
        Returns:
            List of row dictionaries from CSV file
        """
        df = self.load_document()
        all_rows = []
        
        if df is not None:
            # Convert each row to a dictionary
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['source_file'] = self.csv_file_path.name
                row_dict['row_index'] = idx
                all_rows.append(row_dict)
        
        logger.info(f"Successfully loaded {len(all_rows)} rows from {self.csv_file_path.name}")
        return all_rows
        
    def load_documents_batch(self, batch_size: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Load rows from CSV in batches for memory efficiency
        
        Args:
            batch_size: Number of rows per batch
            
        Yields:
            Batches of row dictionaries
        """
        df = self.load_document()
        if df is None:
            return
        
        batch = []
        # Convert each row to a dictionary
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['source_file'] = self.csv_file_path.name
            row_dict['row_index'] = idx
            batch.append(row_dict)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining rows
        if batch:
            yield batch
    
    def extract_text_content(self, document: Dict[str, Any], text_fields: List[str] = None) -> str:
        """
        Extract text content from row dictionary for classification
        
        Args:
            document: Row dictionary from CSV
            text_fields: List of fields to extract (if None, uses 'head' and 'content')
            
        Returns:
            Extracted text content
        """
        if text_fields is None:
            text_fields = ['head', 'content']
        
        texts = []
        for field in text_fields:
            if field in document and pd.notna(document[field]):
                texts.append(str(document[field]))
        
        return " ".join(texts)
