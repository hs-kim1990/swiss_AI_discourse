"""
Configuration module for LLM classifier
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class Config:
    """Configuration for LLM classification framework"""
    
    # vLLM Server settings
    vllm_base_url: str = "http://localhost:8000"
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    api_key: Optional[str] = None
    
    # Classification settings
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
    
    # Document processing
    input_csv_file: str = "./documents/data.csv"
    output_folder: str = "./results"
    batch_size: int = 10
    
    # Classification categories
    categories: List[str] = None
    
    # Prompt template
    system_prompt: str = "You are a precise document classifier. Classify documents into the given categories based on their content."
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
    
    @classmethod
    def from_json(cls, config_path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            "vllm_base_url": self.vllm_base_url,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "input_csv_file": self.input_csv_file,
            "output_folder": self.output_folder,
            "batch_size": self.batch_size,
            "categories": self.categories,
            "system_prompt": self.system_prompt
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.categories:
            raise ValueError("Categories list cannot be empty")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return True
