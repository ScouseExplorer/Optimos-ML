from typing import List, Dict

class TextProcessor:
    def __init__(self):
        # Setup cleaning rules, regex patterns, etc.
        pass
    
    def clean_text(self, text: str) -> str:
        # Remove noise, fix formatting
        return text
    
    def normalize(self, text: str) -> str:
        # Lowercase, expand contractions
        return text.lower()
    
    def extract_hashtags(self, text: str) -> List[str]:
        # Get all hashtags
        return [part for part in text.split() if part.startswith('#')]
    
    def extract_mentions(self, text: str) -> List[str]:
        # Get all @mentions
        return [part for part in text.split() if part.startswith('@')]
    
    def process_emojis(self, text: str) -> Dict:
        # Count/extract emojis
        return {}
    
    def preprocess_for_ml(self, text: str) -> str:
        # Full pipeline for ML input
        cleaned = self.clean_text(text)
        return self.normalize(cleaned)
        