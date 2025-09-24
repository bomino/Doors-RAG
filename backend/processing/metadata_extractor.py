"""Extract metadata from document content"""
import re
from typing import Dict, List, Optional

class MetadataExtractor:
    """Extract structured metadata from text"""

    def extract_door_number(self, text: str) -> Optional[str]:
        """Extract door numbers from text"""
        pattern = r'(?:Door\s+)?(\d{1,4}[A-Z]?(?:[A-Z]|\.\d+)?)'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None

    def extract_fire_rating(self, text: str) -> Optional[str]:
        """Extract fire ratings from text"""
        pattern = r'(\d+)[\s-]?(?:min(?:ute)?|hr|hour)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return f"{matches[0]} minutes" if matches else None
