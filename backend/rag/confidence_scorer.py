"""Confidence scoring module"""
from typing import List, Dict, Tuple

class ConfidenceScorer:
    """Calculate confidence scores for responses"""

    def calculate_confidence(self, docs: List[Dict], scores: List[float]) -> Tuple[str, float]:
        """Calculate overall confidence level"""
        if not docs:
            return ("Low", 0.0)

        avg_score = sum(scores) / len(scores)

        if avg_score > 0.85:
            return ("High", min(avg_score * 100, 100))
        elif avg_score > 0.65:
            return ("Medium", avg_score * 100)
        else:
            return ("Low", avg_score * 100)
