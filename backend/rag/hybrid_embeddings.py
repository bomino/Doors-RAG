"""
Hybrid Embeddings System
Uses OpenAI for high-quality embeddings with local fallback
"""

import os
import logging
from typing import List, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class HybridEmbeddings:
    """Hybrid embedding system with OpenAI primary and local fallback"""

    def __init__(self):
        # Try to use OpenAI embeddings first (much better quality)
        self.openai_client = None
        self.use_openai = False

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("your-"):
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.use_openai = True
                self.embedding_dim = 1536  # OpenAI ada-002 dimension
                logger.info("Using OpenAI embeddings for better accuracy")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")

        # Always have local fallback
        if not self.use_openai:
            # Use a better local model if OpenAI isn't available
            self.local_model = SentenceTransformer('all-mpnet-base-v2')  # Better than MiniLM
            self.embedding_dim = 768  # MPNet dimension
            logger.info("Using local MPNet embeddings")
        else:
            # Still load local for fallback
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""

        if not texts:
            return np.array([])

        # Clean texts
        texts = [str(text).strip() for text in texts]

        if self.use_openai:
            try:
                # Use OpenAI's text-embedding-ada-002 (best quality)
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )

                embeddings = [data.embedding for data in response.data]

                if convert_to_numpy:
                    return np.array(embeddings)
                return embeddings

            except Exception as e:
                logger.error(f"OpenAI embedding failed, using fallback: {e}")
                # Fall back to local model
                return self.local_model.encode(texts, convert_to_numpy=convert_to_numpy)
        else:
            # Use local model
            return self.local_model.encode(texts, convert_to_numpy=convert_to_numpy)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim