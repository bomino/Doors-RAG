"""API endpoints for the RAG system"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..rag.retriever import HybridRetriever
from ..rag.chain import RAGChain

logger = logging.getLogger(__name__)

# Global variables for components (initialized on startup)
rag_chain = None

def initialize_rag_components():
    """Initialize RAG components"""
    global rag_chain

    try:
        # Initialize Qdrant client
        qdrant_host = os.getenv("QDRANT_URL", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize components
        retriever = HybridRetriever(qdrant_client, embedding_model)
        rag_chain = RAGChain(retriever)

        logger.info("RAG components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        rag_chain = None

# Initialize components on module load
initialize_rag_components()

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    include_cross_references: bool = True
    max_results: int = 10

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    confidence_score: float
    sources: List[str]
    conflicts: Optional[List[str]] = None

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the RAG pipeline"""
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please ensure Qdrant is running and embeddings are created."
        )

    try:
        # Process query through RAG pipeline
        result = await rag_chain.process(
            query=request.query,
            filters=request.filters,
            max_results=request.max_results,
            include_cross_references=request.include_cross_references
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if rag_chain else "unhealthy"
    return {"status": status, "rag_initialized": rag_chain is not None}
