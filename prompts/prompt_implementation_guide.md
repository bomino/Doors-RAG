# Implementation Guide for Enhanced Door Specification RAG System

This guide provides technical instructions for integrating the enhanced system prompt with the proposed RAG architecture.

## Quick Start Checklist

- [ ] Set up Python environment (3.11+)
- [ ] Install required packages (see requirements.txt below)
- [ ] Process PDFs with hierarchical chunking
- [ ] Initialize Qdrant vector database
- [ ] Configure LLM with enhanced prompt
- [ ] Implement confidence scoring
- [ ] Set up monitoring and feedback loops

## 1. Environment Setup

### Requirements.txt
```txt
# Core RAG Framework
fastapi==0.104.1
haystack-ai==2.0.0
llama-index==0.10.0
langchain==0.1.0

# Vector Database
qdrant-client==1.7.0

# PDF Processing
pymupdf==1.23.0
pdfplumber==0.10.0
pandas==2.1.0

# LLM Integration
openai==1.0.0
ollama==0.1.0
anthropic==0.15.0

# Embeddings & ML
sentence-transformers==2.3.0
torch==2.1.0
numpy==1.24.0

# Utilities
pydantic==2.0.0
python-dotenv==1.0.0
redis==5.0.0
python-multipart==0.0.6
uvicorn==0.24.0

# Monitoring
prometheus-client==0.19.0
structlog==24.1.0
```

## 2. PDF Processing Pipeline

### A. Extraction with Enhanced Metadata

```python
import pymupdf
import pdfplumber
from typing import Dict, List, Any
import re
from datetime import datetime

class EnhancedPDFProcessor:
    def __init__(self):
        self.metadata_schema = {
            'doc_title': '',
            'division_section': '',
            'page': 0,
            'heading_path': '',
            'door_no': '',
            'room_name': '',
            'hardware_set': '',
            'fire_rating': '',
            'door_type': '',
            'frame_type': '',
            'sheet': '',
            'last_revision': None,
            'submittal_status': '',
            'rfi_references': [],
            'acoustic_rating': '',
            'security_requirements': '',
            'ada_compliance': False,
            'coordination_notes': '',
            'cross_references': []
        }

    def extract_with_metadata(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text with enhanced metadata from PDFs"""
        chunks = []

        # Use PyMuPDF for fast text extraction
        doc = pymupdf.open(pdf_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()

            # Extract metadata from text patterns
            metadata = self.extract_metadata(text, page_num)
            metadata['doc_title'] = self.get_doc_title(pdf_path)
            metadata['page'] = page_num + 1

            # Check for tables with pdfplumber
            if self.has_tables(pdf_path, page_num):
                table_data = self.extract_tables(pdf_path, page_num)
                chunks.extend(self.process_table_data(table_data, metadata))

            # Process text chunks
            text_chunks = self.create_hierarchical_chunks(text, metadata)
            chunks.extend(text_chunks)

        return chunks

    def extract_metadata(self, text: str, page_num: int) -> Dict:
        """Extract specific metadata from text content"""
        metadata = self.metadata_schema.copy()

        # Extract door numbers (e.g., "148A", "Door 627C")
        door_pattern = r'(?:Door\s+)?(\d{1,4}[A-Z]?(?:[A-Z]|\.\d+)?)'
        doors = re.findall(door_pattern, text)
        if doors:
            metadata['door_no'] = doors[0]

        # Extract hardware set references
        hw_pattern = r'(?:Hardware Set|HS|Set)\s*[:#]?\s*([A-Z0-9-]+)'
        hw_sets = re.findall(hw_pattern, text)
        if hw_sets:
            metadata['hardware_set'] = hw_sets[0]

        # Extract fire ratings
        fire_pattern = r'(\d+)[\s-]?(?:min(?:ute)?|hr|hour)'
        fire_ratings = re.findall(fire_pattern, text, re.IGNORECASE)
        if fire_ratings:
            metadata['fire_rating'] = f"{fire_ratings[0]} minutes"

        # Extract STC ratings
        stc_pattern = r'STC[\s-]?(\d+)'
        stc_ratings = re.findall(stc_pattern, text)
        if stc_ratings:
            metadata['acoustic_rating'] = f"STC {stc_ratings[0]}"

        # Extract cross-references
        xref_pattern = r'(?:See|Refer to)\s+(?:Section\s+)?(\d{6}|\d{2}\s\d{2}\s\d{2})'
        xrefs = re.findall(xref_pattern, text)
        metadata['cross_references'] = list(set(xrefs))

        return metadata

    def create_hierarchical_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Create parent and child chunks with metadata"""
        chunks = []

        # Split into sections (parent chunks)
        sections = self.split_into_sections(text)

        for section in sections:
            # Create parent chunk (500-1000 tokens)
            parent_chunk = {
                'content': section['content'][:1000],  # Simplified token counting
                'metadata': {**metadata, 'chunk_type': 'parent'},
                'chunk_id': self.generate_chunk_id(),
                'children': []
            }

            # Create child chunks (200-500 tokens)
            paragraphs = section['content'].split('\n\n')
            for para in paragraphs:
                if len(para) > 50:  # Minimum content threshold
                    child_chunk = {
                        'content': para[:500],
                        'metadata': {**metadata, 'chunk_type': 'child'},
                        'chunk_id': self.generate_chunk_id(),
                        'parent_id': parent_chunk['chunk_id']
                    }
                    parent_chunk['children'].append(child_chunk['chunk_id'])
                    chunks.append(child_chunk)

            chunks.append(parent_chunk)

        return chunks
```

## 3. Vector Database Configuration

### A. Qdrant Setup with Enhanced Metadata

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType
)
import numpy as np
from typing import List, Dict, Any

class QdrantManager:
    def __init__(self, collection_name="door_specs"):
        self.client = QdrantClient(path="./qdrant_db")
        self.collection_name = collection_name
        self.setup_collection()

    def setup_collection(self):
        """Initialize collection with proper schema"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,  # For sentence-transformers
                distance=Distance.COSINE
            ),
            payload_schema={
                "doc_title": PayloadSchemaType.KEYWORD,
                "division_section": PayloadSchemaType.KEYWORD,
                "door_no": PayloadSchemaType.KEYWORD,
                "hardware_set": PayloadSchemaType.KEYWORD,
                "fire_rating": PayloadSchemaType.KEYWORD,
                "chunk_type": PayloadSchemaType.KEYWORD,
                "confidence_score": PayloadSchemaType.FLOAT,
                "cross_references": PayloadSchemaType.KEYWORD,
            }
        )

    def upsert_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """Insert chunks with metadata into Qdrant"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "content": chunk['content'],
                    **chunk['metadata']
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def hybrid_search(self, query_embedding: np.ndarray,
                     query_text: str,
                     filters: Dict = None,
                     limit: int = 10) -> List[Dict]:
        """Perform hybrid search with metadata filtering"""

        # Build filter conditions
        must_conditions = []
        if filters:
            for key, value in filters.items():
                if value:
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

        # Vector search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True
        )

        # Convert results with confidence scores
        results = []
        for hit in search_result:
            results.append({
                'content': hit.payload.get('content'),
                'metadata': hit.payload,
                'score': hit.score,
                'confidence': self.calculate_confidence(hit.score, hit.payload)
            })

        return results

    def calculate_confidence(self, score: float, payload: Dict) -> str:
        """Calculate confidence level based on score and metadata"""
        if score > 0.9 and payload.get('chunk_type') == 'parent':
            return "High"
        elif score > 0.7:
            return "Medium"
        else:
            return "Low"
```

## 4. RAG Pipeline with Confidence Scoring

### A. Enhanced RAG Chain Implementation

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict, Tuple
import asyncio

class EnhancedRAGPipeline:
    def __init__(self, vector_store: QdrantManager, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client
        self.system_prompt = self.load_enhanced_prompt()

    def load_enhanced_prompt(self) -> str:
        """Load the enhanced system prompt"""
        with open('prompt_enhanced.md', 'r') as f:
            return f.read()

    async def process_query(self, query: str) -> Dict:
        """Process query with confidence scoring and cross-references"""

        # Step 1: Query classification
        query_type = self.classify_query(query)

        # Step 2: Query enhancement (synonyms, references)
        enhanced_query = self.enhance_query(query, query_type)

        # Step 3: Extract metadata filters from query
        filters = self.extract_filters(query)

        # Step 4: Generate embedding
        query_embedding = self.generate_embedding(enhanced_query)

        # Step 5: Perform hybrid retrieval
        retrieved_docs = self.vector_store.hybrid_search(
            query_embedding,
            enhanced_query,
            filters
        )

        # Step 6: CRAG validation
        validated_docs = self.crag_validation(retrieved_docs, query)

        # Step 7: Check for cross-references
        enriched_docs = await self.enrich_with_cross_references(validated_docs)

        # Step 8: Calculate overall confidence
        confidence = self.calculate_query_confidence(enriched_docs)

        # Step 9: Generate response with LLM
        response = await self.generate_response(
            query,
            enriched_docs,
            confidence,
            query_type
        )

        # Step 10: Format output with citations
        formatted_response = self.format_response(response, enriched_docs, confidence)

        return formatted_response

    def classify_query(self, query: str) -> str:
        """Classify query into categories A-E"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['door', 'hardware set', 'size', 'rating']):
            return 'A_schedule_lookup'
        elif any(word in query_lower for word in ['warranty', 'standard', 'submittal']):
            return 'B_spec_requirement'
        elif any(word in query_lower for word in ['which', 'reference', 'between']):
            return 'C_coordination'
        elif any(word in query_lower for word in ['how', 'why', 'best']):
            return 'D_interpretive'
        else:
            return 'E_unknown'

    def enhance_query(self, query: str, query_type: str) -> str:
        """Enhance query with synonyms and references"""
        enhancements = {
            'panic bar': 'panic bar OR exit device',
            'auto door': 'auto door OR automatic operator',
            'closer': 'closer OR door closer',
            'lockset': 'lockset OR lock OR mortise lock OR cylindrical lock'
        }

        enhanced = query
        for term, expansion in enhancements.items():
            if term in query.lower():
                enhanced = query.replace(term, expansion)

        # Add likely spec references based on query type
        if query_type == 'A_schedule_lookup':
            enhanced += " door schedule A9.11"
        elif 'hardware' in query.lower():
            enhanced += " 087100"
        elif 'wood door' in query.lower():
            enhanced += " 081416"

        return enhanced

    def crag_validation(self, docs: List[Dict], query: str) -> List[Dict]:
        """Validate document relevance using CRAG"""
        validated = []
        for doc in docs:
            # Simple CRAG: check if document contains query terms
            query_terms = set(query.lower().split())
            doc_terms = set(doc['content'].lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms)

            if overlap > 0.3 or doc['confidence'] == 'High':
                validated.append(doc)

        return validated

    async def enrich_with_cross_references(self, docs: List[Dict]) -> List[Dict]:
        """Fetch and add cross-referenced content"""
        enriched = []

        for doc in docs:
            xrefs = doc['metadata'].get('cross_references', [])
            if xrefs:
                # Fetch cross-referenced sections
                for xref in xrefs[:2]:  # Limit to 2 cross-refs
                    xref_docs = self.vector_store.hybrid_search(
                        query_embedding=None,
                        query_text=xref,
                        filters={'division_section': xref},
                        limit=1
                    )
                    if xref_docs:
                        doc['cross_referenced_content'] = xref_docs[0]['content']

            enriched.append(doc)

        return enriched

    def calculate_query_confidence(self, docs: List[Dict]) -> Tuple[str, float]:
        """Calculate overall confidence for the query"""
        if not docs:
            return ("Low", 0.0)

        scores = [d['score'] for d in docs]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.85 and docs[0]['confidence'] == 'High':
            return ("High", min(avg_score * 100, 100))
        elif avg_score > 0.65:
            return ("Medium", avg_score * 100)
        else:
            return ("Low", avg_score * 100)
```

## 5. Conflict Detection and Resolution

```python
class ConflictResolver:
    def __init__(self):
        self.hierarchy = [
            'door_schedule',
            'technical_specifications',
            'general_requirements',
            'industry_standards'
        ]

    def detect_conflicts(self, docs: List[Dict]) -> List[Dict]:
        """Detect conflicts between retrieved documents"""
        conflicts = []

        # Group docs by door number
        door_groups = {}
        for doc in docs:
            door_no = doc['metadata'].get('door_no')
            if door_no:
                if door_no not in door_groups:
                    door_groups[door_no] = []
                door_groups[door_no].append(doc)

        # Check for conflicts within each door group
        for door_no, group_docs in door_groups.items():
            fire_ratings = set()
            hardware_sets = set()

            for doc in group_docs:
                if doc['metadata'].get('fire_rating'):
                    fire_ratings.add(doc['metadata']['fire_rating'])
                if doc['metadata'].get('hardware_set'):
                    hardware_sets.add(doc['metadata']['hardware_set'])

            if len(fire_ratings) > 1:
                conflicts.append({
                    'type': 'fire_rating',
                    'door': door_no,
                    'values': list(fire_ratings)
                })

            if len(hardware_sets) > 1:
                conflicts.append({
                    'type': 'hardware_set',
                    'door': door_no,
                    'values': list(hardware_sets)
                })

        return conflicts

    def resolve_conflict(self, conflict: Dict, docs: List[Dict]) -> str:
        """Resolve conflicts based on hierarchy"""
        resolution = f"Conflict detected for {conflict['type']} on door {conflict['door']}:\n"
        resolution += f"Values found: {', '.join(conflict['values'])}\n"

        # Find source of each value
        for value in conflict['values']:
            for doc in docs:
                if doc['metadata'].get(conflict['type']) == value:
                    source = doc['metadata'].get('doc_title', 'Unknown')
                    resolution += f"- {value} from {source}\n"

        resolution += "\nResolution: Door Schedule takes precedence per project hierarchy."
        return resolution
```

## 6. Monitoring and Feedback Implementation

```python
import structlog
from prometheus_client import Counter, Histogram, Gauge
from datetime import datetime
import json

logger = structlog.get_logger()

# Prometheus metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries', ['query_type'])
confidence_gauge = Gauge('rag_confidence_score', 'Average confidence score')
response_time_histogram = Histogram('rag_response_seconds', 'Response time in seconds')
conflict_counter = Counter('rag_conflicts_detected', 'Conflicts detected')

class MonitoringSystem:
    def __init__(self, db_path="./monitoring.db"):
        self.db_path = db_path
        self.feedback_log = []

    def log_query(self, query: str, response: Dict, metrics: Dict):
        """Log query and response for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': metrics.get('query_type'),
            'confidence': metrics.get('confidence'),
            'response_time': metrics.get('response_time'),
            'retrieved_docs': metrics.get('doc_count'),
            'conflicts': metrics.get('conflicts', [])
        }

        # Update Prometheus metrics
        query_counter.labels(query_type=metrics.get('query_type', 'unknown')).inc()
        confidence_gauge.set(metrics.get('confidence_score', 0))
        response_time_histogram.observe(metrics.get('response_time', 0))

        if metrics.get('conflicts'):
            conflict_counter.inc()

        # Structured logging
        logger.info("query_processed", **log_entry)

        # Save to database for analysis
        self.save_to_db(log_entry)

    def analyze_feedback(self) -> Dict:
        """Analyze logged queries for improvement opportunities"""
        analysis = {
            'low_confidence_patterns': [],
            'common_conflicts': [],
            'failed_queries': [],
            'improvement_suggestions': []
        }

        # Analyze patterns (simplified)
        for entry in self.feedback_log:
            if entry.get('confidence') == 'Low':
                analysis['low_confidence_patterns'].append(entry['query'])

            if entry.get('conflicts'):
                analysis['common_conflicts'].extend(entry['conflicts'])

        # Generate suggestions
        if len(analysis['low_confidence_patterns']) > 5:
            analysis['improvement_suggestions'].append(
                "Consider adding more training data for low-confidence queries"
            )

        return analysis

    def export_metrics(self) -> str:
        """Export metrics for dashboard"""
        metrics = {
            'total_queries': len(self.feedback_log),
            'avg_confidence': self.calculate_avg_confidence(),
            'conflict_rate': self.calculate_conflict_rate(),
            'query_type_distribution': self.get_query_distribution()
        }
        return json.dumps(metrics, indent=2)
```

## 7. API Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="Door Specification RAG API")

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    include_cross_references: bool = True
    max_results: int = 10

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    confidence_score: float
    sources: List[Dict]
    conflicts: Optional[List[Dict]] = None
    metrics: Dict

# Initialize components
pdf_processor = EnhancedPDFProcessor()
vector_store = QdrantManager()
rag_pipeline = EnhancedRAGPipeline(vector_store, llm_client)
conflict_resolver = ConflictResolver()
monitoring = MonitoringSystem()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the enhanced RAG pipeline"""
    try:
        start_time = asyncio.get_event_loop().time()

        # Process query
        result = await rag_pipeline.process_query(request.query)

        # Detect conflicts
        conflicts = conflict_resolver.detect_conflicts(result['documents'])

        # Calculate metrics
        metrics = {
            'response_time': asyncio.get_event_loop().time() - start_time,
            'query_type': result.get('query_type'),
            'confidence_score': result.get('confidence_score'),
            'doc_count': len(result.get('documents', [])),
            'conflicts': conflicts
        }

        # Log for monitoring
        monitoring.log_query(request.query, result, metrics)

        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            confidence_score=result['confidence_score'],
            sources=result['sources'],
            conflicts=conflicts if conflicts else None,
            metrics=metrics
        )

    except Exception as e:
        logger.error("query_failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(query: str, rating: int, comment: Optional[str] = None):
    """Submit feedback on a query response"""
    monitoring.feedback_log.append({
        'query': query,
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    })
    return {"status": "Feedback received"}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return monitoring.export_metrics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "vector_store": "connected"}
```

## 8. Deployment Instructions

### A. Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process PDFs
python process_pdfs.py --input ./pdfs --output ./processed

# 3. Initialize vector database
python init_qdrant.py

# 4. Start API server
uvicorn main:app --reload --port 8000
```

### B. Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### C. Environment Variables
```env
# .env file
OPENAI_API_KEY=your_key_here
QDRANT_URL=localhost
QDRANT_PORT=6333
REDIS_URL=localhost:6379
LOG_LEVEL=INFO
ENABLE_MONITORING=true
CACHE_TTL=3600
```

## 9. Testing Strategy

```python
import pytest
from fastapi.testclient import TestClient

class TestEnhancedRAG:
    def test_door_lookup(self):
        """Test door schedule lookup with high confidence"""
        response = client.post("/query", json={
            "query": "What are the specifications for door 148A?"
        })
        assert response.status_code == 200
        assert response.json()["confidence"] == "High"
        assert "148A" in response.json()["answer"]

    def test_conflict_detection(self):
        """Test conflict detection between documents"""
        # Test with known conflicting data
        response = client.post("/query", json={
            "query": "Check for conflicts in door specifications"
        })
        assert "conflicts" in response.json()

    def test_cross_reference_enrichment(self):
        """Test cross-reference retrieval"""
        response = client.post("/query", json={
            "query": "What are the hardware requirements including cross-references?"
        })
        assert len(response.json()["sources"]) > 1
        assert any("cross_reference" in s for s in response.json()["sources"])
```

## 10. Performance Optimization Checklist

- [ ] Implement Redis caching for embeddings
- [ ] Use batch processing for PDF ingestion
- [ ] Enable GPU acceleration for embeddings (if available)
- [ ] Implement connection pooling for database
- [ ] Use async/await throughout the pipeline
- [ ] Enable response streaming for large results
- [ ] Implement rate limiting for API endpoints
- [ ] Use CDN for static assets
- [ ] Enable compression for API responses
- [ ] Monitor and optimize slow queries

## Next Steps

1. **Immediate**: Deploy local prototype with sample PDFs
2. **Week 1**: Complete PDF processing and vector database setup
3. **Week 2**: Implement full RAG pipeline with confidence scoring
4. **Week 3**: Add monitoring, feedback loops, and optimization
5. **Month 2**: Scale to production with cloud deployment

This implementation guide provides a complete technical blueprint for building the enhanced door specification RAG system with all the improvements from the enhanced prompt.