# CLAUDE.md - AI Assistant Context File

## Project Overview
**Project Name**: Door Specifications RAG System
**Repository**: https://github.com/DefoxxAnalytics/Doors-RAG
**Author**: MLawali@versatexmsp.com
**Created**: September 2025
**Last Updated**: September 2025

## System Description
This is an enterprise-grade Retrieval-Augmented Generation (RAG) system designed to intelligently query door specifications from construction documentation PDFs. The system uses advanced NLP techniques, vector databases, and LLMs to provide accurate, contextual answers about door specifications, fire ratings, materials, and hardware requirements.

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Vector Database**: Qdrant (384-dimensional embeddings)
- **Embeddings**:
  - Primary: OpenAI text-embedding-ada-002 (1536 dims)
  - Fallback: SentenceTransformers all-MiniLM-L6-v2 (384 dims)
- **LLMs**:
  - Primary: OpenAI GPT-4o-mini
  - Fallback: Anthropic Claude-3-haiku
- **Cache**: Redis
- **PDF Processing**: PyPDF2, pdfplumber

### Frontend
- **Framework**: Streamlit
- **Theme**: Navy Blue & White with VTX branding
- **Logos**: vtx_logo1.png (header), vtx_logo2.png (sidebar)

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Services**: 4 containers (backend, frontend, qdrant, redis)
- **Ports**: 8000 (API), 8502 (UI), 6333 (Qdrant), 6379 (Redis)

## Key Features Implementation

### 1. Door Schedule Parser (`backend/rag/door_parser.py`)
- Handles multiple formats (pipe-separated, space-separated)
- Extracts 12+ fields per door (number, location, type, dimensions, etc.)
- Uses regex patterns for robust parsing
- 95% accuracy for structured data extraction

### 2. Entity Extraction (`backend/rag/entity_extractor.py`)
- Recognizes 11 entity types
- Smart query parsing with intent detection
- Entity-aware retrieval with boosting (+0.5 for exact matches)
- Relationship extraction between entities

### 3. Hybrid Retrieval (`backend/rag/retriever.py`)
- Semantic search with Qdrant
- Reranking based on entity matches
- Hierarchical chunking (parent: 800 tokens, child: 300 tokens)
- Query expansion for better recall

### 4. RAG Chain (`backend/rag/chain.py`)
- Dual LLM support with automatic fallback
- Intelligent fallback when APIs unavailable
- Conflict detection across documents
- Confidence scoring (High/Medium/Low)

## Critical Code Patterns

### Docker Network Communication
```python
# Frontend connects to backend using Docker service name
BACKEND_URL = os.getenv("API_URL", "http://backend:8000")
```

### Door Number Extraction
```python
door_patterns = [
    r'\b(\d{1,3}[A-Z]{1,2})\b',  # 148A, 627C
    r'\b([A-Z]\d{1,3}[A-Z]?)\b',  # B12, A101
]
```

### Entity Boosting
```python
if query_door_nums:
    for door_num in query_door_nums:
        if door_num in doc['text'].upper():
            relevance_score += 0.5  # Strong boost
```

## File Structure

```
doors-rag-app/
├── backend/
│   ├── rag/           # Core RAG components
│   ├── processing/    # Document processing
│   ├── api/          # FastAPI endpoints
│   └── scripts/      # Utility scripts
├── frontend/
│   └── app.py        # Streamlit UI
├── data/
│   ├── pdfs/         # Source documents
│   └── vectordb/     # Qdrant storage
└── config/
    └── settings.py   # Centralized config
```

## Common Commands

### Start System
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

### Docker Operations
```bash
docker-compose up -d              # Start all services
docker-compose down              # Stop all services
docker-compose restart frontend  # Restart frontend
docker logs doors-backend -f    # View backend logs
```

### Process PDFs
```bash
docker exec doors-backend python backend/scripts/process_pdfs.py
```

## Environment Variables
```env
OPENAI_API_KEY=sk-proj-...      # Required
ANTHROPIC_API_KEY=sk-ant-...    # Optional fallback
QDRANT_HOST=qdrant              # Default: qdrant
QDRANT_PORT=6333                # Default: 6333
```

## Performance Metrics
- Query response: <2 seconds
- Embedding generation: ~3 seconds/10 docs
- Accuracy: 95% for structured data
- Database: 670+ vectors from 3 PDFs
- Context window: 8 docs × 800 tokens

## UI Components

### Header
- Navy Blue gradient background
- VTX logo on far left
- Centered title and subtitle

### Search Interface
- Horizontally aligned: Input (70%), Search (15%), Clear (15%)
- Auto-clears after search
- Enter key support

### Sidebar
- VTX logo2 at top
- System stats (documents, queries)
- Quick door lookup
- Sample queries
- Backend status indicator

### Response Display
- User messages: Light blue background
- Assistant messages: White with shadow
- Confidence scores with color coding
- Source citations
- Conflict detection warnings

## Testing Queries

### Basic Door Query
```
What are the specifications for door 148A?
```

### Fire Rating Query
```
List all doors with 90 MIN fire rating
```

### Material Query
```
What materials are used for exterior doors?
```

## Known Issues & Solutions

### Backend Connection Error
- Check Docker network: `docker network ls`
- Verify env variable: `API_URL=http://backend:8000`
- Restart frontend: `docker-compose restart frontend`

### Logo Not Displaying
- Path should be: `frontend/vtx_logo1.png`
- Check mounting: `docker exec doors-frontend ls frontend/`

### PDF Processing Fails
- Check memory: Need 4GB+ RAM
- Verify PDF format: Must be searchable PDFs
- Check logs: `docker logs doors-backend`

## Development Guidelines

### Code Style
- No comments unless essential
- Use type hints for functions
- Follow existing patterns
- Test with both LLMs

### Git Workflow
1. Never commit API keys
2. Use .env for secrets
3. Test Docker build before pushing
4. Update CLAUDE.md for AI context

### Adding New Features
1. Update entity patterns if needed
2. Test with sample PDFs
3. Verify Docker compatibility
4. Update documentation

## Deployment Checklist

- [ ] Remove .env from git tracking
- [ ] Update API keys in production
- [ ] Test all Docker services
- [ ] Verify PDF processing
- [ ] Check UI responsiveness
- [ ] Test query accuracy
- [ ] Monitor memory usage
- [ ] Backup vector database

## Contact
**Developer**: MLawali@versatexmsp.com
**Organization**: VersaTex MSP
**Repository**: https://github.com/DefoxxAnalytics/Doors-RAG

---

*This file provides context for AI assistants working on the project. Keep it updated with critical implementation details and patterns.*