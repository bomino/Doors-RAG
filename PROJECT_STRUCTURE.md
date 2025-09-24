# Project Structure

## Directory Layout

```
doors-rag-app/
│
├── backend/                 # Backend API and processing
│   ├── api/                # API endpoints and models
│   │   ├── endpoints.py    # FastAPI route handlers
│   │   ├── models.py       # Pydantic models
│   │   └── __init__.py
│   │
│   ├── rag/                # RAG pipeline components
│   │   ├── chain.py        # Main RAG orchestration
│   │   ├── retriever.py    # Hybrid retrieval system
│   │   ├── door_parser.py  # Door specification parser
│   │   ├── entity_extractor.py  # Entity recognition
│   │   ├── hybrid_embeddings.py # OpenAI/local embeddings
│   │   ├── confidence_scorer.py # Confidence scoring
│   │   └── __init__.py
│   │
│   ├── processing/         # Document processing
│   │   ├── pdf_extractor.py     # PDF text extraction
│   │   ├── chunker.py           # Text chunking
│   │   ├── metadata_extractor.py # Metadata extraction
│   │   └── __init__.py
│   │
│   ├── scripts/            # Utility scripts
│   │   └── process_pdfs.py # PDF processing script
│   │
│   ├── monitoring/         # Monitoring and metrics
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── config.py          # Backend configuration
│   └── main.py            # FastAPI application
│
├── frontend/              # Streamlit frontend
│   ├── app.py            # Main Streamlit application
│   ├── vtx_logo1.png     # VTX logo for header
│   ├── vtx_logo2.png     # VTX logo for sidebar
│   └── __init__.py
│
├── config/               # Configuration files
│   └── settings.py       # Application settings
│
├── data/                 # Data storage
│   ├── pdfs/            # Source PDF documents
│   │   ├── 081000 - Doors and Frames.pdf
│   │   ├── 087100 - Door Hardware.pdf
│   │   └── Door Schedule and Details.pdf
│   │
│   └── vectordb/        # Qdrant vector database
│       └── [vector data files]
│
├── prompts/             # LLM prompts (archived)
│   ├── prompt_enhanced.md
│   ├── prompt_implementation_guide.md
│   └── promt1.md
│
├── docs/                # Documentation
│   └── [documentation files]
│
├── tests/               # Test files
│   └── [test files]
│
├── scripts/             # Shell scripts
│   └── [utility scripts]
│
├── docker-compose.yml   # Docker orchestration
├── Dockerfile          # Backend container
├── Dockerfile.frontend # Frontend container
├── .dockerignore       # Docker ignore patterns
├── .gitignore         # Git ignore patterns
├── .env               # Environment variables
├── .env.example       # Environment template
├── requirements_docker.txt  # Python dependencies
├── pyproject.toml     # Python project config
├── README.md          # Project documentation
└── PROJECT_STRUCTURE.md # This file

## Key Components

### Backend Services
- **FastAPI**: REST API framework
- **Qdrant**: Vector database for semantic search
- **OpenAI/Anthropic**: LLM providers for answer generation
- **SentenceTransformers**: Local embeddings fallback

### Frontend
- **Streamlit**: Web interface
- **Navy Blue & White Theme**: Custom CSS styling
- **VTX Branding**: Logo integration

### Docker Services
1. **backend**: FastAPI application (port 8000)
2. **frontend**: Streamlit UI (port 8502)
3. **qdrant**: Vector database (port 6333)
4. **redis**: Cache service (port 6379)

## Data Flow

1. **PDF Processing**:
   - PDFs → Text Extraction → Chunking → Embeddings → Qdrant

2. **Query Processing**:
   - User Query → Entity Extraction → Vector Search → Reranking → LLM → Response

3. **Door Specification Pipeline**:
   - Query → DoorScheduleParser → Structured Extraction → Formatted Response

## Configuration

Key settings are managed through:
- `.env`: API keys and environment-specific settings
- `config/settings.py`: Application constants
- `docker-compose.yml`: Container configuration

## Maintenance

- Logs: Check Docker container logs with `docker logs [container-name]`
- Database: Vector data persisted in `data/vectordb/`
- Updates: Restart services with `docker-compose restart`