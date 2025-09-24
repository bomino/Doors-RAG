# Door Specifications RAG System 🚪

[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)](https://openai.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red)]()

An enterprise-grade Retrieval-Augmented Generation (RAG) system for querying door specifications, construction documentation, and architectural details from PDF documents. Built with modern AI technologies and featuring a Navy Blue & White themed interface with VTX branding.

![Door RAG System](assets/placeholder/banner.png)

## 🌟 Features

### Core Capabilities
- **🔍 Intelligent Document Search**: Semantic search across door specification PDFs
- **📊 Structured Data Extraction**: Automatically parses door schedules and specifications
- **🏷️ Entity Recognition**: Identifies and extracts 11+ entity types including:
  - Door numbers (148A, B12, etc.)
  - Room locations
  - Fire ratings
  - Hardware groups
  - Materials and finishes
  - Dimensions
  - Frame types
  - Standards and warranties

### Advanced Features
- **🧠 Hybrid Embeddings**: Uses OpenAI embeddings with local fallback
- **🎯 Smart Query Parser**: Understands query intent and context
- **⚡ Entity-Aware Retrieval**: Boosts results based on entity matches
- **📚 Hierarchical Chunking**: Parent-child chunks for better context
- **⚠️ Conflict Detection**: Identifies inconsistencies across documents
- **📄 Multi-Format Support**: Handles both pipe-separated and space-separated door schedules

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Streamlit     │────▶│    FastAPI      │────▶│    Qdrant       │
│   Frontend      │     │    Backend      │     │  Vector DB      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  LLM Integration    │
                    │  - OpenAI (primary) │
                    │  - Anthropic (backup)│
                    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (recommended) or Anthropic API key
- 4GB+ RAM available
- Windows 10/11, macOS, or Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DefoxxAnalytics/Doors-RAG.git
cd Doors-RAG/doors-rag-app
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-proj-...
# ANTHROPIC_API_KEY=sk-ant-... (optional fallback)
```

3. **Start the system**

**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
./start.sh
```

4. **Access the application**
- 🌐 Frontend: http://localhost:8502
- 📡 API Documentation: http://localhost:8000/docs
- 🗄️ Qdrant Dashboard: http://localhost:6333/dashboard

## 📖 Usage Examples

### Query Door Specifications
```
"What are the specifications for door 148A?"

Response:
**Door 148A Specifications:**
**Location:** RETAIL
**Door Type:** F1 (Flush door, type 1)
**Dimensions:** 3' - 0" wide x 7' - 0" high
**Thickness:** 0' - 1 3/4"
**Material:** HM (Hollow Metal)
**Finish:** Paint finish on both sides
**Fire Rating:** 90 MIN (90-minute fire rating)
**Hardware Group:** 86.0
**Frame Type:** HMF1 (Hollow Metal Frame type 1)
```

### Query Standards
```
"What fire rating standards are used?"
```

### Query Materials
```
"What materials are commonly used for doors?"
```

## 🔧 API Documentation

### REST Endpoints

#### Query Endpoint
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are the specifications for door 116B?",
  "filters": {
    "fire_rating": "90 MIN"  // optional
  }
}
```

#### Response Format
```json
{
  "answer": "Detailed answer with markdown formatting...",
  "confidence": "High",
  "confidence_score": 0.95,
  "sources": [
    "document.pdf (Page 23)",
    "schedule.pdf (Page 45)"
  ],
  "conflicts": []
}
```

## ⚙️ Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=your-openai-key

# Optional
ANTHROPIC_API_KEY=your-anthropic-key  # Fallback LLM
QDRANT_HOST=qdrant  # Default: qdrant
QDRANT_PORT=6333    # Default: 6333
```

### Docker Services
- **backend**: FastAPI application (port 8000)
- **frontend**: Streamlit UI (port 8502)
- **qdrant**: Vector database (port 6333)
- **redis**: Cache service (port 6379)

## 🛠️ Development

### Running Locally (without Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Processing New PDFs
Place PDFs in `data/pdfs/` and run:
```bash
docker exec doors-backend python backend/scripts/process_pdfs.py
```

### Testing
```python
from rag.door_parser import DoorScheduleParser
parser = DoorScheduleParser()
info = parser.parse_door_info(text, "148A")
print(parser.format_door_specifications(info))
```

## 📊 Performance Metrics

- **Embedding Generation**: ~2-3 seconds for 10 documents
- **Query Response Time**: <2 seconds average
- **Accuracy**: 95% confidence for structured door data
- **Database Size**: 670+ vectors from 3 PDFs
- **Context Window**: 8 documents × 800 tokens

## 🐛 Troubleshooting

### Common Issues

1. **Cannot connect to backend**
   ```bash
   docker-compose restart frontend
   docker logs doors-backend --tail 50
   ```

2. **No results returned**
   - Verify PDFs were processed
   - Check Qdrant has 600+ vectors
   - Ensure API keys are configured

3. **Memory issues**
   ```bash
   docker-compose down
   docker system prune -a
   docker-compose up -d
   ```

## 📁 Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory layout.

## 🤝 Contributing

This is a proprietary project. For internal contributions:
1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📝 License

Proprietary - Internal Use Only
© 2025 VersaTex MSP. All rights reserved.

## 👨‍💻 Author

**Developed by MLawali@versatexmsp.com**

## 🙏 Acknowledgments

- OpenAI for GPT-4 models
- Anthropic for Claude models
- Qdrant for vector database
- Streamlit for the frontend framework
- FastAPI for the backend framework

## 📞 Support

For issues or questions:
- Create an issue in this repository
- Contact: MLawali@versatexmsp.com

---

*Door Specifications RAG System - Intelligent Document Search for Construction Professionals*
