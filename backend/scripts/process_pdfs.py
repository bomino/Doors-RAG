#!/usr/bin/env python3
"""
PDF Processing Script for Door Specification RAG System
Processes PDF documents and creates vector embeddings
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import fitz  # PyMuPDF
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF processor with dual extraction methods"""

    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Use local embedding model instead of OpenAI
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Collection name
        self.collection_name = "door_specifications"

        # Chunk settings
        self.parent_chunk_size = 800  # Parent chunks: 600-1000 tokens
        self.child_chunk_size = 300   # Child chunks: 200-400 tokens
        self.chunk_overlap = 100      # 20% overlap

    async def initialize_collection(self):
        """Initialize Qdrant collection with proper settings"""
        try:
            # Delete existing collection if it exists
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass  # Collection doesn't exist

            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def extract_text_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF - better for layout preservation"""
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with layout preservation
                text_dict = page.get_text("dict")

                # Extract plain text
                text = page.get_text()

                # Extract tables and structured content
                tables = []
                blocks = text_dict.get("blocks", [])

                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                # Look for table-like structures
                                font_size = span.get("size", 0)
                                if font_size < 10:  # Small text often indicates table content
                                    tables.append(span.get("text", ""))

                pages.append({
                    "page_num": page_num + 1,
                    "text": text.strip(),
                    "tables": tables,
                    "method": "pymupdf"
                })

            doc.close()
            return pages

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return []

    def extract_text_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using pdfplumber - better for tables"""
        try:
            pages = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract plain text
                    text = page.extract_text() or ""

                    # Extract tables
                    tables = []
                    page_tables = page.extract_tables()

                    for table in page_tables:
                        if table:
                            # Convert table to text format
                            table_text = []
                            for row in table:
                                if row:
                                    clean_row = [cell.strip() if cell else "" for cell in row]
                                    table_text.append(" | ".join(clean_row))
                            tables.append("\n".join(table_text))

                    pages.append({
                        "page_num": page_num + 1,
                        "text": text.strip(),
                        "tables": tables,
                        "method": "pdfplumber"
                    })

            return pages

        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return []

    def merge_extractions(self, pymupdf_pages: List[Dict], pdfplumber_pages: List[Dict]) -> List[Dict[str, Any]]:
        """Merge results from both extraction methods"""
        merged_pages = []

        for i in range(min(len(pymupdf_pages), len(pdfplumber_pages))):
            pymupdf_page = pymupdf_pages[i]
            pdfplumber_page = pdfplumber_pages[i]

            # Use the longer text extraction
            if len(pymupdf_page["text"]) > len(pdfplumber_page["text"]):
                base_text = pymupdf_page["text"]
            else:
                base_text = pdfplumber_page["text"]

            # Combine tables from both methods
            all_tables = []
            all_tables.extend(pdfplumber_page.get("tables", []))
            all_tables.extend(pymupdf_page.get("tables", []))

            # Remove duplicates and empty tables
            unique_tables = []
            for table in all_tables:
                if table and table.strip() and table not in unique_tables:
                    unique_tables.append(table)

            merged_pages.append({
                "page_num": i + 1,
                "text": base_text,
                "tables": unique_tables,
                "extraction_method": "hybrid"
            })

        return merged_pages

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def create_hierarchical_chunks(self, pages: List[Dict[str, Any]], pdf_name: str) -> List[Dict[str, Any]]:
        """Create hierarchical parent-child chunks"""
        all_chunks = []

        for page in pages:
            page_text = page["text"]
            tables_text = "\n\n".join(page.get("tables", []))
            combined_text = f"{page_text}\n\n{tables_text}".strip()

            if not combined_text:
                continue

            # Create parent chunks
            parent_chunks = self.split_text(combined_text, self.parent_chunk_size)

            for parent_idx, parent_text in enumerate(parent_chunks):
                parent_id = f"{pdf_name}_page_{page['page_num']}_parent_{parent_idx}"

                # Create child chunks from parent
                child_chunks = self.split_text(parent_text, self.child_chunk_size)

                for child_idx, child_text in enumerate(child_chunks):
                    child_id = f"{parent_id}_child_{child_idx}"

                    # Extract metadata
                    metadata = self.extract_metadata(child_text, page, pdf_name)

                    all_chunks.append({
                        "id": child_id,
                        "parent_id": parent_id,
                        "text": child_text,
                        "page_num": page["page_num"],
                        "pdf_name": pdf_name,
                        "chunk_type": "child",
                        "metadata": metadata
                    })

                # Also store parent chunk
                parent_metadata = self.extract_metadata(parent_text, page, pdf_name)
                all_chunks.append({
                    "id": parent_id,
                    "parent_id": None,
                    "text": parent_text,
                    "page_num": page["page_num"],
                    "pdf_name": pdf_name,
                    "chunk_type": "parent",
                    "metadata": parent_metadata
                })

        return all_chunks

    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []

        # Split by sentences first
        sentences = text.replace('. ', '.\n').split('\n')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_chunk = f"{current_chunk} {sentence}".strip()

            if self.count_tokens(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def extract_metadata(self, text: str, page: Dict, pdf_name: str) -> Dict[str, Any]:
        """Extract metadata from text content"""
        metadata = {
            "source": pdf_name,
            "page": page["page_num"],
            "extraction_method": page.get("extraction_method", "unknown")
        }

        text_lower = text.lower()

        # Extract door numbers
        import re
        door_patterns = [
            r'\bdoor\s+(\d+[a-z]?)\b',
            r'\b(\d{3}[a-z]?)\s+door\b',
            r'\b(\d+[a-z]?)\s*[-–]\s*door\b'
        ]

        doors = set()
        for pattern in door_patterns:
            matches = re.findall(pattern, text_lower)
            doors.update(matches)

        if doors:
            metadata["door_numbers"] = list(doors)

        # Extract fire ratings
        fire_rating_patterns = [
            r'(\d+)\s*(?:minute|min)\s*fire\s*rat',
            r'fire\s*rat\w*\s*(?:of\s*)?(\d+)\s*(?:minute|min)',
            r'(\d+)\s*hr\s*fire\s*rat'
        ]

        fire_ratings = set()
        for pattern in fire_rating_patterns:
            matches = re.findall(pattern, text_lower)
            fire_ratings.update([f"{m} minutes" for m in matches])

        if fire_ratings:
            metadata["fire_rating"] = list(fire_ratings)

        # Extract hardware mentions
        hardware_keywords = [
            "hinge", "lock", "handle", "lever", "deadbolt", "cylinder",
            "strike", "weatherstrip", "threshold", "closer", "exit device"
        ]

        found_hardware = [hw for hw in hardware_keywords if hw in text_lower]
        if found_hardware:
            metadata["hardware_types"] = found_hardware

        # Extract specifications
        if any(word in text_lower for word in ["specification", "spec", "requirement"]):
            metadata["content_type"] = "specification"
        elif any(word in text_lower for word in ["schedule", "table", "list"]):
            metadata["content_type"] = "schedule"
        elif any(word in text_lower for word in ["warranty", "guarantee"]):
            metadata["content_type"] = "warranty"
        else:
            metadata["content_type"] = "general"

        return metadata

    async def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks using local model"""
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")

        # Process in batches for memory efficiency
        batch_size = 100
        embedded_chunks = []

        for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]

            try:
                # Use local sentence transformer model
                vectors = self.embedding_model.encode(texts, convert_to_numpy=True)

                for j, chunk in enumerate(batch):
                    chunk["vector"] = vectors[j].tolist()
                    embedded_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i}: {e}")
                # Continue with next batch
                continue

        return embedded_chunks

    async def store_in_qdrant(self, chunks: List[Dict[str, Any]]):
        """Store chunks in Qdrant vector database"""
        logger.info(f"Storing {len(chunks)} chunks in Qdrant...")

        points = []
        for chunk in chunks:
            if "vector" not in chunk:
                continue

            point = PointStruct(
                id=hash(chunk["id"]) % (2**63),  # Convert string ID to int
                vector=chunk["vector"],
                payload={
                    "id": chunk["id"],
                    "parent_id": chunk["parent_id"],
                    "text": chunk["text"],
                    "page_num": chunk["page_num"],
                    "pdf_name": chunk["pdf_name"],
                    "chunk_type": chunk["chunk_type"],
                    "metadata": chunk["metadata"]
                }
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]

            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            except Exception as e:
                logger.error(f"Error uploading batch {i}: {e}")
                continue

        logger.info("Successfully stored all chunks in Qdrant")

    async def process_pdf(self, pdf_path: str) -> bool:
        """Process a single PDF file"""
        pdf_name = Path(pdf_path).stem
        logger.info(f"Processing PDF: {pdf_name}")

        try:
            # Extract using both methods
            logger.info("Extracting text with PyMuPDF...")
            pymupdf_pages = self.extract_text_pymupdf(pdf_path)

            logger.info("Extracting text with pdfplumber...")
            pdfplumber_pages = self.extract_text_pdfplumber(pdf_path)

            # Merge extractions
            logger.info("Merging extractions...")
            merged_pages = self.merge_extractions(pymupdf_pages, pdfplumber_pages)

            if not merged_pages:
                logger.warning(f"No content extracted from {pdf_name}")
                return False

            # Create hierarchical chunks
            logger.info("Creating hierarchical chunks...")
            chunks = self.create_hierarchical_chunks(merged_pages, pdf_name)

            if not chunks:
                logger.warning(f"No chunks created from {pdf_name}")
                return False

            logger.info(f"Created {len(chunks)} chunks from {pdf_name}")

            # Create embeddings
            embedded_chunks = await self.create_embeddings(chunks)

            if not embedded_chunks:
                logger.warning(f"No embeddings created for {pdf_name}")
                return False

            # Store in Qdrant
            await self.store_in_qdrant(embedded_chunks)

            logger.info(f"Successfully processed {pdf_name}")
            return True

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return False

async def main():
    """Main processing function"""
    qdrant_host = os.getenv("QDRANT_URL", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    # Initialize processor with local embeddings
    processor = PDFProcessor(qdrant_host, qdrant_port)

    # Initialize collection
    await processor.initialize_collection()

    # Find PDF files
    data_dir = Path(__file__).parent.parent.parent / "data" / "pdfs"
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF
    successful = 0
    for pdf_path in pdf_files:
        success = await processor.process_pdf(str(pdf_path))
        if success:
            successful += 1

    logger.info(f"Processing complete: {successful}/{len(pdf_files)} PDFs processed successfully")

if __name__ == "__main__":
    asyncio.run(main())