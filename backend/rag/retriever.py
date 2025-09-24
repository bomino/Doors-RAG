"""Document retrieval module"""
from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .entity_extractor import EntityExtractor, SmartQueryParser, EntityAwareRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval with semantic and keyword search"""

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer, collection_name: str = "door_specifications"):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Initialize entity-aware components
        self.entity_extractor = EntityExtractor()
        self.query_parser = SmartQueryParser()
        self.entity_retriever = EntityAwareRetriever(
            qdrant_client, embedding_model, self.entity_extractor, self.query_parser
        )

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant documents using entity-aware semantic search"""
        try:
            # First, try entity-aware retrieval
            parsed_query = self.query_parser.parse_query(query)

            # If specific entities are found, use entity-aware retrieval
            if parsed_query['entities'] and parsed_query['intent'] != 'general':
                logger.info(f"Using entity-aware retrieval for: {parsed_query['entities']}")
                entity_results = self.entity_retriever.retrieve_by_entity(query, top_k)

                # If we got good results from entity search, use them
                if entity_results and len(entity_results) >= 3:
                    return entity_results

            # Otherwise, fall back to expanded query search
            # Query expansion - generate multiple search queries for better coverage
            expanded_queries = self._expand_query(query)

            all_documents = []
            seen_ids = set()

            for exp_query in expanded_queries:
                # Create query embedding
                query_vector = self.embedding_model.encode([exp_query])[0].tolist()

                # Build filter conditions for Qdrant
                qdrant_filter = None
                if filters:
                    conditions = []

                    # Door number filter
                    if filters.get("door_no"):
                        conditions.append({
                            "key": "metadata.door_numbers",
                            "match": {"any": [filters["door_no"].lower()]}
                        })

                    # Fire rating filter
                    if filters.get("fire_rating"):
                        conditions.append({
                            "key": "metadata.fire_rating",
                            "match": {"any": [filters["fire_rating"]]}
                        })

                    if conditions:
                        qdrant_filter = {"must": conditions}

                # Search in Qdrant - get more candidates for reranking
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k * 2,  # Get more candidates
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False
                )

                # Format results
                for result in search_results:
                    doc_id = result.payload.get("id", "")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc = {
                            "id": doc_id,
                            "text": result.payload.get("text", ""),
                            "score": float(result.score),
                            "metadata": result.payload.get("metadata", {}),
                            "source": result.payload.get("pdf_name", ""),
                            "page": result.payload.get("page_num", 0),
                            "chunk_type": result.payload.get("chunk_type", ""),
                            "parent_id": result.payload.get("parent_id", "")
                        }
                        all_documents.append(doc)

            # Rerank documents based on relevance
            reranked_docs = self._rerank_documents(all_documents, query, top_k)

            # Fetch parent chunks for better context
            final_docs = self._get_parent_context(reranked_docs)

            logger.info(f"Retrieved and reranked {len(final_docs)} documents for query: {query}")
            return final_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations for better recall"""
        import re
        query_lower = query.lower()
        expanded = [query]  # Original query

        # Extract door numbers from query
        door_pattern = r'\b(\d{1,3}[a-zA-Z]?)\b'
        door_matches = re.findall(door_pattern, query)

        if door_matches:
            # Add specific door number searches
            for door_num in door_matches:
                expanded.append(f"door {door_num} schedule specifications")
                expanded.append(f"{door_num} door frame hardware")

        # Add domain-specific expansions
        if "fire rating" in query_lower or "fire" in query_lower:
            expanded.append("NFPA 80 fire rated doors standards compliance")
            expanded.append("fire protection rating temperature rise limit")

        if "hardware" in query_lower:
            expanded.append("door hinges locks handles cylinders strikes closers")
            expanded.append("hardware specifications requirements mounting installation")

        if "warranty" in query_lower or "guarantee" in query_lower:
            expanded.append("manufacturer warranty guarantee period coverage")
            expanded.append("warranty terms conditions maintenance requirements")

        if "specification" in query_lower or "spec" in query_lower:
            expanded.append("door specifications requirements standards compliance dimensions")

        # Add technical terms if present
        if any(term in query_lower for term in ["hinge", "lock", "handle", "cylinder"]):
            expanded.append("door hardware components installation requirements")

        return expanded[:4]  # Limit to top 4 expansions

    def _rerank_documents(self, documents: List[Dict], query: str, top_k: int) -> List[Dict]:
        """Rerank documents based on multiple relevance factors"""
        import re
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Extract door numbers from query for special boosting
        door_pattern = r'\b(\d{1,3}[a-zA-Z]?)\b'
        query_door_nums = re.findall(door_pattern, query.upper())

        for doc in documents:
            # Base score from vector similarity
            relevance_score = doc['score']

            # Boost for exact term matches
            doc_text_lower = doc['text'].lower()
            term_matches = sum(1 for term in query_terms if term in doc_text_lower)
            relevance_score += term_matches * 0.1

            # STRONG boost for exact door number matches in text
            if query_door_nums:
                for door_num in query_door_nums:
                    if door_num in doc['text'].upper():
                        relevance_score += 0.5  # Strong boost for exact door match

            # Boost for chunk type (prefer parent chunks for context)
            if doc.get('chunk_type') == 'parent':
                relevance_score += 0.15

            # Boost for specific metadata matches
            metadata = doc.get('metadata', {})

            # Fire rating boost
            if "fire" in query_lower and metadata.get('fire_rating'):
                relevance_score += 0.2

            # Door number boost
            if metadata.get('door_numbers'):
                for door_no in metadata['door_numbers']:
                    if door_no in query_lower:
                        relevance_score += 0.3

            # Content type boost
            content_type = metadata.get('content_type', '')
            if content_type == 'specification' and 'spec' in query_lower:
                relevance_score += 0.15
            elif content_type == 'schedule' and 'schedule' in query_lower:
                relevance_score += 0.15
            elif content_type == 'warranty' and 'warranty' in query_lower:
                relevance_score += 0.2

            # Update document with reranked score
            doc['reranked_score'] = relevance_score

        # Sort by reranked score
        reranked = sorted(documents, key=lambda x: x.get('reranked_score', 0), reverse=True)

        return reranked[:top_k]

    def _get_parent_context(self, documents: List[Dict]) -> List[Dict]:
        """Fetch parent chunks for child chunks to provide better context"""
        enhanced_docs = []

        for doc in documents:
            # If it's a child chunk, try to get parent for more context
            if doc.get('chunk_type') == 'child' and doc.get('parent_id'):
                try:
                    # Search for parent chunk
                    parent_results = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter={
                            "must": [{"key": "id", "match": {"value": doc['parent_id']}}]
                        },
                        limit=1,
                        with_payload=True
                    )

                    if parent_results and parent_results[0]:
                        parent_doc = parent_results[0][0]
                        # Use parent text for better context
                        doc['text'] = parent_doc.payload.get('text', doc['text'])
                        doc['enhanced_context'] = True
                except Exception as e:
                    logger.debug(f"Could not fetch parent chunk: {e}")

            enhanced_docs.append(doc)

        return enhanced_docs
