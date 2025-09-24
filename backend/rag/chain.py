"""RAG chain implementation with OpenAI and Anthropic support"""
from typing import Dict, List, Optional
import logging
import os
import re
from anthropic import Anthropic
from .door_parser import DoorScheduleParser

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

class RAGChain:
    """Main RAG pipeline with enhanced door parsing and LLM support"""

    def __init__(self, retriever, confidence_scorer=None):
        self.retriever = retriever
        self.confidence_scorer = confidence_scorer
        self.door_parser = DoorScheduleParser()

        # Try OpenAI first (better availability)
        self.openai_client = None
        if OPENAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and not openai_key.startswith("your-"):
                try:
                    self.openai_client = OpenAI(api_key=openai_key)
                    logger.info("Using OpenAI for answer generation")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI: {e}")

        # Fallback to Anthropic
        self.anthropic_client = None
        if not self.openai_client:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key and not api_key.startswith("your-"):
                try:
                    self.anthropic_client = Anthropic(api_key=api_key)
                    logger.info("Using Anthropic for answer generation")
                except Exception as e:
                    logger.warning(f"Failed to initialize Anthropic: {e}")

        if not self.openai_client and not self.anthropic_client:
            logger.warning("No LLM API configured - using intelligent fallback")

    async def process(self, query: str, filters: Optional[Dict] = None, max_results: int = 10,
                     include_cross_references: bool = True) -> Dict:
        """Process a query and generate an answer"""
        try:
            # Enhanced door pattern to catch more formats
            door_patterns = [
                r'\b(\d{1,3}[A-Z]{1,2})\b',  # 148A, 627C
                r'\b([A-Z]\d{1,3}[A-Z]?)\b',  # B12, A101
                r'\b(\d{1,3})\b(?=\s*(?:door|Door|DOOR))',  # 111 door
                r'(?:door|Door|DOOR)\s+(\d{1,3}[A-Z]?)',  # door 148A
            ]

            door_matches = []
            for pattern in door_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                door_matches.extend([m.upper() for m in matches])

            # Remove duplicates while preserving order
            seen = set()
            unique_doors = []
            for door in door_matches:
                if door not in seen:
                    seen.add(door)
                    unique_doors.append(door)
            door_matches = unique_doors

            # Retrieve relevant documents
            documents = self.retriever.retrieve(query, top_k=max_results, filters=filters)

            if not documents:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "confidence": "Low",
                    "confidence_score": 0.0,
                    "sources": [],
                    "conflicts": []
                }

            # Check if this is a door-specific query
            query_lower = query.lower()
            is_door_query = (
                door_matches and
                any(term in query_lower for term in [
                    'specification', 'spec', 'detail', 'info', 'information',
                    'what are', 'what is', 'door', 'dimensions', 'fire rating',
                    'material', 'hardware', 'type', 'finish', 'frame'
                ])
            )

            # For door queries, try structured extraction first
            if is_door_query:
                all_door_info = {}

                for target_door in door_matches:
                    best_info = None
                    best_score = 0

                    # Search through documents for door schedule information
                    for doc in documents:
                        parsed_info = self.door_parser.parse_door_info(doc['text'], target_door)
                        if parsed_info:
                            # Score based on completeness
                            score = len(parsed_info)
                            if score > best_score:
                                best_info = parsed_info
                                best_score = score

                    if best_info:
                        all_door_info[target_door] = best_info

                # If we found structured door info, format it
                if all_door_info:
                    if len(all_door_info) == 1:
                        # Single door query
                        door_num, info = list(all_door_info.items())[0]
                        answer = self.door_parser.format_door_specifications(info)
                    else:
                        # Multiple doors query
                        answer_parts = []
                        for door_num, info in all_door_info.items():
                            answer_parts.append(self.door_parser.format_door_specifications(info))
                            answer_parts.append("")  # Add spacing
                        answer = "\n".join(answer_parts).strip()

                    # High confidence for structured extraction
                    confidence_score = 0.95
                    confidence = "High"
                    sources = self._format_sources(documents[:3])

                    return {
                        "answer": answer,
                        "confidence": confidence,
                        "confidence_score": confidence_score,
                        "sources": sources,
                        "conflicts": []
                    }

            # Otherwise, use LLM for general answer generation
            answer, confidence_score = await self._generate_answer(query, documents)

            # Determine confidence level
            confidence = self._get_confidence_level(confidence_score)

            # Format sources
            sources = self._format_sources(documents)

            # Detect conflicts
            conflicts = self._detect_conflicts(documents) if include_cross_references else []

            return {
                "answer": answer,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "sources": sources,
                "conflicts": conflicts
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "confidence": "Error",
                "confidence_score": 0.0,
                "sources": [],
                "conflicts": []
            }

    async def _generate_answer(self, query: str, documents: List[Dict]) -> tuple[str, float]:
        """Generate answer using LLM (OpenAI or Anthropic)"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc['source']} (Page {doc['page']})\n{doc['text']}"
                for doc in documents[:8]
            ])

            # System prompt
            system_prompt = """You are a technical assistant specializing in door specifications and construction documentation.
            Provide accurate, detailed answers based on the provided context.

            For door specifications, format your answer clearly with:
            - Door number and location
            - Dimensions (width x height)
            - Material and finish
            - Fire rating (if applicable)
            - Hardware group
            - Any special requirements

            For other queries, provide comprehensive but concise information.
            If information is not found in the context, say so clearly."""

            user_prompt = f"""Context from door specification documents:
            {context}

            Question: {query}

            Please provide a detailed answer based on the context above. Format lists and specifications clearly.
            If the context doesn't contain relevant information, indicate that clearly."""

            # Try OpenAI first
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # Fast and cost-effective
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content

                    # Calculate confidence
                    confidence_score = 0.85
                    if "not found" in answer.lower() or "no information" in answer.lower():
                        confidence_score = 0.4
                    elif "based on the context" in answer.lower():
                        confidence_score = 0.9

                    return answer, confidence_score

                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")

            # Try Anthropic as backup
            if self.anthropic_client:
                try:
                    response = self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1500,
                        temperature=0.1,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    answer = response.content[0].text

                    # Calculate confidence
                    confidence_score = 0.85
                    if "not found" in answer.lower() or "no information" in answer.lower():
                        confidence_score = 0.4
                    elif "based on the context" in answer.lower():
                        confidence_score = 0.9

                    return answer, confidence_score

                except Exception as e:
                    logger.error(f"Anthropic API error: {e}")

            # If both fail, use intelligent fallback
            return self._generate_intelligent_fallback(documents, query), 0.6

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._generate_intelligent_fallback(documents, query), 0.5

    def _generate_intelligent_fallback(self, documents: List[Dict], query: str) -> str:
        """Generate intelligent answer without LLM"""

        if not documents:
            return "No relevant information found."

        query_lower = query.lower()

        # Check for door-specific queries
        door_pattern = r'\b(\d{1,3}[A-Z]?)\b'
        door_matches = re.findall(door_pattern, query.upper())

        if door_matches:
            # Try to extract relevant information for the door
            for door_num in door_matches:
                for doc in documents[:5]:
                    if door_num in doc['text'].upper():
                        # Extract sentences mentioning the door
                        sentences = doc['text'].split('.')
                        relevant = []
                        for sent in sentences:
                            if door_num in sent.upper():
                                relevant.append(sent.strip())

                        if relevant:
                            answer = f"**Information for Door {door_num}:**\n\n"
                            answer += "\n".join(f"- {sent}" for sent in relevant[:5])
                            answer += f"\n\n**Source:** {doc['source']} (Page {doc['page']})"
                            return answer

        # Extract key information based on query type
        extracted_info = []

        for doc in documents[:5]:
            text = doc['text']

            # Extract relevant sentences based on keywords
            keywords = []
            if 'fire' in query_lower:
                keywords.extend(['fire', 'rating', 'NFPA', 'minute', 'MIN'])
            if 'hardware' in query_lower:
                keywords.extend(['hardware', 'hinge', 'lock', 'handle', 'cylinder'])
            if 'specification' in query_lower or 'spec' in query_lower:
                keywords.extend(['specification', 'requirement', 'must', 'shall'])
            if 'warranty' in query_lower:
                keywords.extend(['warranty', 'guarantee', 'year', 'manufacturer'])
            if 'dimension' in query_lower or 'size' in query_lower:
                keywords.extend(['width', 'height', 'dimension', "'-", '"-'])

            if not keywords:
                # General query - extract first few sentences
                sentences = text.split('.')[:3]
                extracted_info.extend([s.strip() for s in sentences if len(s.strip()) > 20])
            else:
                # Extract sentences with keywords
                sentences = text.split('.')
                for sent in sentences:
                    sent_lower = sent.lower()
                    if any(kw.lower() in sent_lower for kw in keywords):
                        cleaned = sent.strip()
                        if len(cleaned) > 20 and len(cleaned) < 500:
                            extracted_info.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        unique_info = []
        for info in extracted_info:
            info_lower = info.lower()
            if info_lower not in seen:
                seen.add(info_lower)
                unique_info.append(info)

        # Build answer
        if unique_info:
            answer = "**Based on the documents:**\n\n"
            for i, info in enumerate(unique_info[:10], 1):
                if not info.endswith('.'):
                    info += '.'
                answer += f"{i}. {info}\n"
            answer += f"\n**Primary Source:** {documents[0]['source']} (Page {documents[0]['page']})"
            return answer
        else:
            # Return document excerpt
            excerpt = documents[0]['text'][:500]
            return f"**Document Excerpt:**\n\n{excerpt}...\n\n**Source:** {documents[0]['source']} (Page {documents[0]['page']})"

    def _get_confidence_level(self, score: float) -> str:
        """Convert confidence score to level"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _format_sources(self, documents: List[Dict]) -> List[str]:
        """Format document sources"""
        sources = []
        seen_sources = set()

        for doc in documents[:5]:
            source_key = f"{doc['source']}:p{doc['page']}"
            if source_key not in seen_sources:
                source_str = f"{doc['source']} (Page {doc['page']})"
                if 'chunk_type' in doc:
                    source_str += f" [{doc['chunk_type']}]"
                sources.append(source_str)
                seen_sources.add(source_key)

        return sources

    def _detect_conflicts(self, documents: List[Dict]) -> List[str]:
        """Detect potential conflicts in retrieved documents"""
        conflicts = []

        # Simple conflict detection based on different values
        fire_ratings = set()
        dimensions = set()
        materials = set()

        for doc in documents:
            text = doc.get('text', '').upper()

            # Extract fire ratings
            fire_pattern = r'(\d{2,3})\s*MIN'
            fire_matches = re.findall(fire_pattern, text)
            fire_ratings.update(fire_matches)

            # Extract dimensions
            dim_pattern = r"(\d+['-]\s*\d+[\"']?)\s*[xX]\s*(\d+['-]\s*\d+[\"']?)"
            dim_matches = re.findall(dim_pattern, text)
            for match in dim_matches:
                dimensions.add(f"{match[0]} x {match[1]}")

            # Extract materials
            mat_pattern = r'\b(HM|WD|AL|GL|MFR)\b'
            mat_matches = re.findall(mat_pattern, text)
            materials.update(mat_matches)

        # Report conflicts
        if len(fire_ratings) > 1:
            conflicts.append(f"Multiple fire ratings found: {', '.join(sorted(fire_ratings))} MIN")

        if len(dimensions) > 2:
            conflicts.append(f"Multiple dimensions found: {', '.join(list(dimensions)[:3])}")

        if len(materials) > 2:
            conflicts.append(f"Multiple materials found: {', '.join(sorted(materials))}")

        return conflicts