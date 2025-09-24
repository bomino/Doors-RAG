"""
Entity Extraction and Search System
Extracts and indexes all types of entities from documents for comprehensive lookups
"""

import re
from typing import Dict, List, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts various entity types from document text"""

    def __init__(self):
        # Define patterns for different entity types
        self.patterns = {
            'door_numbers': [
                r'\b(\d{1,3}[A-Z]{1,2})\b',  # e.g., 148A, 627C
                r'DOOR\s+(\d{1,3}[A-Z]?)',    # DOOR 148A
                r'door\s+no\.?\s*(\d{1,3}[A-Z]?)',  # door no. 148A
            ],
            'room_names': [
                r'(\w+\s+ROOM)\b',  # e.g., CONFERENCE ROOM
                r'(RETAIL|OFFICE|STORAGE|VAULT|LOBBY|CORRIDOR|STAIR|VESTIBULE|ELECTRICAL|MECHANICAL|ELEVATOR)',
                r'ROOM\s+([A-Z0-9\-]+)',  # ROOM A-101
                r'(TRANSFORMER|GENERATOR|SWITCHGEAR|PANEL)\s*\w*',  # Technical rooms
            ],
            'areas': [
                r'LEVEL\s+(\d+|[A-Z]+)',  # LEVEL 1, LEVEL B
                r'FLOOR\s+(\d+|[A-Z]+)',  # FLOOR 2
                r'(NORTH|SOUTH|EAST|WEST)\s+(WING|TOWER|SECTION)',
                r'AREA\s+([A-Z0-9\-]+)',  # AREA A-1
                r'ZONE\s+([A-Z0-9\-]+)',  # ZONE 2
            ],
            'hardware_groups': [
                r'(?:HARDWARE\s+)?(?:GROUP\s+)?(\d{1,3}\.\d)',  # 86.0, 111.0
                r'HW\s*[-:]?\s*(\d{1,3}\.\d)',  # HW-86.0
                r'SET\s+(\d{1,3}\.\d)',  # SET 86.0
            ],
            'fire_ratings': [
                r'(\d{2,3})\s*MIN(?:UTE)?S?\b',  # 90 MIN, 120 MINUTES
                r'(\d{1,2})\s*H(?:OU)?R?\b',  # 2 HR, 1.5 HOUR
                r'(\d{2,3})\-MIN(?:UTE)?',  # 90-MINUTE
                r'FIRE\s*RATED?\s*(\d{2,3})',  # FIRE RATED 90
            ],
            'door_types': [
                r'\b(HM|WD|AL|GL|MFR)\b',  # Material codes
                r'\b(F\d|FG\d|SC\d|PR\d)\b',  # Door type codes
                r'(FLUSH|PANEL|GLAZED|SOLID|HOLLOW)',  # Door descriptions
                r'(SINGLE|DOUBLE|PAIR|SLIDING|OVERHEAD)',  # Door configurations
            ],
            'dimensions': [
                r"(\d+['-]\s*\d+[\"']?)\s*[xX]\s*(\d+['-]\s*\d+[\"']?)",  # 3'-0" x 7'-0"
                r'(\d+)\s*[xX]\s*(\d+)\s*(?:MM|CM|IN)?',  # 900 x 2100 mm
                r'WIDTH\s*[:=]\s*(\d+[^\s]*)',  # WIDTH: 36"
                r'HEIGHT\s*[:=]\s*(\d+[^\s]*)',  # HEIGHT: 84"
            ],
            'frame_types': [
                r'(HMF\d+)',  # HMF1, HMF2
                r'(KNOCK[- ]?DOWN|WELDED|MASONRY)',  # Frame construction
                r'FRAME\s+TYPE\s*[:=]?\s*([A-Z0-9]+)',  # FRAME TYPE A1
            ],
            'finishes': [
                r'(PAINT(?:ED)?|STAIN(?:ED)?|PRIME(?:[DR])?|GALV(?:ANIZED)?)',
                r'(ANODIZED|POWDER\s*COAT(?:ED)?|MILL\s*FINISH)',
                r'(WOOD\s*VENEER|PLASTIC\s*LAMINATE|VINYL)',
                r'FINISH\s*[:=]?\s*([A-Z\s]+)',
            ],
            'manufacturers': [
                r'(?:MFR|MANUFACTURER)\s*[:=]?\s*([A-Z][A-Z\s&]+)',
                r'BY\s+([A-Z][A-Z\s&]+(?:INC|LLC|CORP)?)',
                r'(MASONITE|OSHKOSH|VT\s*INDUSTRIES|SCHLAGE|STANLEY|VON\s*DUPRIN)',
            ],
            'standards': [
                r'(NFPA\s*\d+)',  # NFPA 80
                r'(ANSI[/\s]*[A-Z]*\d+[\.\d]*)',  # ANSI A117.1
                r'(UL\s*\d+[A-Z]?)',  # UL 10C
                r'(ASTM\s*[A-Z]*\d+)',  # ASTM E152
                r'(AWI\s*[A-Z\-\d]+)',  # AWI standards
            ]
        }

    def extract_all_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entity types from text"""
        entities = {}

        for entity_type, patterns in self.patterns.items():
            found_entities = set()

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

                # Handle different match types (single or tuple)
                for match in matches:
                    if isinstance(match, tuple):
                        # For dimension patterns that capture multiple groups
                        if entity_type == 'dimensions':
                            found_entities.add(f"{match[0]} x {match[1]}")
                        else:
                            found_entities.add(match[0])
                    else:
                        found_entities.add(match)

            # Clean and normalize entities
            cleaned_entities = self._clean_entities(list(found_entities), entity_type)
            if cleaned_entities:
                entities[entity_type] = cleaned_entities

        return entities

    def _clean_entities(self, entities: List[str], entity_type: str) -> List[str]:
        """Clean and normalize extracted entities"""
        cleaned = []

        for entity in entities:
            # Remove extra whitespace
            entity = ' '.join(entity.split())

            # Type-specific cleaning
            if entity_type == 'door_numbers':
                # Ensure consistent format (e.g., 148A not 148a)
                entity = entity.upper()

            elif entity_type == 'room_names':
                # Capitalize room names
                entity = entity.upper()

            elif entity_type == 'fire_ratings':
                # Normalize fire ratings
                entity = entity.upper()
                # Convert hours to minutes if needed
                hour_match = re.match(r'(\d+)\s*H(?:OU)?R?', entity)
                if hour_match:
                    hours = int(hour_match.group(1))
                    entity = f"{hours * 60} MIN"

            elif entity_type == 'dimensions':
                # Keep dimension format consistent
                entity = entity.replace('"', '"').replace("'", "'")

            elif entity_type == 'hardware_groups':
                # Ensure decimal point
                if '.' not in entity and entity.isdigit():
                    entity = f"{entity}.0"

            # Only add non-empty, cleaned entities
            if entity and len(entity) > 1:
                cleaned.append(entity)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        return unique

    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities (e.g., door 148A is in RETAIL room)"""
        relationships = []

        # Find door-room relationships
        door_room_pattern = r'(\d{1,3}[A-Z]?)\s+(\w+(?:\s+\w+)?)\s+[FG]?\d'
        matches = re.findall(door_room_pattern, text, re.IGNORECASE)

        for match in matches:
            door, room = match
            relationships.append({
                'type': 'door_location',
                'door': door.upper(),
                'room': room.upper(),
                'confidence': 0.9
            })

        # Find door-hardware relationships
        door_hw_pattern = r'(\d{1,3}[A-Z]?)[^0-9]*(\d{1,3}\.\d)'
        matches = re.findall(door_hw_pattern, text)

        for match in matches:
            door, hw_group = match
            relationships.append({
                'type': 'door_hardware',
                'door': door.upper(),
                'hardware_group': hw_group,
                'confidence': 0.85
            })

        return relationships


class SmartQueryParser:
    """Parses queries to identify what entities the user is looking for"""

    def __init__(self):
        self.query_patterns = {
            'door_lookup': [
                r'door\s+(\d{1,3}[A-Z]?)',
                r'specifications?\s+(?:for\s+)?(?:door\s+)?(\d{1,3}[A-Z]?)',
                r'what\s+(?:is|are)\s+.*?(\d{1,3}[A-Z]?)',
                r'(\d{1,3}[A-Z]?)\s+door',
            ],
            'room_lookup': [
                r'(?:room|space)\s+([A-Z0-9\-]+)',
                r'(RETAIL|OFFICE|STORAGE|VAULT|LOBBY|CORRIDOR)',
                r'in\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:room|area)',
            ],
            'hardware_lookup': [
                r'hardware\s+(?:group\s+)?(\d{1,3}\.?\d?)',
                r'(?:set|group)\s+(\d{1,3}\.?\d?)',
                r'what\s+hardware.*?(\d{1,3}\.?\d?)',
            ],
            'fire_rating_lookup': [
                r'(\d{2,3})\s*(?:minute|min|hour|hr)',
                r'fire\s+rat\w*\s+(?:of\s+)?(\d{2,3})',
                r'doors?\s+with\s+(\d{2,3})\s*(?:minute|min)',
            ],
            'area_lookup': [
                r'(?:on|at|in)\s+(?:level|floor)\s+([A-Z0-9]+)',
                r'(north|south|east|west)\s+(?:wing|tower|section)',
                r'area\s+([A-Z0-9\-]+)',
            ],
            'general_spec': [
                r'(?:specification|spec|requirement|standard)',
                r'(?:what|which|list|show|find|get)',
            ]
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query to identify entity lookups"""
        query_lower = query.lower()
        parsed = {
            'original_query': query,
            'query_type': 'general',
            'entities': {},
            'filters': {},
            'intent': 'search'
        }

        # Check each query pattern type
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    parsed['query_type'] = query_type

                    # Extract specific entity
                    if query_type == 'door_lookup':
                        parsed['entities']['door_number'] = matches[0].upper()
                        parsed['intent'] = 'specific_lookup'
                    elif query_type == 'room_lookup':
                        parsed['entities']['room'] = matches[0].upper()
                        parsed['intent'] = 'location_search'
                    elif query_type == 'hardware_lookup':
                        parsed['entities']['hardware_group'] = matches[0]
                        parsed['intent'] = 'hardware_search'
                    elif query_type == 'fire_rating_lookup':
                        rating = matches[0]
                        # Normalize to minutes
                        if 'hour' in query_lower or 'hr' in query_lower:
                            rating = str(int(rating) * 60)
                        parsed['entities']['fire_rating'] = f"{rating} MIN"
                        parsed['intent'] = 'filter_search'
                    elif query_type == 'area_lookup':
                        parsed['entities']['area'] = matches[0].upper()
                        parsed['intent'] = 'location_search'

                    break

            if parsed['query_type'] != 'general':
                break

        # Determine if user wants a list
        if any(word in query_lower for word in ['list', 'all', 'show all', 'every']):
            parsed['intent'] = 'list_all'

        # Check for comparison queries
        if any(word in query_lower for word in ['compare', 'difference', 'between']):
            parsed['intent'] = 'comparison'

        return parsed


class EntityAwareRetriever:
    """Retriever that uses entity extraction for better search"""

    def __init__(self, qdrant_client, embedding_model, entity_extractor, query_parser):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.entity_extractor = entity_extractor
        self.query_parser = query_parser
        self.collection_name = "door_specifications"

    def retrieve_by_entity(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve documents based on entity matching"""

        # Parse the query to understand what entities to look for
        parsed_query = self.query_parser.parse_query(query)
        logger.info(f"Parsed query: {parsed_query}")

        # Build search filters based on entities
        filters = []
        boost_terms = []

        # Add entity-specific filters
        if 'door_number' in parsed_query['entities']:
            door_num = parsed_query['entities']['door_number']
            filters.append({
                'key': 'text',
                'match': {'text': door_num}
            })
            boost_terms.append(door_num)

        if 'room' in parsed_query['entities']:
            room = parsed_query['entities']['room']
            filters.append({
                'key': 'text',
                'match': {'text': room}
            })
            boost_terms.append(room)

        if 'hardware_group' in parsed_query['entities']:
            hw_group = parsed_query['entities']['hardware_group']
            filters.append({
                'key': 'text',
                'match': {'text': hw_group}
            })
            boost_terms.append(hw_group)

        if 'fire_rating' in parsed_query['entities']:
            fire_rating = parsed_query['entities']['fire_rating']
            filters.append({
                'key': 'metadata.fire_rating',
                'match': {'any': [fire_rating]}
            })
            boost_terms.append(fire_rating)

        # Create query vector
        query_vector = self.embedding_model.encode([query])[0].tolist()

        # Search with filters
        qdrant_filter = {'should': filters} if filters else None

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k * 3,  # Get more candidates for filtering
            query_filter=qdrant_filter,
            with_payload=True
        )

        # Post-process and rerank based on entity matches
        documents = []
        for result in search_results:
            doc = {
                'id': result.payload.get('id', ''),
                'text': result.payload.get('text', ''),
                'score': float(result.score),
                'metadata': result.payload.get('metadata', {}),
                'source': result.payload.get('pdf_name', ''),
                'page': result.payload.get('page_num', 0),
                'chunk_type': result.payload.get('chunk_type', '')
            }

            # Extract entities from this document
            doc_entities = self.entity_extractor.extract_all_entities(doc['text'])
            doc['extracted_entities'] = doc_entities

            # Boost score based on entity matches
            entity_boost = 0
            for term in boost_terms:
                if term in doc['text'].upper():
                    entity_boost += 0.3

            # Check for exact entity matches
            if parsed_query['intent'] == 'specific_lookup':
                for entity_type, entity_values in parsed_query['entities'].items():
                    if entity_type in ['door_number', 'room', 'hardware_group']:
                        if entity_values in doc['text'].upper():
                            entity_boost += 0.5

            doc['score'] = doc['score'] + entity_boost
            documents.append(doc)

        # Sort by score and return top k
        documents.sort(key=lambda x: x['score'], reverse=True)

        return documents[:top_k]