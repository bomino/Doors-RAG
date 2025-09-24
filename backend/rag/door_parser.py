"""
Specialized Door Schedule Parser
Parses door schedule tables to extract specific door information
"""

import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DoorScheduleParser:
    """Parses door schedule information from text"""

    def __init__(self):
        # Define the door schedule column structure based on the actual documents
        self.schedule_columns = [
            'door_number',
            'location/room',
            'door_type',
            'width',
            'height',
            'thickness',
            'material',
            'finish',
            'hardware_group',
            'frame_type',
            'detail_refs',
            'fire_rating'
        ]

    def parse_door_info(self, text: str, target_door: str) -> Optional[Dict[str, Any]]:
        """Extract specific door information from schedule text"""

        # Clean the target door number
        target_door = target_door.upper().strip()

        # Look for the door number in the text
        if target_door not in text.upper():
            return None

        # Try to find structured data around the door number
        door_info = {}

        # Try multiple parsing strategies

        # Strategy 1: Parse pipe-separated format (most structured)
        pipe_pattern = rf'{target_door}\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)'
        pipe_match = re.search(pipe_pattern, text, re.IGNORECASE)

        if pipe_match:
            door_info = {
                'door_number': target_door,
                'location': pipe_match.group(1).strip(),
                'door_type': pipe_match.group(2).strip(),
                'width': pipe_match.group(3).strip(),
                'height': pipe_match.group(4).strip(),
                'thickness': pipe_match.group(5).strip(),
                'material': pipe_match.group(6).strip(),
                'finish': pipe_match.group(7).strip(),
                'hardware_group': pipe_match.group(8).strip(),
                'frame_type': pipe_match.group(9).strip()
            }
        else:
            # Strategy 2: Enhanced space-separated format with better location capture
            patterns = [
                # Pattern for space-separated with multi-word location
                rf'{target_door}\s+([\w\-]+(?:\s+\w+)*?)\s+([FG]\d+|SC\d+|PR\d+)\s+(\d+[\'"]?\s*-\s*\d+["\']?)\s+(\d+[\'"]?\s*-\s*\d+["\']?)\s+(\d+[\'"]?\s*-\s*\d+[^\'"\s]*[\'"]?)\s+(HM|WD|AL|GL|MFR)\s+(\w+)\s+([\d.]+)\s+([A-Z0-9]+)',

                # Simpler pattern for partial matches
                rf'{target_door}\s+([\w\-]+(?:\s+\w+)*?)\s+([FG]\d+)\s+(\d+[\'"]?\s*-\s*\d+["\']?)\s+(\d+[\'"]?\s*-\s*\d+["\']?)',

                # Most flexible pattern
                rf'{target_door}[^\n]*',
            ]

            for pattern in patterns:
                matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    if matches.lastindex and matches.lastindex >= 8:
                        # Full match with all fields
                        door_info = {
                            'door_number': target_door,
                            'location': matches.group(1),
                            'door_type': matches.group(2),
                            'width': matches.group(3),
                            'height': matches.group(4),
                            'thickness': matches.group(5),
                            'material': matches.group(6),
                            'finish': matches.group(7),
                            'hardware_group': matches.group(8),
                            'frame_type': matches.group(9) if matches.lastindex >= 9 else None
                        }
                        break
                    elif matches.lastindex and matches.lastindex >= 4:
                        # Partial match with basic fields
                        door_info = {
                            'door_number': target_door,
                            'location': matches.group(1),
                            'door_type': matches.group(2),
                            'width': matches.group(3),
                            'height': matches.group(4)
                        }
                        # Continue parsing the rest of the line
                        line = matches.group(0)
                        additional_info = self._parse_door_line(line, target_door)
                        door_info.update(additional_info)
                        break
                    else:
                        # Fallback to line parsing
                        line = matches.group(0)
                        door_info = self._parse_door_line(line, target_door)
                        break

        # Look for fire rating anywhere near the door
        fire_patterns = [
            rf'{target_door}[^\n]*?(\d{{2,3}})\s*MIN',
            rf'{target_door}[^\n]*?(\d{{2,3}})\s*MINUTE',
            rf'(\d{{2,3}})\s*MIN[^\n]*?{target_door}'
        ]

        for fire_pattern in fire_patterns:
            fire_match = re.search(fire_pattern, text, re.IGNORECASE)
            if fire_match:
                door_info['fire_rating'] = f"{fire_match.group(1)} MIN"
                break

        # Clean up finish codes
        if 'finish' in door_info:
            finish_map = {
                'PT': 'PAINT',
                'ST': 'STAIN',
                'PR': 'PRIME',
                'GALV': 'GALVANIZED',
                'FACT': 'FACTORY',
                'AN': 'ANODIZED'
            }
            finish_code = door_info['finish'].upper()
            door_info['finish'] = finish_map.get(finish_code, finish_code)

        return door_info if door_info else None

    def _parse_door_line(self, line: str, door_number: str) -> Dict[str, Any]:
        """Parse a door schedule line with flexible extraction"""

        # Split the line into tokens
        tokens = line.split()
        door_info = {'door_number': door_number}

        # Track position for context-aware parsing
        location_found = False
        type_found = False

        for i, token in enumerate(tokens):
            token_upper = token.upper()

            # Skip the door number itself
            if token_upper == door_number.upper():
                continue

            # Location/Room names (capture multi-word locations)
            if not location_found and i < len(tokens) - 1:
                # Check for known location keywords
                location_keywords = ['RETAIL', 'OFFICE', 'STORAGE', 'VAULT', 'LOBBY',
                                   'CORRIDOR', 'STAIR', 'VESTIBULE', 'ELECTRICAL',
                                   'MECHANICAL', 'TRANSFORMER', 'VALET', 'KITCHEN',
                                   'DINING', 'CONFERENCE', 'RESTROOM', 'RR', 'JAN',
                                   'BUFFET', 'PDR', '3-MEAL']

                if any(kw in token_upper for kw in location_keywords):
                    # Check if next token is also part of location
                    location = token_upper
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1].upper()
                        if next_token in ['ROOM', 'KITCHEN', 'DINING', 'HALL', 'AREA']:
                            location += ' ' + next_token
                    door_info['location'] = location
                    location_found = True

            # Door type (F1, FG2, etc.)
            elif not type_found and re.match(r'^[FG]\d+$|^SC\d+$|^PR\d+$', token_upper):
                door_info['door_type'] = token_upper
                type_found = True

            # Dimensions (3'-0", 7'-0")
            elif re.match(r"^\d+['-]\s*\d+[\"']?$", token):
                if 'width' not in door_info:
                    door_info['width'] = token
                elif 'height' not in door_info:
                    door_info['height'] = token
                elif 'thickness' not in door_info:
                    door_info['thickness'] = token

            # Material codes
            elif token_upper in ['HM', 'WD', 'AL', 'GL', 'MFR']:
                if 'material' not in door_info:
                    door_info['material'] = token_upper

            # Finish types
            elif token_upper in ['PT', 'ST', 'PR', 'PAINT', 'STAIN', 'PRIME', 'GALV', 'FACT', 'AN']:
                if 'finish' not in door_info:
                    door_info['finish'] = token_upper

            # Hardware group (decimal numbers)
            elif re.match(r'^\d+\.\d+$', token):
                if 'hardware_group' not in door_info:
                    door_info['hardware_group'] = token

            # Frame type
            elif re.match(r'^HMF\d+$|^WDF\d+$|^ALF\d+$', token_upper):
                door_info['frame_type'] = token_upper

            # Fire rating
            elif 'MIN' in token_upper or 'MINUTE' in token_upper:
                # Look for preceding number
                if i > 0 and re.match(r'^\d+$', tokens[i-1]):
                    door_info['fire_rating'] = f"{tokens[i-1]} MIN"

        return door_info

    def format_door_specifications(self, door_info: Dict[str, Any]) -> str:
        """Format door information into a readable specification"""

        if not door_info:
            return "No information found for this door."

        lines = [f"**Door {door_info.get('door_number', 'Unknown')} Specifications:**\n"]

        # Add each field if present
        if 'location' in door_info:
            lines.append(f"**Location:** {door_info['location']}")

        if 'door_type' in door_info:
            door_type = door_info['door_type']
            # Add description
            type_desc = {
                'F1': 'Flush door, type 1',
                'F2': 'Flush door, type 2',
                'F3': 'Flush door, type 3',
                'F4': 'Flush door, type 4',
                'F5': 'Flush door, type 5',
                'FG1': 'Flush glazed door, type 1',
                'FG2': 'Flush glazed door, type 2',
                'SC1': 'Solid core door, type 1',
                'PR1': 'Pair door, type 1'
            }
            desc = type_desc.get(door_type.upper(), door_type)
            lines.append(f"**Door Type:** {door_type} ({desc})")

        if 'width' in door_info and 'height' in door_info:
            lines.append(f"**Dimensions:** {door_info['width']} wide x {door_info['height']} high")
        elif 'width' in door_info:
            lines.append(f"**Width:** {door_info['width']}")
        elif 'height' in door_info:
            lines.append(f"**Height:** {door_info['height']}")

        if 'thickness' in door_info:
            lines.append(f"**Thickness:** {door_info['thickness']}")

        if 'material' in door_info:
            mat = door_info['material']
            mat_desc = {
                'HM': 'Hollow Metal',
                'WD': 'Wood',
                'AL': 'Aluminum',
                'GL': 'Glass',
                'MFR': 'Manufacturer Standard'
            }
            desc = mat_desc.get(mat.upper(), mat)
            lines.append(f"**Material:** {mat} ({desc})")

        if 'finish' in door_info:
            finish = door_info['finish']
            finish_desc = {
                'PAINT': 'Paint finish on both sides',
                'PT': 'Paint finish on both sides',
                'STAIN': 'Stained finish',
                'ST': 'Stained finish',
                'PRIME': 'Primer finish',
                'PR': 'Primer finish',
                'GALV': 'Galvanized finish',
                'GALVANIZED': 'Galvanized finish',
                'ANODIZED': 'Anodized finish',
                'AN': 'Anodized finish',
                'FACT': 'Factory finish'
            }
            desc = finish_desc.get(finish.upper(), finish)
            lines.append(f"**Finish:** {desc}")

        if 'fire_rating' in door_info:
            rating = door_info['fire_rating']
            lines.append(f"**Fire Rating:** {rating} ({rating.replace(' MIN', '-minute')} fire rating)")

        if 'hardware_group' in door_info:
            lines.append(f"**Hardware Group:** {door_info['hardware_group']}")

        if 'frame_type' in door_info:
            frame = door_info['frame_type']
            if frame.startswith('HMF'):
                lines.append(f"**Frame Type:** {frame} (Hollow Metal Frame type {frame[-1]})")
            elif frame.startswith('WDF'):
                lines.append(f"**Frame Type:** {frame} (Wood Frame type {frame[-1]})")
            elif frame.startswith('ALF'):
                lines.append(f"**Frame Type:** {frame} (Aluminum Frame type {frame[-1]})")
            else:
                lines.append(f"**Frame Type:** {frame}")

        # Add any additional fields
        for key, value in door_info.items():
            if key not in ['door_number', 'location', 'door_type', 'width', 'height',
                          'thickness', 'material', 'finish', 'fire_rating', 'hardware_group',
                          'frame_type'] and value:
                # Format the key nicely
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"**{formatted_key}:** {value}")

        return '\n'.join(lines)