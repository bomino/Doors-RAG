"""Pydantic models for API"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class Document(BaseModel):
    content: str
    metadata: Dict
    confidence: Optional[float] = None

class ChunkMetadata(BaseModel):
    doc_title: str
    division_section: Optional[str] = None
    page: Optional[int] = None
    door_no: Optional[str] = None
    hardware_set: Optional[str] = None
    fire_rating: Optional[str] = None
