"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint():
    """Test query endpoint"""
    response = client.post(
        "/api/v1/query",
        json={"query": "Test query"}
    )
    assert response.status_code == 200
