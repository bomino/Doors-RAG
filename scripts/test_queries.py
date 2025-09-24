#!/usr/bin/env python3
"""Test queries for the RAG system"""
import requests
import json
from typing import List, Dict

# Test queries
TEST_QUERIES = [
    "What are the specifications for door 148A?",
    "What is the warranty period for solid-core interior wood doors?",
    "How many hinges are required on a 7'-0" interior door?",
    "What fire rating standards apply to wood doors?",
    "List all 90-minute fire rated doors",
    "What are the hardware requirements for door 627C?",
    "Are there any conflicts in the specifications?"
]

def test_query(query: str, api_url: str = "http://localhost:8000/api/v1") -> Dict:
    """Test a single query"""
    response = requests.post(
        f"{api_url}/query",
        json={"query": query}
    )
    return response.json() if response.status_code == 200 else None

def main():
    """Run test queries"""
    print("Testing RAG System Queries\n")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        result = test_query(query)
        if result:
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Confidence: {result['confidence']} ({result['confidence_score']:.1f}%)")
            print(f"Sources: {len(result.get('sources', []))} documents")
        else:
            print("Failed to get response")

    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    main()
