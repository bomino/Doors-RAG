#!/bin/bash

# Door Specifications RAG System - Quick Start Script
# Author: MLawali@versatexmsp.com
# © 2025 All rights reserved

echo "🚪 Door Specifications RAG System"
echo "================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your API keys"
    exit 1
fi

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service status..."
docker-compose ps

# Process PDFs if needed
echo ""
read -p "Do you need to process PDF documents? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📄 Processing PDFs..."
    docker exec doors-backend python backend/scripts/process_pdfs.py
fi

echo ""
echo "✅ System is ready!"
echo ""
echo "🌐 Access the application at:"
echo "   Frontend: http://localhost:8502"
echo "   Backend API: http://localhost:8000/docs"
echo "   Qdrant: http://localhost:6333/dashboard"
echo ""
echo "📊 To view logs: docker-compose logs -f [service-name]"
echo "🛑 To stop: docker-compose down"
echo ""
echo "Developed by MLawali@versatexmsp.com | © 2025"