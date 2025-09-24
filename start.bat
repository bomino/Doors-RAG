@echo off
REM Door Specifications RAG System - Quick Start Script
REM Author: MLawali@versatexmsp.com
REM © 2025 All rights reserved

echo.
echo  Door Specifications RAG System
echo ================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo  Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo  Docker is running
echo.

REM Check for .env file
if not exist .env (
    echo  Warning: .env file not found. Creating from template...
    copy .env.example .env
    echo  Please edit .env and add your API keys
    pause
    exit /b 1
)

REM Start services
echo  Starting services...
docker-compose up -d

REM Wait for services
echo  Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo.
echo  Checking service status...
docker-compose ps

REM Ask about PDF processing
echo.
set /p process_pdfs="Do you need to process PDF documents? (y/n): "
if /i "%process_pdfs%"=="y" (
    echo  Processing PDFs...
    docker exec doors-backend python backend/scripts/process_pdfs.py
)

echo.
echo  System is ready!
echo.
echo  Access the application at:
echo    Frontend: http://localhost:8502
echo    Backend API: http://localhost:8000/docs
echo    Qdrant: http://localhost:6333/dashboard
echo.
echo  To view logs: docker-compose logs -f [service-name]
echo  To stop: docker-compose down
echo.
echo  Developed by MLawali@versatexmsp.com - 2025
echo.
pause