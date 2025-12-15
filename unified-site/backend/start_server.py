#!/usr/bin/env python3
"""
Start script for the Physical AI & Humanoid Robotics Backend API
"""
import uvicorn
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import settings

def main():
    print("🚀 Starting Physical AI & Humanoid Robotics Backend API...")
    print(f"📚 API Documentation available at: http://localhost:{settings.PORT}/docs")
    print(f"📡 Health check at: http://localhost:{settings.PORT}/health")
    print(f"🔒 Admin endpoints require authentication")
    print("-" * 50)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()