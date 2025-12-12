"""
Vercel-safe entry point that avoids all problematic dependencies
"""
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

# Capture any import errors for debugging
error_buffer = io.StringIO()

try:
    # Try to import the Vercel-safe app with proper error handling
    backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
    sys.path.insert(0, backend_dir)

    # Temporarily redirect stderr to capture import errors
    with redirect_stderr(error_buffer):
        from vercel_safe_server import app as main_app
        app = main_app
except Exception as e:
    # If main import fails, create a minimal working app
    print(f"Main app import failed: {e}")
    print(f"Import errors captured: {error_buffer.getvalue()}")

    # Create a minimal FastAPI app that will work in Vercel
    from fastapi import FastAPI

    app = FastAPI(
        title="Physical AI & Humanoid Robotics Educational Platform API - Vercel Fallback",
        description="Vercel-compatible fallback API server",
        version="1.0.0"
    )

    @app.get("/")
    @app.get("/health")
    def health_check():
        return {
            "status": "error",
            "message": "Main application failed to load",
            "error": str(e),
            "debug_info": error_buffer.getvalue()
        }

    @app.get("/api")
    def api_root():
        return {
            "message": "API server is running in fallback mode",
            "status": "fallback"
        }

# This is the entry point for Vercel
app_instance = app