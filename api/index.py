# Vercel entry point for the unified server
import sys
import os

# Add the backend directory to the Python path to ensure imports work correctly
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_dir)

try:
    # Import the app from the unified server
    from unified_server_final import app

    # This is the entry point for Vercel
    app_instance = app
except ImportError as e:
    print(f"Import error: {e}")
    # Provide a fallback app in case of import issues
    from fastapi import FastAPI
    app_instance = FastAPI()

    @app_instance.get("/")
    @app_instance.get("/health")
    def health_check():
        return {"status": "error", "message": f"Failed to load application: {str(e)}"}
except Exception as e:
    print(f"Unexpected error during initialization: {e}")
    # Provide a fallback app in case of other issues
    from fastapi import FastAPI
    app_instance = FastAPI()

    @app_instance.get("/")
    @app_instance.get("/health")
    def health_check():
        return {"status": "error", "message": f"Application failed to start: {str(e)}"}