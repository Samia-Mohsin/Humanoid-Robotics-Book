"""
Minimal Vercel entry point that avoids all complex imports to prevent crashes
"""
import os

# Set environment variable to identify Vercel environment early
os.environ["VERCEL_ENV"] = "true"

# Create a minimal FastAPI app without any complex imports
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Physical AI & Humanoid Robotics Educational Platform API - Minimal",
        description="Minimal API server for Vercel deployment",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define all routes directly without importing external modules
    @app.get("/")
    async def root():
        return {
            "message": "Physical AI & Humanoid Robotics Educational Platform API is running",
            "api_docs": "/docs",
            "status": "healthy",
            "environment": "vercel-minimal"
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "Physical AI & Humanoid Robotics API", "environment": "vercel-minimal"}

    # API root endpoint
    @app.get("/api")
    async def api_root():
        return {"message": "Physical AI & Humanoid Robotics Educational Platform API", "environment": "vercel-minimal"}

    # Basic query endpoint
    @app.post("/api/query")
    async def query_endpoint():
        return {
            "response": "Query service is available",
            "sources": [],
            "selected_text_used": False,
            "message": "Service running in minimal mode - full functionality requires proper backend setup"
        }

    # Basic auth endpoint
    @app.post("/api/auth/login")
    async def login():
        return {"message": "Login endpoint available", "status": "minimal"}

    # Basic personalize endpoint
    @app.get("/api/personalize/preferences")
    async def get_preferences():
        return {"message": "Preferences endpoint available", "status": "minimal"}

    # Basic translate endpoint
    @app.post("/api/translate")
    async def translate_text():
        return {"message": "Translate endpoint available", "status": "minimal"}

    # Basic quizzes endpoint
    @app.get("/api/quizzes/questions")
    async def get_questions():
        return {"message": "Quizzes endpoint available", "status": "minimal"}

    # Fallback for any other API routes
    @app.get("/api/{path:path}")
    @app.post("/api/{path:path}")
    @app.put("/api/{path:path}")
    @app.delete("/api/{path:path}")
    async def api_fallback(path: str):
        return {
            "message": f"API endpoint /api/{path} is available in minimal mode",
            "status": "minimal"
        }

    # Catch-all for non-API routes
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        return {
            "message": "API server running",
            "requested_path": full_path,
            "available_endpoints": ["/", "/health", "/api", "/api/query", "/api/auth/login", "/api/personalize/preferences", "/api/translate", "/api/quizzes/questions"]
        }

    # This is the entry point for Vercel
    app_instance = app

except Exception as e:
    # If FastAPI import fails, create a minimal WSGI app as ultimate fallback
    def app_instance(environ, start_response):
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        response_data = {
            "status": "error",
            "message": "Application failed to initialize properly",
            "error": str(e),
            "environment": "vercel-minimal-fallback"
        }
        return [json.dumps(response_data).encode('utf-8')]