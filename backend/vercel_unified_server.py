"""
Vercel-compatible unified server that avoids problematic imports
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from pathlib import Path
import os
import uvicorn

# Check if running in Vercel environment early
is_vercel = os.environ.get("VERCEL", False)

# Create the main FastAPI app with error handling for router imports
app = FastAPI(
    title="Physical AI & Humanoid Robotics Educational Platform API",
    description="Unified server for the educational platform with RAG chatbot, authentication, personalization, and translation features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers for client-side access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Import and include the API routers with comprehensive error handling
try:
    from app.routers import query, auth, personalize, translate, quizzes

    # Include the existing API routers - only if they have a router attribute
    try:
        app.include_router(query.router, prefix="/api", tags=["query"])
    except AttributeError:
        # If query is a router instance itself, add it directly
        app.include_router(query, prefix="/api", tags=["query"])

    try:
        app.include_router(auth.router, prefix="/api", tags=["auth"])
    except AttributeError:
        app.include_router(auth, prefix="/api", tags=["auth"])

    try:
        app.include_router(personalize.router, prefix="/api", tags=["personalize"])
    except AttributeError:
        app.include_router(personalize, prefix="/api", tags=["personalize"])

    try:
        app.include_router(translate.router, prefix="/api", tags=["translate"])
    except AttributeError:
        app.include_router(translate, prefix="/api", tags=["translate"])

    try:
        app.include_router(quizzes.router, prefix="/api", tags=["quizzes"])
    except AttributeError:
        app.include_router(quizzes, prefix="/api", tags=["quizzes"])

except ImportError as e:
    print(f"Router import error (expected in Vercel): {e}")
    print("Creating fallback API routes...")

    # Create fallback routes in case the actual routers can't be imported
    @app.get("/api/query")
    @app.post("/api/query")
    async def query_fallback():
        return {"status": "error", "message": "Query service not available in this environment"}

    @app.get("/api/auth")
    @app.post("/api/auth")
    async def auth_fallback():
        return {"status": "error", "message": "Auth service not available in this environment"}

    @app.get("/api/personalize")
    @app.post("/api/personalize")
    async def personalize_fallback():
        return {"status": "error", "message": "Personalize service not available in this environment"}

    @app.get("/api/translate")
    @app.post("/api/translate")
    async def translate_fallback():
        return {"status": "error", "message": "Translate service not available in this environment"}

    @app.get("/api/quizzes")
    @app.post("/api/quizzes")
    async def quizzes_fallback():
        return {"status": "error", "message": "Quizzes service not available in this environment"}

# Define the frontend build directory
frontend_build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")

if not is_vercel:
    # Mount static assets if the build directory exists (for local deployment)
    if os.path.exists(frontend_build_dir):
        # Mount the assets directory (CSS, JS, images)
        assets_path = os.path.join(frontend_build_dir, "assets")
        if os.path.exists(assets_path):
            app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

        # Mount other static directories that might exist
        for subdir in ["img", "js", "css", "static"]:
            subdir_path = os.path.join(frontend_build_dir, subdir)
            if os.path.exists(subdir_path):
                app.mount(f"/{subdir}", StaticFiles(directory=subdir_path), name=subdir)
    else:
        print("Warning: Frontend build directory does not exist. Run 'npm run build' in the frontend directory to build the frontend.")
else:
    print("Running in Vercel environment - skipping static file mounts")

@app.get("/")
async def root():
    if not is_vercel and os.path.exists(frontend_build_dir):
        return FileResponse(os.path.join(frontend_build_dir, "index.html"))
    else:
        # In Vercel, return a simple message or redirect to API docs
        return {"message": "Physical AI & Humanoid Robotics Educational Platform API is running", "api_docs": "/docs", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Physical AI & Humanoid Robotics API"}

# API root endpoint
@app.get("/api")
async def api_root():
    return {"message": "Physical AI & Humanoid Robotics Educational Platform API"}

# Catch-all route for client-side routing (excluding API routes)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # If the path starts with /api/, return a more specific message
    if full_path.startswith("api/"):
        return JSONResponse(
            status_code=404,
            content={"detail": "API route not found or service unavailable"}
        )

    # In Vercel environment, don't try to serve frontend files
    if is_vercel:
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not available in this environment. See API documentation at /docs"}
        )

    # Check if it's a static file in the build directory
    file_path = os.path.join(frontend_build_dir, full_path)

    # Handle specific static asset paths that might exist
    if os.path.exists(file_path) and os.path.isfile(file_path):
        # For known file types, return with appropriate media type
        extension = os.path.splitext(full_path)[1].lower()
        if extension in ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.json', '.txt', '.xml']:
            return FileResponse(file_path)
        else:
            return FileResponse(file_path)

    # For all other paths (that don't correspond to static files), serve index.html
    # This enables client-side routing for the React/Docusaurus app
    if os.path.exists(frontend_build_dir):
        return FileResponse(os.path.join(frontend_build_dir, "index.html"))
    else:
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)