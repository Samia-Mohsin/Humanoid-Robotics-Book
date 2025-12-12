"""
Vercel-safe unified server that uses Vercel-compatible routers
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Check if running in Vercel environment early
is_vercel = os.environ.get("VERCEL", False)

# Create the main FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Educational Platform API - Vercel Safe",
    description="Vercel-safe server for the educational platform with RAG chatbot, authentication, personalization, and translation features",
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

# Import and include the Vercel-safe API routers
try:
    from app.routers.vercel_query import router as query_router
    from app.routers.vercel_auth import router as auth_router
    from app.routers.vercel_personalize import router as personalize_router
    from app.routers.vercel_translate import router as translate_router
    from app.routers.vercel_quizzes import router as quizzes_router

    # Include the Vercel-safe API routers
    app.include_router(query_router, prefix="/api", tags=["query"])
    app.include_router(auth_router, prefix="/api", tags=["auth"])
    app.include_router(personalize_router, prefix="/api", tags=["personalize"])
    app.include_router(translate_router, prefix="/api", tags=["translate"])
    app.include_router(quizzes_router, prefix="/api", tags=["quizzes"])

except ImportError as e:
    print(f"Vercel-safe router import error: {e}")
    # If the Vercel-safe routers fail to import, create fallback routes
    @app.get("/api/query")
    @app.post("/api/query")
    async def query_fallback():
        return {"status": "error", "message": "Query service not available"}

    @app.get("/api/auth")
    @app.post("/api/auth")
    async def auth_fallback():
        return {"status": "error", "message": "Auth service not available"}

    @app.get("/api/personalize")
    @app.post("/api/personalize")
    async def personalize_fallback():
        return {"status": "error", "message": "Personalize service not available"}

    @app.get("/api/translate")
    @app.post("/api/translate")
    async def translate_fallback():
        return {"status": "error", "message": "Translate service not available"}

    @app.get("/api/quizzes")
    @app.post("/api/quizzes")
    async def quizzes_fallback():
        return {"status": "error", "message": "Quizzes service not available"}

@app.get("/")
async def root():
    # In Vercel, return a simple message or redirect to API docs
    return {"message": "Physical AI & Humanoid Robotics Educational Platform API is running", "api_docs": "/docs", "status": "healthy", "environment": "vercel"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Physical AI & Humanoid Robotics API", "environment": "vercel"}

# API root endpoint
@app.get("/api")
async def api_root():
    return {"message": "Physical AI & Humanoid Robotics Educational Platform API", "environment": "vercel"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)