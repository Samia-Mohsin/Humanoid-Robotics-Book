from fastapi import APIRouter
import os

# Check if running in Vercel environment
is_vercel = os.environ.get("VERCEL", False)

router = APIRouter(
    prefix="/translate",
    tags=["translate"],
    responses={404: {"description": "Not found"}},
)

# Create a minimal working translate router for Vercel
@router.get("/")
@router.post("/")
async def translate_endpoint():
    return {
        "status": "available",
        "service": "Translate service in Vercel"
    }

@router.get("/health")
async def translate_health():
    return {"status": "healthy", "service": "Translate Service (Vercel)"}

@router.post("/translate")
async def translate_text():
    return {"message": "Translate endpoint - requires proper implementation in Vercel"}