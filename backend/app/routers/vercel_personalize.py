from fastapi import APIRouter
import os

# Check if running in Vercel environment
is_vercel = os.environ.get("VERCEL", False)

router = APIRouter(
    prefix="/personalize",
    tags=["personalize"],
    responses={404: {"description": "Not found"}},
)

# Create a minimal working personalize router for Vercel
@router.get("/")
@router.post("/")
async def personalize_endpoint():
    return {
        "status": "available",
        "service": "Personalize service in Vercel"
    }

@router.get("/health")
async def personalize_health():
    return {"status": "healthy", "service": "Personalize Service (Vercel)"}

@router.get("/preferences")
async def get_preferences():
    return {"message": "Preferences endpoint - requires proper implementation in Vercel"}

@router.post("/preferences")
async def set_preferences():
    return {"message": "Set preferences endpoint - requires proper implementation in Vercel"}