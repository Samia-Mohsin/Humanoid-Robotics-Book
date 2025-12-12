from fastapi import APIRouter
import os

# Check if running in Vercel environment
is_vercel = os.environ.get("VERCEL", False)

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

# Create a minimal working auth router for Vercel
@router.get("/")
@router.post("/")
async def auth_endpoint():
    return {
        "status": "available",
        "service": "Auth service in Vercel"
    }

@router.get("/health")
async def auth_health():
    return {"status": "healthy", "service": "Auth Service (Vercel)"}

@router.post("/login")
async def login():
    return {"message": "Login endpoint - requires proper auth implementation in Vercel"}

@router.post("/register")
async def register():
    return {"message": "Register endpoint - requires proper auth implementation in Vercel"}

@router.get("/profile")
async def get_profile():
    return {"message": "Profile endpoint - requires proper auth implementation in Vercel"}