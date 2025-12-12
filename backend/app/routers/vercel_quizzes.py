from fastapi import APIRouter
import os

# Check if running in Vercel environment
is_vercel = os.environ.get("VERCEL", False)

router = APIRouter(
    prefix="/quizzes",
    tags=["quizzes"],
    responses={404: {"description": "Not found"}},
)

# Create a minimal working quizzes router for Vercel
@router.get("/")
@router.post("/")
async def quizzes_endpoint():
    return {
        "status": "available",
        "service": "Quizzes service in Vercel"
    }

@router.get("/health")
async def quizzes_health():
    return {"status": "healthy", "service": "Quizzes Service (Vercel)"}

@router.get("/questions")
async def get_questions():
    return {"message": "Questions endpoint - requires proper implementation in Vercel"}

@router.post("/submit")
async def submit_quiz():
    return {"message": "Submit quiz endpoint - requires proper implementation in Vercel"}