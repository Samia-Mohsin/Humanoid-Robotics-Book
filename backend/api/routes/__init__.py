# Import all route modules to make them available when importing from api.routes
from .chat import router as chat_router
from .translate import router as translate_router
from .profile import router as profile_router
from .ingest import router as ingest_router

__all__ = [
    "chat_router",
    "translate_router",
    "profile_router",
    "ingest_router"
]