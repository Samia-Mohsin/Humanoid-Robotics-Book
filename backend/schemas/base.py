from pydantic import BaseModel
from typing import Optional, Dict, Any

class BaseResponse(BaseModel):
    """
    Base response model for all API responses
    """
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {},
                "error": None
            }
        }

class ErrorResponse(BaseResponse):
    """
    Error response model for API errors
    """
    success: bool = False
    message: str
    error: str
    data: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Operation failed",
                "error": "Detailed error message",
                "data": {}
            }
        }

class HealthCheckResponse(BaseModel):
    """
    Health check response model
    """
    status: str = "healthy"
    service: str
    version: str
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "NeuralReader API",
                "version": "1.0.0",
                "timestamp": "2025-12-14T10:00:00Z"
            }
        }