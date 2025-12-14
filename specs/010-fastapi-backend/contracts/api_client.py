from typing import Optional, Dict, Any, List
import httpx
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""
    sessionId: Optional[str] = None
    userId: Optional[str] = None
    chatHistory: Optional[List[Dict[str, str]]] = []


class ChatResponse(BaseModel):
    id: str
    response: str
    timestamp: str
    contextUsed: bool
    sources: List[str]
    ragContext: Optional[Dict[str, Any]] = None


class TranslateChapterRequest(BaseModel):
    chapterId: str
    targetLanguage: str
    userId: Optional[str] = None
    preserveFormat: Optional[bool] = True


class TranslateChapterResponse(BaseModel):
    chapterId: str
    translatedContent: str
    targetLanguage: str
    metadata: Optional[Dict[str, Any]] = None


class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    name: str
    preferences: Dict[str, Any]
    created_at: str
    updated_at: str


class UpdateUserProfileRequest(BaseModel):
    preferences: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    source: Optional[str] = None
    force_reindex: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    jobId: str
    status: str
    message: str
    estimated_completion: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ApiClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Send a message to the AI chatbot with LangChain RAG capabilities
        """
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return ChatResponse.model_validate(response.json())

    async def translate_chapter(self, request: TranslateChapterRequest) -> TranslateChapterResponse:
        """
        Request translation of a book chapter to the specified target language
        """
        response = await self.client.post(
            f"{self.base_url}/api/translate/chapter",
            json=request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return TranslateChapterResponse.model_validate(response.json())

    async def get_user_profile(self, user_id: str) -> UserProfileResponse:
        """
        Retrieve the profile information for a specific user
        """
        response = await self.client.get(
            f"{self.base_url}/api/profile/{user_id}",
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return UserProfileResponse.model_validate(response.json())

    async def update_user_profile(self, user_id: str, request: UpdateUserProfileRequest) -> UserProfileResponse:
        """
        Update the profile information for a specific user
        """
        response = await self.client.post(
            f"{self.base_url}/api/profile/{user_id}",
            json=request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return UserProfileResponse.model_validate(response.json())

    async def ingest_content(self, request: IngestRequest) -> IngestResponse:
        """
        Trigger the ingestion process to re-index book content in the vector database
        """
        response = await self.client.post(
            f"{self.base_url}/api/ingest",
            json=request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return IngestResponse.model_validate(response.json())

    async def close(self):
        """
        Close the HTTP client
        """
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Example usage:
async def example_usage():
    async with ApiClient(base_url="http://localhost:8000") as client:
        # Example chat request
        chat_req = ChatRequest(message="What is ROS2?", context="", sessionId="session-123")
        chat_resp = await client.chat(chat_req)
        print(f"Chat response: {chat_resp.response}")

        # Example translation request
        trans_req = TranslateChapterRequest(chapterId="chapter-1", targetLanguage="ur")
        trans_resp = await client.translate_chapter(trans_req)
        print(f"Translated content: {trans_resp.translatedContent[:100]}...")

        # Example profile request
        profile_resp = await client.get_user_profile(user_id="user-456")
        print(f"User name: {profile_resp.name}")

        # Example ingestion request
        ingest_req = IngestRequest(source="/path/to/book/content", force_reindex=False)
        ingest_resp = await client.ingest_content(ingest_req)
        print(f"Ingestion job started: {ingest_resp.jobId}")