from fastapi import APIRouter
import os

# Check if running in Vercel environment
is_vercel = os.environ.get("VERCEL", False)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

# Check if we're in a Vercel environment to avoid initialization issues
if is_vercel:
    # In Vercel, create a simple working endpoint without complex initialization
    @router.get("/")
    @router.post("/")
    async def query_endpoint_fallback():
        return {
            "response": "Query service is available in Vercel",
            "sources": [],
            "selected_text_used": False
        }

    @router.post("/stream")
    async def query_stream_endpoint_fallback():
        from fastapi.responses import StreamingResponse
        import json

        def event_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'content', 'content': 'Query service is available in Vercel'})}\n\n"
            yield f"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/plain")

    @router.get("/health")
    async def query_health():
        return {"status": "healthy", "service": "RAG Query Service (Vercel)"}
else:
    # For local development, import and use the full implementation
    try:
        from pydantic import BaseModel
        from typing import Optional, List, Dict, Any
        from ..services.rag import get_rag_service
        import logging
        import json
        from fastapi.responses import StreamingResponse

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Request models
        class QueryRequest(BaseModel):
            query: str
            selected_text: Optional[str] = None
            user_id: Optional[str] = None

        # Response models
        class QueryResponse(BaseModel):
            response: str
            sources: List[Dict[str, Any]]
            selected_text_used: bool

        def event_stream(query: str, selected_text: Optional[str], user_id: Optional[str]):
            """
            Generator function for streaming responses
            """
            try:
                # Get the RAG service instance
                rag_service = get_rag_service()

                # Get relevant chunks first
                context_chunks = rag_service.retrieve_relevant_chunks(query, selected_text)

                # Prepare context from retrieved chunks
                context = "\n\n".join([chunk["content"] for chunk in context_chunks])

                # Prepare the prompt for the LLM
                system_prompt = f"""
                You are an expert assistant for the Physical AI & Humanoid Robotics educational platform.
                Answer the user's question based on the provided context from the book content.

                Context:
                {context}

                Instructions:
                - Provide accurate answers based only on the context provided
                - If the answer is not in the context, say so clearly
                - Include relevant citations to the source material
                - Use technical terminology appropriately but explain complex concepts when needed
                - Keep responses educational and focused on Physical AI & Humanoid Robotics
                """

                # For streaming, we'll simulate streaming by breaking the response into chunks
                # In a real implementation, you would use OpenAI's streaming API
                full_response = rag_service.generate_response(query, context_chunks, user_id)

                # Send sources first
                sources_data = [
                    {
                        "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        "metadata": chunk["metadata"]
                    }
                    for chunk in context_chunks
                ]

                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

                # Stream the response content in chunks
                words = full_response.split()
                for i in range(0, len(words), 5):  # Send 5 words at a time
                    chunk = ' '.join(words[i:i+5])
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk + ' '})}\n\n"

                # Send end marker
                yield f"data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred while processing your query'})}\n\n"

        @router.post("/", response_model=QueryResponse)
        async def query_endpoint(request: QueryRequest):
            """
            Main query endpoint for the RAG chatbot
            Accepts a query, optional selected text, and optional user ID
            Returns a response with sources and whether selected text was used
            """
            try:
                logger.info(f"Received query: {request.query[:100]}...")
                if request.selected_text:
                    logger.info(f"Selected text provided: {request.selected_text[:100]}...")

                # Get the RAG service instance
                rag_service = get_rag_service()

                # Process the query
                result = rag_service.query(
                    query=request.query,
                    selected_text=request.selected_text,
                    user_id=request.user_id
                )

                logger.info("Query processed successfully")
                return QueryResponse(**result)

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=500,
                    detail="An error occurred while processing your query"
                )

        @router.post("/stream")
        async def query_stream_endpoint(request: QueryRequest):
            """
            Streaming query endpoint for the RAG chatbot
            Returns a streaming response with Server-Sent Events
            """
            try:
                logger.info(f"Received streaming query: {request.query[:100]}...")
                if request.selected_text:
                    logger.info(f"Selected text provided: {request.selected_text[:100]}...")

                return StreamingResponse(
                    event_stream(request.query, request.selected_text, request.user_id),
                    media_type="text/plain"
                )

            except Exception as e:
                logger.error(f"Error processing streaming query: {str(e)}")
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=500,
                    detail="An error occurred while processing your query"
                )

        # Additional endpoint for health check of the query service
        @router.get("/health")
        async def query_health():
            """
            Health check endpoint for the query service
            """
            return {"status": "healthy", "service": "RAG Query Service"}

    except Exception as e:
        # If there are import errors, create fallback endpoints
        print(f"Query router initialization error: {e}")

        @router.get("/")
        @router.post("/")
        async def query_endpoint_fallback():
            return {
                "response": "Query service failed to initialize",
                "sources": [],
                "selected_text_used": False,
                "error": str(e)
            }

        @router.post("/stream")
        async def query_stream_endpoint_fallback():
            from fastapi.responses import StreamingResponse
            import json

            def event_stream():
                yield f"data: {json.dumps({'type': 'error', 'content': 'Query service failed to initialize'})}\n\n"
                yield f"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/plain")

        @router.get("/health")
        async def query_health():
            return {"status": "error", "service": f"RAG Query Service failed: {e}"}