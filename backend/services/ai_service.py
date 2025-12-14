from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import AsyncQdrantClient
from typing import Optional, List, Dict, Any, AsyncGenerator
from core.config import settings
import logging
import time

logger = logging.getLogger(__name__)

class AIService:
    """
    AI Service that handles interaction with language models and vector stores
    """

    def __init__(self):
        # Initialize OpenAI models
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            temperature=0.3,  # Lower temperature for more consistent factual responses
            openai_api_key=settings.OPENAI_API_KEY
        )

        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Initialize Qdrant client
        self.vector_store = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )

        # Set up the collection name
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Set up the system prompt template
        self.system_prompt = """You are an intelligent, helpful AI assistant specialized in discussing and answering questions about a specific book. Your knowledge is limited to the content of this book, which is provided in the retrieved context below.

Use the following guidelines:
- Answer the user's question using ONLY the information from the provided context.
- Be accurate, concise, and directly relevant. Use natural, engaging language.
- If the context contains direct quotes or key details (e.g., chapter references, character names, events), include them where helpful, with brief citations like "(Chapter X)" if available.
- Keep responses to a reasonable length: aim for 3-8 sentences unless the question requires more detail.
- If the question is conversational or follow-up, build naturally on the discussion.
- NEVER invent, assume, or add information not present in the context.
- If the context does not contain enough information to answer fully, say: "Based on the available book content, I don't have sufficient details to answer this accurately." Do not speculate.

Context from the book:
{context}"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])

        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def retrieve_context(self, query: str, k: int = 6) -> List[str]:
        """
        Retrieve relevant context from the vector store based on the query
        """
        try:
            # Embed the query
            query_embedding = await self.embeddings.aembed_query(query)

            # Search in Qdrant
            search_results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True
            )

            # Extract context from results
            contexts = []
            for result in search_results:
                if result.payload and 'content' in result.payload:
                    contexts.append(result.payload['content'])

            return contexts

        except Exception as e:
            logger.error(f"Error retrieving context from vector store: {e}")
            return []

    async def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a response to the query using the provided context
        """
        try:
            start_time = time.time()

            # If no context provided, retrieve it from the vector store
            if not context:
                retrieved_contexts = await self.retrieve_context(query)
                context = "\n\n".join(retrieved_contexts)

            # Format the response
            response = await self.chain.ainvoke({
                "context": context,
                "question": query
            })

            end_time = time.time()
            logger.info(f"Generated response in {end_time - start_time:.2f}s for query: {query[:50]}...")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error processing your request. Please try again."

    async def stream_response(self, query: str, context: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a response to the query using the provided context
        """
        try:
            # If no context provided, retrieve it from the vector store
            if not context:
                retrieved_contexts = await self.retrieve_context(query)
                context = "\n\n".join(retrieved_contexts)

            # Prepare the input for streaming
            inputs = {
                "context": context,
                "question": query
            }

            # Stream the response
            async for chunk in self.chain.astream(inputs):
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield "I encountered an error processing your request."

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the OpenAI embeddings model
        """
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    async def add_to_vector_store(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Add texts to the vector store with metadata
        """
        try:
            # Embed the documents
            embeddings = await self.embed_documents(texts)

            # Add to Qdrant
            await self.vector_store.upload_collection(
                collection_name=self.collection_name,
                vectors=embeddings,
                payloads=[{"content": text, **metadata} for text, metadata in zip(texts, metadatas)],
                ids=ids
            )

            logger.info(f"Added {len(texts)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise


# Singleton instance
_ai_service = None

def get_ai_service() -> AIService:
    """
    Get the singleton AI service instance
    """
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service