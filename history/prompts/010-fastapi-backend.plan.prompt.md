---
id: 12
title: fastapi-backend-langchain-rag
stage: plan
date: 2025-12-14
surface: agent
model: Claude Sonnet 4.5
feature: fastapi-backend
branch: 010-fastapi-backend
user: Claude User
command: /sp.plan
labels: ["backend", "fastapi", "langchain", "rag", "api", "plan", "architecture"]
links:
  spec: ../specs/010-fastapi-backend/spec.md
  plan: ../specs/010-fastapi-backend/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - ../specs/010-fastapi-backend/plan.md
 - ../specs/010-fastapi-backend/research.md
 - ../specs/010-fastapi-backend/data-model.md
 - ../specs/010-fastapi-backend/quickstart.md
 - ../specs/010-fastapi-backend/contracts/chat-api.yaml
 - ../specs/010-fastapi-backend/contracts/api_client.py
tests:
 - API endpoint functionality tests
 - LangChain RAG accuracy tests
 - Translation quality tests
 - User profile management tests
---

## Prompt

# System Prompt for Book-Based RAG Chatbot (/api/chat) with LangChain Implementation

## Purpose
This system prompt is designed for a FastAPI + LangChain RAG application that answers user questions strictly based on the content of a specific book indexed in Qdrant. It ensures grounded, accurate responses with minimal hallucination through proper prompt engineering with ChatPromptTemplate.

## Core System Prompt (Recommended Implementation)

```python
from langchain_core.prompts import ChatPromptTemplate

system_prompt = """You are an intelligent, helpful AI assistant specialized in discussing and answering questions about a specific book. Your knowledge is limited to the content of this book, which is provided in the retrieved context below.

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

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),  # Or use MessagesPlaceholder for full chat history
])
```

## Response snapshot

Created comprehensive implementation plan for the FastAPI backend with LangChain RAG for the Physical AI & Humanoid Robotics Educational Platform. The plan includes:

- Detailed technical context with specific technologies (Python 3.11, FastAPI 0.104.1, Uvicorn 0.24.0, SQLAlchemy 2.0, asyncpg, LangChain 0.3+, langchain-core, OpenAI SDK, Qdrant Client)
- Complete project structure with all necessary directories and files for backend implementation with LangChain-specific components
- Research document outlining technology stack decisions, LangChain implementation patterns, and alternatives considered
- Data model with five key entities: UserProfile, ChatSession, ChatMessage, ChapterContent, TranslationJob, UserProgress, and IngestionLog
- API contracts defined in OpenAPI 3.0 format for all endpoints (/api/chat with LangChain RAG, /api/translate/chapter, /api/profile/{user_id}, /api/ingest)
- Quickstart guide with setup instructions, environment configuration, and deployment information for LangChain-based implementation
- API client implementation for all endpoints

The plan follows modern Python/async best practices and maintains clean architecture patterns with separation of concerns. Special attention was given to LangChain implementation details including:
- ChatPromptTemplate for proper prompt engineering
- RAG (Retrieval Augmented Generation) architecture with Qdrant vector database
- Proper context handling to minimize hallucinations
- Follow-up question processing with separate contextualization chain

All required artifacts have been created and are ready for the implementation phase.

## Outcome

- ✅ Impact: Created complete LangChain-powered backend architecture plan with all required documentation
- 🧪 Tests: Defined key test scenarios for API endpoints, RAG accuracy, translation quality, and user profiles
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md, API contracts, and API client
- 🔁 Next prompts: Ready for /sp.tasks to generate implementation tasks
- 🧠 Reflection: Plan addresses all required endpoints and follows LangChain best practices for RAG applications

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All planning requirements met
- Prompt variant (if applicable): Standard planning workflow with LangChain-specific implementation details
- Next experiment (smallest change to try): Proceed to task generation phase