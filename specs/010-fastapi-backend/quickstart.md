# Quickstart Guide: FastAPI Backend with LangChain RAG for Physical AI & Humanoid Robotics Platform

## Prerequisites

- Python 3.11 or higher
- pip package manager
- PostgreSQL database (for user data)
- Qdrant vector database (for document embeddings)
- OpenAI API key
- Git for version control

## Setup Instructions

### 1. Clone and Initialize the Repository

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies including LangChain and related packages
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost/dbname

# Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=book_content

# AI Service Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo  # or gpt-3.5-turbo

# Application Configuration
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
FRONTEND_URL=http://localhost:3000

# Logging Configuration
LOG_LEVEL=INFO

# LangChain Configuration
LANGCHAIN_TRACING_V2=false  # Set to true if you want to use LangSmith for tracing
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""  # Your LangSmith API key if using tracing
LANGCHAIN_PROJECT="physical-ai-humanoid-platform"
```

### 3. Database Setup

```bash
# Run database migrations (if using Alembic)
alembic upgrade head

# Or set up tables directly using SQLAlchemy models
python -c "from backend.core.database import engine; from backend.models import Base; Base.metadata.create_all(bind=engine)"
```

### 4. Initialize Vector Store with Book Content

```bash
# Index the book content into Qdrant using LangChain
python -m backend.scripts.initialize_vector_store
```

### 5. Run the Development Server

```bash
# Run the FastAPI server with LangChain integration
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Or using the run script
python -m backend
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Chat Endpoint with LangChain RAG
- **POST** `/api/chat` - Interact with the AI assistant powered by LangChain RAG
- Uses ChatPromptTemplate for structured prompt engineering
- Example: `curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message": "What is ROS2?", "sessionId": "session-123"}'`

### Translation Endpoint
- **POST** `/api/translate/chapter` - Translate a chapter to another language
- Example: `curl -X POST http://localhost:8000/api/translate/chapter -H "Content-Type: application/json" -d '{"chapterId": "chapter-1", "targetLanguage": "ur"}'`

### Profile Endpoints
- **GET** `/api/profile/{user_id}` - Get user profile
- **POST** `/api/profile/{user_id}` - Update user profile

### Ingestion Endpoint
- **POST** `/api/ingest` - Re-index book content using LangChain document loaders

## LangChain Implementation Details

### Chat Chain Architecture
The chat functionality uses a LangChain chain with the following components:
1. **ChatPromptTemplate**: System message enforces using only provided context
2. **Qdrant Retriever**: Retrieves relevant book content based on user query
3. **OpenAI LLM**: Generates responses grounded in the retrieved context
4. **Output Parser**: Formats the response for the API

### Prompt Template Structure
The system uses a carefully crafted ChatPromptTemplate:
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
    ("human", "{question}"),
])
```

### Document Retrieval
The RAG system uses Qdrant vector database with LangChain's integration to retrieve relevant book content based on semantic similarity to the user's query.

## Development Workflow

### Adding New Routes
- Add new route files in `backend/api/routes/`
- Import and include them in the main application in `backend/main.py`

### LangChain Chain Development
- LangChain-specific functionality goes in `backend/chains/`
- Follow the pattern of separating RAG logic from API layer
- Use dedicated modules for different chain types (chat, context, etc.)

### Service Layer
- Business logic should be implemented in `backend/services/`
- Separate LangChain-specific operations in dedicated service files
- Follow the pattern of separating concerns between data access, business logic, and API presentation

### Models and Schemas
- Data models (SQLAlchemy) go in `backend/models/`
- Request/response schemas (Pydantic) go in `backend/schemas/`
- LangChain-specific schemas go in `backend/chains/prompt_templates.py`

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_chat.py

# Test LangChain chains separately
pytest tests/test_chains/
```

## Deployment

### For Vercel
1. Ensure all dependencies are in `requirements.txt` (including langchain, langchain-community, langchain-openai)
2. Create a `vercel.json` file with the proper configuration
3. Deploy using `vercel --prod`

### For Railway
1. Set up environment variables in the Railway dashboard (including OPENAI_API_KEY and database URLs)
2. Connect your GitHub repository
3. Deploy using Railway's CI/CD pipeline

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running and credentials are correct
2. **Vector Store**: Verify Qdrant is running and accessible
3. **API Keys**: Check that OpenAI API key is valid and has sufficient credits
4. **CORS Errors**: Verify that FRONTEND_URL is correctly set in environment variables
5. **LangChain Errors**: Check that all LangChain dependencies are installed correctly
6. **Embedding Issues**: Ensure the document loader is correctly chunking and embedding book content

### Getting Help

- Check the API documentation at `/docs` and `/redoc`
- Review the logs for detailed error messages
- Consult the LangChain documentation for RAG-specific issues
- Reach out to the development team if issues persist

## Performance Tips

- Use async functions throughout the application for better concurrency
- Implement caching for frequently accessed content
- Monitor database queries and optimize slow ones
- Use proper indexing in Qdrant for fast vector search
- Implement rate limiting to protect against API abuse
- Optimize chunk size and overlap for document processing to balance retrieval quality and performance
- Use LangChain's caching mechanisms for repeated queries