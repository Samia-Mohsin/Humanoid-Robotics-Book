# NeuralReader - AI-Powered Book Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)

NeuralReader is a production-ready FastAPI application that provides an intelligent interface for interacting with book content using AI. It features RAG (Retrieval Augmented Generation) capabilities, translation services, user management, and content ingestion.

## Features

- **AI-Powered Chat**: RAG-based chat interface for discussing book content using OpenAI GPT-4o
- **Content Translation**: Translate book chapters to different languages
- **User Management**: JWT-based authentication with admin and user roles
- **Content Ingestion**: Upload and process PDF, TXT, and EPUB files with vector storage
- **Streaming Responses**: Server-Sent Events for real-time chat responses
- **Profile Management**: User preferences and profile management
- **Progress Tracking**: Learning progress and chapter tracking

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL with SQLAlchemy 2.0+ (async)
- **Vector Database**: Qdrant for semantic search
- **AI Integration**: LangChain 0.3+ with OpenAI SDK
- **Authentication**: JWT tokens with python-jose and passlib
- **File Processing**: pypdf for PDF, ebooklib for EPUB
- **Containerization**: Docker and Docker Compose

## Architecture

```
backend/
├── main.py                 # FastAPI application entry point
├── core/                   # Core configurations and utilities
│   ├── config.py          # Application settings
│   ├── database.py        # Database configuration
│   └── security.py        # Authentication and security utilities
├── models/                 # SQLAlchemy models
│   ├── user.py            # User and profile models
│   ├── chat.py            # Chat session and message models
│   └── chapter.py         # Chapter and translation models
├── schemas/                # Pydantic schemas for API validation
├── services/               # Business logic services
│   ├── ai_service.py      # AI and RAG service
│   ├── chat_service.py    # Chat management service
│   ├── translation_service.py # Translation service
│   ├── profile_service.py # Profile management service
│   └── ingestion_service.py # Content ingestion service
├── api/                    # API routes and dependencies
│   ├── routes/            # API endpoints
│   └── deps.py            # Dependency injection utilities
├── utils/                  # Utility functions
├── static/                 # Static files (including logo)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── docker-compose.yml     # Multi-service orchestration
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://neuralreader_user:neuralreader_password@localhost:5432/neuralreader

# Security Configuration
SECRET_KEY=your-super-secret-jwt-signing-key-here-make-it-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Qdrant Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key-here
QDRANT_COLLECTION_NAME=neuralreader_books

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
```

## Installation and Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL (if not using Docker)
- Qdrant (if not using Docker)

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create .env file with your configuration
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Initialize the database
python init_db.py
```

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
# Create database and user as configured in DATABASE_URL

# Initialize the database
python init_db.py

# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh access token

### Chat
- `POST /api/chat` - RAG chat with context
- `POST /api/chat/stream` - Streaming chat response

### Translation
- `POST /api/translate/chapter` - Translate book chapter
- `POST /api/translate/job` - Create translation job
- `GET /api/translate/job/{job_id}` - Get translation job status

### Profile Management
- `GET /api/profile/{user_id}` - Get user profile
- `POST /api/profile/{user_id}` - Update user profile

### Content Ingestion (Admin Only)
- `POST /api/ingest` - Upload and process book files
- `POST /api/ingest/manual` - Manually add text content
- `GET /api/ingest/status/{log_id}` - Get ingestion status

## Usage Examples

### Chat with Book Content

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main themes in this book?",
    "conversation_id": "optional-session-id"
  }'
```

### Upload Book Content

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Authorization: Bearer <admin-jwt-token>" \
  -F "file=@your-book.pdf"
```

### Translate Chapter

```bash
curl -X POST http://localhost:8000/api/translate/chapter \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "chapter_text": "Your chapter content here...",
    "target_language": "es",
    "source_language": "en"
  }'
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=.
```

### Database Migrations

For database schema changes, use Alembic:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

### Adding Dependencies

Add new dependencies to `requirements.txt` and rebuild containers if using Docker:

```bash
# Rebuild containers after adding dependencies
docker-compose build
docker-compose up -d
```

## Deployment

### Docker Compose Deployment

For production deployment with Docker Compose:

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

### Environment-Specific Configurations

- **Development**: Use debug mode, reload, and development settings
- **Staging**: Use staging database and API endpoints
- **Production**: Use production settings with security hardening

## Security Considerations

- JWT tokens with configurable expiration
- Rate limiting to prevent abuse
- Input validation and sanitization
- Secure password hashing with bcrypt
- Admin-only endpoints for sensitive operations
- CORS configuration for web security

## Performance Optimization

- Async database operations with asyncpg
- Vector database for fast semantic search
- Background tasks for long-running operations
- Connection pooling and caching
- Streaming responses for large content

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- PostgreSQL (if not using Docker)
- Qdrant (if not using Docker)
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/samiamohsin/samiamohsin.git
   cd samiamohsin/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   python init_db.py
   ```

6. Start the application:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database Configuration
DATABASE_URL=sqlite+aiosqlite:///./neuralreader.db

# Security Configuration
SECRET_KEY=your-super-secret-jwt-signing-key-here-make-it-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Qdrant Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key-here
QDRANT_COLLECTION_NAME=neuralreader_books

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
```

## API Endpoints

The application provides a comprehensive REST API:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/docs` - Swagger UI documentation
- `GET /api/redoc` - ReDoc documentation
- `POST /api/chat` - RAG chat with context
- `POST /api/chat/stream` - Streaming chat response
- `POST /api/translate/chapter` - Translate book chapter
- `POST /api/translate/job` - Create translation job
- `GET /api/translate/job/{job_id}` - Get translation job status
- `GET /api/profile/{user_id}` - Get user profile
- `POST /api/profile/{user_id}` - Update user profile
- `POST /api/ingest` - Upload and process book files
- `POST /api/ingest/manual` - Manually add text content
- `GET /api/ingest/status/{log_id}` - Get ingestion status

## Usage Examples

### Chat with Book Content

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main themes in this book?",
    "conversation_id": "optional-session-id"
  }'
```

### Upload Book Content

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Authorization: Bearer <admin-jwt-token>" \
  -F "file=@your-book.pdf"
```

### Translate Chapter

```bash
curl -X POST http://localhost:8000/api/translate/chapter \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "chapter_text": "Your chapter content here...",
    "target_language": "es",
    "source_language": "en"
  }'
```

## Architecture

```
backend/
├── main.py                 # FastAPI application entry point
├── core/                   # Core configurations and utilities
│   ├── config.py          # Application settings
│   ├── database.py        # Database configuration
│   └── security.py        # Authentication and security utilities
├── models/                 # SQLAlchemy models
│   ├── user.py            # User and profile models
│   ├── chat.py            # Chat session and message models
│   └── chapter.py         # Chapter and translation models
├── schemas/                # Pydantic schemas for API validation
├── services/               # Business logic services
│   ├── ai_service.py      # AI and RAG service
│   ├── chat_service.py    # Chat management service
│   ├── translation_service.py # Translation service
│   ├── profile_service.py # Profile management service
│   └── ingestion_service.py # Content ingestion service
├── api/                    # API routes and dependencies
│   ├── routes/            # API endpoints
│   └── deps.py            # Dependency injection utilities
├── utils/                  # Utility functions
├── static/                 # Static files (including logo)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── docker-compose.yml     # Multi-service orchestration
```

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running and accessible
2. **Qdrant Connection**: Verify Qdrant is running and API key is correct
3. **OpenAI API**: Check API key validity and rate limits
4. **File Uploads**: Verify file size limits and supported formats

### Logging

The application uses structured logging. Check logs in:
- Console output (for development)
- Application log files (for production)
- Container logs (when using Docker)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the excellent web framework
- LangChain for AI integration
- Qdrant for vector database capabilities
- OpenAI for the powerful language models