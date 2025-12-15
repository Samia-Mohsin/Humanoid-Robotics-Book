# Physical AI & Humanoid Robotics Educational Platform

This is a comprehensive educational platform for learning Physical AI and Humanoid Robotics with advanced features like RAG chatbot, personalization, and multilingual support.

## Features

- **Interactive Book Content**: Comprehensive modules on ROS2, Simulation, AI Brain, and VLA Systems
- **RAG-Powered Chatbot**: AI assistant that answers questions based on book content
- **User Authentication**: Sign up/sign in with better-auth
- **Personalization**: Content adapts based on user's background and experience
- **Multilingual Support**: Content translation to Urdu and other languages
- **Responsive Design**: Works on all devices with Docusaurus

## Tech Stack

- **Frontend**: Docusaurus v3, React
- **Backend**: FastAPI, Python
- **Database**: Neon Serverless Postgres
- **Vector Store**: Qdrant
- **AI/ML**: OpenAI, LangChain
- **Authentication**: Custom implementation

## Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL (or Neon account)
- Qdrant (local or cloud)
- OpenAI API key

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/humanoid-robotics-book/humanoid-robotics-book.git
cd humanoid-robotics-book/unified-site
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the backend directory with the following:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/humanoid_robotics
NEON_DATABASE_URL=your_neon_database_url
SECRET_KEY=your-super-secret-jwt-signing-key-make-it-long-and-random
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=humanoid_robotics_book
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

### 3. Set up the frontend

```bash
cd ../
npm install
```

### 4. Run the application

```bash
npm start
```

This will start both the frontend (Docusaurus) and backend (FastAPI) servers concurrently.

## Project Structure

```
unified-site/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application entry point
│   ├── core/               # Configuration and database setup
│   ├── models/             # Database models
│   ├── services/           # Business logic services
│   ├── api/                # API routes
│   ├── scripts/            # Initialization scripts
│   └── requirements.txt    # Python dependencies
├── src/                    # Docusaurus source files
│   ├── components/         # React components
│   ├── contexts/           # React contexts for state management
│   ├── pages/              # Docusaurus pages
│   ├── css/                # Custom CSS
│   └── utils/              # Utility functions
├── docs/                   # Documentation files
├── docusaurus.config.js    # Docusaurus configuration
└── package.json            # Frontend dependencies and scripts
```

## API Endpoints

- `GET /` - Health check
- `GET /docs` - API documentation
- `POST /api/chat` - Chat with RAG bot
- `POST /api/translate` - Translate content
- `GET/PUT /api/profile/preferences` - User preferences
- `GET /api/content/chapters` - Available chapters
- `POST /api/content/personalize` - Personalize content

## Environment Variables

### Backend (.env)
- `DATABASE_URL` - PostgreSQL database URL
- `NEON_DATABASE_URL` - Neon database URL (optional)
- `SECRET_KEY` - JWT secret key
- `QDRANT_HOST` - Qdrant server host
- `QDRANT_PORT` - Qdrant server port
- `QDRANT_API_KEY` - Qdrant API key
- `QDRANT_COLLECTION_NAME` - Qdrant collection name
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL` - OpenAI model to use

### Frontend (.env)
- `REACT_APP_API_URL` - Backend API URL

## Running in Development

```bash
# Start both frontend and backend
npm start

# Or run them separately
npm run frontend  # Start only frontend
npm run backend   # Start only backend
```

## Deployment

### Frontend (GitHub Pages)
```bash
npm run build
npm run deploy
```

### Backend (Vercel, Heroku, etc.)
Deploy the `backend/` directory to your preferred Python hosting platform.

## Key Components

### Chatbot
- RAG (Retrieval Augmented Generation) powered by OpenAI
- Vector search using Qdrant
- Context-aware responses based on book content

### Personalization Engine
- Adapts content based on user's background
- Experience level detection
- Programming language preferences

### Translation Service
- AI-powered translation to Urdu
- Preserves technical terminology
- Context-aware translations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the GitHub repository.