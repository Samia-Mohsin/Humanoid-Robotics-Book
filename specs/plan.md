# Implementation Plan: Physical AI & Humanoid Robotics Educational Platform

## Technical Context

This plan covers the implementation of a comprehensive educational platform for Physical AI & Humanoid Robotics with the following key components:

- Integrated RAG Chatbot with OpenAI Agents/ChatKit SDK
- Claude Code Subagents and Reusable Agent Skills
- Better-Auth with background questions at signup
- Personalization button per chapter
- Urdu + Roman Urdu translation toggle
- Docusaurus frontend with interactive features
- FastAPI backend with Neon Postgres and Qdrant Cloud

### Architecture Overview

- **Frontend**: Docusaurus v3 with React components, interactive MDX, live code blocks, quizzes, and embedded chatbot
- **Backend**: FastAPI with OpenAI integration, Neon Postgres for user data, and Qdrant Cloud for vector storage
- **Deployment**: GitHub Pages for frontend, Vercel serverless for backend
- **Databases**: Neon Serverless Postgres and Qdrant Cloud Free Tier
- **Target Site**:https://humanoid-robotics-book-xi.vercel.app/
- **SiteMap URL**:https://humanoid-robotics-book-xi.vercel.app/sitemap.xml                                      


## Constitution Check

This implementation aligns with the project constitution which mandates:
- Interactive features including live_code_blocks
- Interactive quizzes with detailed explanations and progress tracking
- High-quality translation with proper quality assurance
- Claude Code Subagents activation
- Reusable Agent Skills in RAG chatbot
- Better-Auth with background questions
- Personalization functionality
- Urdu and Roman Urdu translation support

## Gates

✅ All constitution requirements are addressed in this plan
✅ All previous clarifications are respected (live code blocks with execution capability, etc.)
✅ Architecture supports all required bonuses

## Phase 0: Research

### Research Tasks

1. **OpenAI Agents SDK Integration Research**
   - Decision: Use OpenAI Assistant API for RAG chatbot functionality
   - Rationale: Provides managed thread state, tool calling, and streaming capabilities
   - Alternatives considered: OpenAI Chat Completions API vs Assistant API vs LangChain

2. **Better-Auth Implementation Research**
   - Decision: Use Better-Auth with Neon Postgres adapter
   - Rationale: Provides secure authentication with minimal setup and good Docusaurus integration
   - Alternatives considered: NextAuth.js, Supabase Auth, Custom JWT implementation

3. **Qdrant Vector Storage Research**
   - Decision: Use Qdrant Cloud Free Tier with document chunking strategy
   - Rationale: Supports the required vector operations and integrates well with OpenAI embeddings
   - Alternatives considered: Pinecone, Weaviate, ChromaDB

4. **Docusaurus Chatbot Integration Research**
   - Decision: Embed React chatbot component in Docusaurus layout
   - Rationale: Allows for rich interactive features while maintaining Docusaurus benefits
   - Alternatives considered: Iframe embedding, external widget, native Docusaurus plugin

5. **Text Selection API Research**
   - Decision: Use browser Selection API with custom event handlers
   - Rationale: Provides reliable text selection detection across browsers
   - Alternatives considered: Mutation observers, custom selection libraries

## Phase 1: Data Model

### Core Entities

1. **User**
   - id (UUID)
   - email (string)
   - name (string)
   - experience_level (enum: Beginner, Intermediate, Advanced)
   - has_rtx_gpu (boolean)
   - has_jetson (boolean)
   - preferred_language (enum: English, Urdu, Roman Urdu)
   - created_at (timestamp)
   - updated_at (timestamp)

2. **Conversation**
   - id (UUID)
   - user_id (UUID, foreign key to User)
   - title (string)
   - created_at (timestamp)
   - updated_at (timestamp)

3. **Message**
   - id (UUID)
   - conversation_id (UUID, foreign key to Conversation)
   - role (enum: user, assistant)
   - content (text)
   - sources (JSON array of source references)
   - created_at (timestamp)

4. **QuizAttempt**
   - id (UUID)
   - user_id (UUID, foreign key to User)
   - chapter_id (string)
   - score (integer)
   - total_questions (integer)
   - completed_at (timestamp)

5. **TranslationCache**
   - id (UUID)
   - chapter_id (string)
   - target_language (enum: urdu, roman_urdu)
   - content_hash (string)
   - translated_content (text)
   - created_at (timestamp)

### API Contracts

1. **Authentication Endpoints**
   - `POST /api/auth/register` - User registration with background questions
   - `POST /api/auth/login` - User login
   - `GET /api/auth/me` - Get current user info

2. **RAG Query Endpoints**
   - `POST /api/query` - RAG query with optional selected_text parameter
   - `POST /api/ingest` - Trigger document ingestion to Qdrant

3. **Personalization Endpoints**
   - `GET /api/personalize/{chapter_id}` - Get personalized chapter content
   - `POST /api/personalize/{chapter_id}` - Update personalization preferences

4. **Translation Endpoints**
   - `GET /api/translate/{chapter_id}/{language}` - Get translated chapter content

5. **Quiz Endpoints**
   - `POST /api/quizzes/{chapter_id}/submit` - Submit quiz answers
   - `GET /api/quizzes/{chapter_id}/results` - Get quiz results

## Phase 2: Implementation Strategy

### Component Implementation Order

1. **Backend Infrastructure**
   - Set up FastAPI application with CORS
   - Implement database models and connections
   - Set up Qdrant client and collection management

2. **Authentication System**
   - Implement Better-Auth with custom fields
   - Create user registration with background questions
   - Implement user context management

3. **RAG Service**
   - Implement document ingestion pipeline
   - Create vector storage and retrieval functions
   - Build OpenAI Assistant with custom tools
   - Implement selected-text priority mode

4. **Frontend Foundation**
   - Set up Docusaurus with required plugins
   - Implement Language Context for multilingual support
   - Create basic layout and navigation

5. **Interactive Features**
   - Implement live code blocks (static highlighting)
   - Create quiz components with Neon tracking
   - Add text selection functionality

6. **Chatbot Integration**
   - Build React chatbot component
   - Implement text selection popup
   - Connect to backend RAG service
   - Add Claude agent skills integration

7. **Personalization Features**
   - Create personalization button component
   - Implement backend personalization service
   - Add adaptive content delivery

8. **Translation Features**
   - Implement translation toggle component
   - Create backend translation service
   - Add caching strategy for translations

9. **Deployment Setup**
   - Configure GitHub Pages deployment
   - Set up Vercel deployment for backend
   - Create ingest script for production content

### Key Implementation Details

1. **RAG Chatbot with Selected-Text Mode**
   - When user selects text and clicks "Explain Selection", the bot prioritizes that text
   - Implementation uses browser Selection API to detect selected text
   - Backend processes selected text as primary context with fallback to full knowledge base

2. **Claude Code Subagents Integration**
   - ContentGenerator subagent for content creation
   - Personalizer subagent for adaptive content
   - UrduTranslator and RomanUrduConverter subagents
   - QuizMaster subagent for quiz generation
   - DiagramExplainer subagent for visual content
   - RagIngester subagent for content indexing

3. **Reusable Agent Skills in RAG**
   - explain_concept: Explain complex robotics concepts
   - generate_quiz: Create chapter-specific quizzes
   - translate_to_urdu: Translate content to Urdu
   - translate_to_roman_urdu: Translate content to Roman Urdu
   - simplify_for_beginner: Adapt content for beginners
   - add_advanced_robotics_code: Add advanced code examples
   - explain_diagram_with_vision: Explain diagrams with vision models
   - retrieve_and_answer_from_selection: Answer from selected text
   - generate_ros2_node: Generate ROS2 node code
   - generate_urdf_from_description: Generate URDF from text
   - debug_gazebo_launch: Debug Gazebo launch files

4. **Better-Auth with Background Questions**
   - Custom registration form with experience_level, has_rtx_gpu, has_jetson, preferred_language
   - Neon Postgres storage for user profiles
   - Integration with personalization features

5. **Personalization Engine**
   - One-click "Personalize for me" button per chapter
   - Adaptive content based on user profile
   - GPT-4o powered content modification

6. **Urdu/Roman Urdu Translation**
   - One-click toggle between English/Urdu/Roman Urdu
   - GPT-4o powered translation with robotics terminology
   - Neon caching per user per chapter

## Phase 3: Deployment Strategy

### Frontend Deployment (GitHub Pages)
- Static site generation with Docusaurus
- Pre-built content with multilingual support
- Embedded chatbot widget
- Optimized for performance

### Backend Deployment (Vercel Serverless)
- FastAPI application deployed as serverless functions
- Environment variables for API keys
- Auto-scaling based on demand
- Integration with Neon and Qdrant

### Content Ingestion
- Script to scrape deployed GitHub Pages content
- Parse and chunk MDX content
- Generate embeddings and load to Qdrant
- Run during deployment process

## Risk Analysis

1. **Qdrant Cloud Free Tier Limitations**
   - Mitigation: Monitor usage and implement caching strategies
   - Fallback: Local Qdrant instance for development

2. **Translation API Costs**
   - Mitigation: Aggressive caching with Neon Postgres
   - Fallback: Static translations for common content

3. **OpenAI API Costs**
   - Mitigation: Caching for common queries and responses
   - Fallback: Reduced functionality mode

4. **Multilingual Content Performance**
   - Mitigation: Dynamic loading and caching strategies
   - Fallback: English-only mode

## Success Criteria

- ✅ RAG Chatbot with selected-text mode working
- ✅ Claude Code Subagents activated and functional
- ✅ All agent skills callable in RAG chatbot
- ✅ Better-Auth with background questions implemented
- ✅ Personalization button per chapter working
- ✅ Urdu/Roman Urdu translation toggle functional
- ✅ Live code blocks with execution capability
- ✅ Interactive quizzes with progress tracking
- ✅ Deployed to GitHub Pages and Vercel
- ✅ All constitution bonuses implemented