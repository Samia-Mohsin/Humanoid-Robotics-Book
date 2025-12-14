# Data Model: FastAPI Backend with LangChain RAG for Physical AI & Humanoid Robotics Platform

## Entities

### UserProfile
**Description**: Stores user-specific settings and preferences for personalization
- **user_id**: string (primary key, UUID format)
- **email**: string (user's email address, unique)
- **name**: string (user's display name)
- **preferences**: JSON object containing:
  - learning_style: 'visual' | 'auditory' | 'reading' | 'kinesthetic'
  - language_preferences: Array<string> (preferred languages)
  - accessibility_settings: object with:
    - high_contrast: boolean
    - font_size: 'small' | 'normal' | 'large' | 'xlarge'
    - screen_reader: boolean
    - reduced_motion: boolean
- **created_at**: DateTime (account creation timestamp)
- **updated_at**: DateTime (last profile update)

### ChatSession
**Description**: Represents a user's ongoing conversation with the AI assistant
- **session_id**: string (primary key, UUID format)
- **user_id**: string (foreign key to UserProfile)
- **title**: string (auto-generated title based on first query)
- **created_at**: DateTime (session creation timestamp)
- **updated_at**: DateTime (last interaction timestamp)
- **expires_at**: DateTime (session expiry timestamp)

### ChatMessage
**Description**: Individual message in a chat conversation
- **message_id**: string (primary key, UUID format)
- **session_id**: string (foreign key to ChatSession)
- **user_id**: string (foreign key to UserProfile)
- **role**: 'user' | 'assistant' (sender of the message)
- **content**: string (message text content)
- **context_used**: string (text context that was used for this response)
- **sources**: Array<string> (references to book content used)
- **timestamp**: DateTime (when message was created)
- **feedback_score**: integer (user feedback rating for the response, -1 to 1)

### ChapterContent
**Description**: Represents book chapters with their content and metadata
- **chapter_id**: string (primary key, UUID format)
- **title**: string (chapter title)
- **content**: string (full chapter content in original language)
- **translated_content**: JSON object containing:
  - ur: string (Urdu translation)
  - [other_language_codes]: string (translations in other languages)
- **metadata**: JSON object with:
  - word_count: integer
  - reading_time: integer (estimated in minutes)
  - topics: Array<string> (main topics covered)
  - difficulty_level: 'beginner' | 'intermediate' | 'advanced'
- **created_at**: DateTime (content creation timestamp)
- **updated_at**: DateTime (last content update)

### TranslationJob
**Description**: Tracks ongoing and completed translation tasks
- **job_id**: string (primary key, UUID format)
- **chapter_id**: string (foreign key to ChapterContent)
- **user_id**: string (foreign key to UserProfile, if initiated by user)
- **source_language**: string (original language code)
- **target_language**: string (target language code)
- **status**: 'pending' | 'in_progress' | 'completed' | 'failed'
- **progress_percentage**: number (0-100)
- **result_url**: string (URL to access translated content when complete)
- **error_message**: string (error details if job failed)
- **created_at**: DateTime (job creation timestamp)
- **completed_at**: DateTime (job completion timestamp, nullable)

### UserProgress
**Description**: Tracks user progress through educational content
- **progress_id**: string (primary key, UUID format)
- **user_id**: string (foreign key to UserProfile)
- **chapter_id**: string (foreign key to ChapterContent)
- **completion_percentage**: number (0-100)
- **time_spent_seconds**: integer (total time spent on chapter)
- **last_accessed**: DateTime (when user last accessed the chapter)
- **bookmarks**: Array<number> (positions in content where user bookmarked)
- **notes**: Array<object> with:
  - position: number (position in content)
  - content: string (user's note)
  - timestamp: DateTime (when note was created)
- **quiz_scores**: Array<object> with:
  - quiz_id: string
  - score: number
  - attempts: number
  - completed_at: DateTime

### IngestionLog
**Description**: Tracks content ingestion and indexing operations
- **log_id**: string (primary key, UUID format)
- **operation_type**: 'add' | 'update' | 'delete' | 'reindex' (type of operation)
- **content_id**: string (identifier of the content being processed)
- **status**: 'started' | 'processing' | 'completed' | 'failed'
- **error_details**: string (details if operation failed)
- **processed_chunks**: integer (number of content chunks processed)
- **total_chunks**: integer (total number of chunks to process)
- **started_at**: DateTime (operation start time)
- **completed_at**: DateTime (operation completion time, nullable)

### RAGContext
**Description**: Stores context information for RAG operations with LangChain
- **context_id**: string (primary key, UUID format)
- **session_id**: string (foreign key to ChatSession)
- **query**: string (original user query)
- **retrieved_context**: string (context retrieved from vector store)
- **formatted_context**: string (context formatted for LLM consumption)
- **relevance_score**: number (similarity score of retrieved content)
- **chunk_ids**: Array<string> (IDs of document chunks used)
- **timestamp**: DateTime (when context was created)

## Relationships

1. **UserProfile** (1) → (Many) **ChatSession**: A user can have multiple chat sessions
2. **ChatSession** (1) → (Many) **ChatMessage**: Each chat session contains multiple messages
3. **ChatSession** (1) → (Many) **RAGContext**: Each chat session generates multiple RAG contexts
4. **UserProfile** (1) → (Many) **UserProgress**: Each user has progress for multiple chapters
5. **ChapterContent** (1) → (Many) **UserProgress**: Each chapter can have progress records for multiple users
6. **ChapterContent** (1) → (Many) **TranslationJob**: Each chapter can have multiple translation jobs (for different languages)
7. **UserProfile** (1) → (Many) **TranslationJob**: Each user can initiate multiple translation jobs

## Validation Rules

### UserProfile
- Email must be a valid email format
- User preferences must conform to defined types
- Accessibility settings must be boolean values

### ChatMessage
- Content must be between 1-2000 characters
- Role must be either 'user' or 'assistant'
- Timestamp must be current or past (not future)

### ChapterContent
- Title and content must not be empty
- Difficulty level must be one of the defined values
- Metadata fields must be properly structured

### TranslationJob
- Source and target languages must be valid language codes
- Status must be one of the defined values
- Progress percentage must be between 0-100

### UserProgress
- Completion percentage must be between 0-100
- Time spent must be non-negative
- Bookmarks must be valid positions within content length

### RAGContext
- Query and context content must not be empty
- Relevance score must be between 0-1 (similarity score)
- Chunk IDs must be valid identifiers from vector store

## State Transitions

### TranslationJob States
1. **Pending**: Job created but not yet started
2. **In Progress**: Translation is actively being processed
3. **Completed**: Translation finished successfully
4. **Failed**: Translation encountered an error

### ChatSession States
1. **Active**: Session has recent activity (within last 24 hours)
2. **Inactive**: Session has no activity for more than 24 hours
3. **Archived**: Session is old and moved to archive for performance

### IngestionLog States
1. **Started**: Operation initiated
2. **Processing**: Content is being processed
3. **Completed**: Operation finished successfully
4. **Failed**: Operation encountered an error

### RAGContext States
1. **Created**: Context generated from user query
2. **Processed**: Context formatted and prepared for LLM
3. **Used**: Context successfully used in LLM call
4. **Cached**: Context stored for potential reuse

## Constraints

1. **Data Privacy**: All user data must comply with privacy regulations (GDPR, etc.)
2. **API Key Security**: AI service API keys must be securely stored and accessed
3. **Rate Limiting**: API calls to external services must be properly limited
4. **Content Integrity**: Book content must be validated and sanitized before processing
5. **Performance**: Database queries should complete within acceptable timeframes (under 500ms for common operations)
6. **Scalability**: System must handle increasing number of users and content without performance degradation
7. **RAG Quality**: Retrieved context must maintain relevance scores above threshold (0.7)
8. **Context Length**: Formatted context for LLM must not exceed model token limits
9. **Memory Management**: Conversation history should be managed to avoid exceeding memory limits