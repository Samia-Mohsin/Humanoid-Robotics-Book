# Data Model: Backend: FastAPI (Deploy on Vercel or Railway)

## Database Models

### UserProfile
- **id**: String (primary key, UUID)
- **email**: String (unique, indexed)
- **full_name**: String (optional)
- **is_active**: Boolean (default: true)
- **created_at**: DateTime (default: now)
- **updated_at**: DateTime (default: now, onupdate: now)
- **experience_level**: String (enum: beginner, intermediate, advanced)
- **programming_languages**: JSON (array of strings)
- **ai_ml_experience**: String (enum: none, beginner, intermediate, advanced)
- **hardware_experience**: String (enum: none, beginner, intermediate, advanced)
- **learning_goals**: Text (optional)
- **gpu_access**: Boolean (default: false)
- **robotics_kit_experience**: String (optional)
- **preferred_topics**: JSON (array of strings)
- **personalization_enabled**: Boolean (default: true)

### ChatSession
- **id**: String (primary key, UUID)
- **user_id**: String (foreign key to UserProfile)
- **title**: String (auto-generated from first message)
- **created_at**: DateTime (default: now)
- **updated_at**: DateTime (default: now, onupdate: now)
- **is_active**: Boolean (default: true)

### ChatMessage
- **id**: String (primary key, UUID)
- **session_id**: String (foreign key to ChatSession)
- **user_id**: String (foreign key to UserProfile)
- **role**: String (enum: user, assistant)
- **content**: Text
- **selected_text**: Text (optional, context from user selection)
- **context_used**: JSON (array of context documents used)
- **timestamp**: DateTime (default: now)

### BookChapter
- **id**: String (primary key, UUID)
- **title**: String
- **content**: Text
- **source_file**: String (path to original file)
- **chapter_number**: Integer
- **module_id**: String
- **module_title**: String
- **word_count**: Integer
- **created_at**: DateTime (default: now)
- **updated_at**: DateTime (default: now, onupdate: now)
- **is_active**: Boolean (default: true)

### TranslationCache
- **id**: String (primary key, UUID)
- **original_content_hash**: String (unique, hash of original content)
- **source_language**: String (default: "en")
- **target_language**: String (default: "ur")
- **translated_content**: Text
- **created_at**: DateTime (default: now)
- **expires_at**: DateTime (default: now + 30 days)

## API Request/Response Models

### ChatRequest
- **message**: String (required)
- **selected_text**: String (optional)
- **session_id**: String (optional)
- **user_id**: String (optional, auto-filled from auth)

### ChatResponse
- **response**: String (required)
- **session_id**: String (required)
- **context_used**: Array of ContextItem objects (optional)

### ContextItem
- **content**: String (required)
- **source**: String (optional)
- **score**: Float (optional)

### TranslateChapterRequest
- **content**: String (required)
- **source_language**: String (default: "en")
- **target_language**: String (default: "ur")

### TranslateChapterResponse
- **translated_content**: String (required)
- **source_language**: String (required)
- **target_language**: String (required)

### ProfileResponse
- **id**: String (required)
- **email**: String (required)
- **full_name**: String (optional)
- **experience_level**: String (required)
- **programming_languages**: Array of strings (required)
- **ai_ml_experience**: String (required)
- **hardware_experience**: String (required)
- **learning_goals**: String (optional)
- **gpu_access**: Boolean (required)
- **robotics_kit_experience**: String (optional)
- **preferred_topics**: Array of strings (required)
- **personalization_enabled**: Boolean (required)

### ProfileUpdateRequest
- **full_name**: String (optional)
- **experience_level**: String (optional)
- **programming_languages**: Array of strings (optional)
- **ai_ml_experience**: String (optional)
- **hardware_experience**: String (optional)
- **learning_goals**: String (optional)
- **gpu_access**: Boolean (optional)
- **robotics_kit_experience**: String (optional)
- **preferred_topics**: Array of strings (optional)
- **personalization_enabled**: Boolean (optional)

### IngestRequest
- **source_path**: String (optional, default: default content path)
- **force_reindex**: Boolean (optional, default: false)