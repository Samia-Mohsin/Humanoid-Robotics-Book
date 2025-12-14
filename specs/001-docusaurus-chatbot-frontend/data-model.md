# Data Model: Docusaurus 3 Frontend with Embedded Chatbot

## Entities

### ChatSession
**Description**: Represents a user's ongoing conversation with the chatbot
- **sessionId**: string (unique identifier)
- **userId**: string (authenticated user ID, optional for anonymous sessions)
- **messages**: Array<Message> (conversation history)
- **createdAt**: Date (session creation timestamp)
- **lastActive**: Date (last message timestamp)
- **context**: string (selected text or page context)

### Message
**Description**: Individual message in a chat conversation
- **id**: string (unique identifier)
- **role**: 'user' | 'assistant' (message sender)
- **content**: string (message text)
- **timestamp**: Date (when message was sent/received)
- **context**: string (optional context from text selection)

### UserPreferences
**Description**: Stores user-specific settings for personalization
- **userId**: string (user identifier)
- **learningStyle**: 'visual' | 'auditory' | 'reading' | 'kinesthetic' (preferred learning approach)
- **languagePreferences**: Array<string> (preferred languages)
- **accessibilitySettings**: AccessibilitySettings (accessibility preferences)
- **lastActiveChapter**: string (most recently accessed chapter)

### AccessibilitySettings
**Description**: Accessibility preferences for the user
- **highContrast**: boolean (high contrast mode)
- **fontSize**: 'small' | 'normal' | 'large' | 'xlarge' (text size preference)
- **screenReader**: boolean (screen reader compatibility mode)
- **reducedMotion**: boolean (reduce animation preference)

### ChapterProgress
**Description**: Tracks user progress through educational content
- **userId**: string (user identifier)
- **chapterId**: string (chapter identifier)
- **completionPercentage**: number (0-100)
- **timeSpent**: number (seconds spent on chapter)
- **lastAccessed**: Date (last time chapter was accessed)
- **bookmarks**: Array<number> (scroll positions bookmarked)
- **notes**: Array<ChapterNote> (user notes on chapter)

### ChapterNote
**Description**: User-generated note for a specific part of a chapter
- **id**: string (unique identifier)
- **chapterId**: string (chapter identifier)
- **position**: number (position in chapter content)
- **content**: string (note text)
- **createdAt**: Date (creation timestamp)
- **updatedAt**: Date (last update timestamp)

### AuthSession
**Description**: Authentication session data
- **sessionId**: string (session identifier)
- **userId**: string (user identifier)
- **expiresAt**: Date (session expiration)
- **createdAt**: Date (session creation)
- **lastAccessed**: Date (last access time)
- **ipAddress**: string (user IP address)
- **userAgent**: string (browser information)

## Relationships

1. **User** (1) → (Many) **ChatSession**: A user can have multiple chat sessions
2. **User** (1) → (1) **UserPreferences**: Each user has one set of preferences
3. **User** (1) → (Many) **ChapterProgress**: A user can have progress for multiple chapters
4. **ChapterProgress** (1) → (Many) **ChapterNote**: Each chapter progress can have multiple notes
5. **ChatSession** (1) → (Many) **Message**: Each chat session contains multiple messages

## Validation Rules

### ChatSession
- Must have at least one message to be considered active
- Session expires after 24 hours of inactivity
- Context length limited to 10,000 characters

### Message
- Content must be between 1-2000 characters
- Role must be either 'user' or 'assistant'
- Timestamp must be current or past (not future)

### UserPreferences
- Learning style must be one of the defined values
- Language preferences must be valid language codes
- User ID must reference an existing user

### ChapterProgress
- Completion percentage must be between 0-100
- Time spent must be positive
- Chapter ID must reference an existing chapter

## State Transitions

### ChatSession States
1. **Created**: Session initialized but no messages exchanged
2. **Active**: Messages have been exchanged, session is ongoing
3. **Inactive**: No activity for specified period
4. **Archived**: Session completed or user explicitly ended

### ChapterProgress States
1. **Not Started**: User has not accessed the chapter
2. **In Progress**: User has started reading the chapter
3. **Completed**: User has finished the chapter (based on completion percentage)
4. **Bookmarked**: User has saved specific positions in the chapter

## Constraints

1. **Data Privacy**: All user data must comply with privacy regulations
2. **Session Management**: Sessions must be securely managed with proper expiration
3. **Rate Limiting**: API calls should be limited to prevent abuse
4. **Content Integrity**: User-generated content should be sanitized
5. **Performance**: Data operations should complete within acceptable timeframes