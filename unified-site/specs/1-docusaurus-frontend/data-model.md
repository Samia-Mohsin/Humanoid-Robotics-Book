# Data Model: Docusaurus 3 Frontend with Embedded Chatbot

## Frontend State Models

### ChatState
- **messages**: Array of Message objects
- **isLoading**: Boolean indicating if chat is processing
- **selectedText**: String of currently selected text (optional)
- **isOpen**: Boolean indicating if chat interface is visible

### Message
- **id**: Unique identifier for the message
- **content**: String content of the message
- **sender**: Enum ("user", "assistant")
- **timestamp**: ISO date string
- **context**: String of context that was provided (optional)

### UserPreferences
- **id**: User identifier
- **experienceLevel**: Enum ("beginner", "intermediate", "advanced")
- **programmingLanguages**: Array of strings
- **preferredLanguage**: String (default: "en")
- **personalizationEnabled**: Boolean (default: true)

### ChapterProgress
- **chapterId**: String identifier for the chapter
- **userId**: User identifier
- **completionPercentage**: Number (0-100)
- **lastAccessed**: ISO date string
- **timeSpent**: Number of seconds

## API Contract Models

### ChatRequest
- **message**: String user message
- **selectedText**: String of selected text (optional)
- **userId**: String user identifier (optional)
- **sessionId**: String session identifier (optional)

### ChatResponse
- **response**: String AI-generated response
- **sessionId**: String session identifier
- **contextUsed**: Array of context objects used
- **timestamp**: ISO date string

### PersonalizationRequest
- **content**: String content to personalize
- **userId**: String user identifier
- **context**: Object with user preferences

### PersonalizationResponse
- **personalizedContent**: String personalized content

### TranslationRequest
- **text**: String text to translate
- **sourceLang**: String source language (default: "en")
- **targetLang**: String target language (default: "ur")
- **userId**: String user identifier (optional)

### TranslationResponse
- **translatedText**: String translated content
- **sourceLang**: String source language
- **targetLang**: String target language

## Component Props Models

### FloatingChatProps
- **initialPosition**: Object with x, y coordinates
- **onMessageSend**: Function callback for message submission
- **onOpen**: Function callback when chat opens
- **onClose**: Function callback when chat closes

### ChapterActionsProps
- **chapterId**: String chapter identifier
- **userId**: String user identifier (optional)
- **isAuthenticated**: Boolean user authentication status
- **onPersonalize**: Function callback for personalization
- **onTranslate**: Function callback for translation
- **onProgressUpdate**: Function callback for progress tracking

### ProgressCircleProps
- **percentage**: Number completion percentage (0-100)
- **size**: String size class for the circle
- **strokeWidth**: Number width of the circle stroke
- **color**: String color class for the progress indicator