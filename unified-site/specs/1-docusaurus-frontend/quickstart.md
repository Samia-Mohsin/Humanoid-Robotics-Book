# Quickstart: Docusaurus 3 Frontend with Embedded Chatbot

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Access to the backend API at `/api/chat`
- Tailwind CSS configured in your Docusaurus project
- @better-auth/react installed and configured

## Installation

1. **Install required dependencies**:
   ```bash
   npm install @better-auth/react
   # Tailwind CSS should already be configured
   # shadcn/ui components should already be installed
   ```

2. **Create the necessary directory structure**:
   ```bash
   mkdir -p src/components/Chatbot
   mkdir -p src/components/Chapter
   mkdir -p src/components/UI
   mkdir -p src/contexts
   mkdir -p src/hooks
   mkdir -p src/utils
   ```

## Setup

### 1. Add the Floating Chat Component

Add the floating chat bubble to your Docusaurus layout by including it in your main layout component or as a plugin:

```jsx
// In your main layout or App component
import FloatingChat from './components/Chatbot/FloatingChat';

function Layout() {
  return (
    <>
      {/* Your existing layout */}
      <FloatingChat />
    </>
  );
}
```

### 2. Configure Text Selection Handling

The chat component will automatically detect selected text and include it as context when the user opens the chat or sends a message.

### 3. Add Chapter Action Buttons

Include the chapter action buttons in your chapter pages:

```jsx
// In your chapter pages/components
import ChapterActions from './components/Chapter/ChapterActions';

function ChapterPage() {
  return (
    <div>
      <ChapterActions chapterId="chapter-1" />
      {/* Your chapter content */}
    </div>
  );
}
```

### 4. Set up Context Providers

Wrap your application with the necessary context providers:

```jsx
// In your main App or clientRoot component
import { AuthProvider } from './contexts/AuthContext';
import { PersonalizationProvider } from './contexts/PersonalizationContext';

export default function Root({ children }) {
  return (
    <AuthProvider>
      <PersonalizationProvider>
        {children}
      </PersonalizationProvider>
    </AuthProvider>
  );
}
```

## Usage

### Floating Chat Bubble
- Appears in the bottom-right corner of all pages
- Click to open the chat interface
- Automatically captures selected text as context
- Connects to the `/api/chat` endpoint

### Chapter Action Buttons (for logged-in users)
- "Personalize for Me": Adapts content based on user preferences
- "اردو میں ترجمہ": Translates content to Urdu
- Progress circle: Shows completion status for the chapter

## API Integration

The frontend components communicate with the backend through:

- `POST /api/chat` - For chat functionality
- `POST /api/content/personalize` - For content personalization
- `POST /api/translate` - For translation functionality
- `GET/PUT /api/profile/preferences` - For user preferences

## Development

1. **Start the development server**:
   ```bash
   npm start
   ```

2. **Verify components are working**:
   - Floating chat bubble appears on all pages
   - Text selection is captured when opening chat
   - Chapter action buttons appear for authenticated users
   - All API calls return expected responses

## Testing

1. **Verify chat functionality**:
   - Chat opens when clicking the floating bubble
   - Selected text is included as context
   - Messages are properly sent and received

2. **Verify chapter actions**:
   - Buttons appear for authenticated users
   - Personalization and translation work as expected
   - Progress tracking updates correctly

3. **Verify authentication integration**:
   - Components behave differently based on authentication status
   - User preferences are properly loaded and applied