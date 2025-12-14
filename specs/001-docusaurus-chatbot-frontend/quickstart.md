# Quickstart Guide: Docusaurus 3 Frontend with Embedded Chatbot

## Prerequisites

- Node.js 20.x or higher
- npm or yarn package manager
- Git for version control

## Setup Instructions

### 1. Clone and Initialize the Project

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
npm install
# or
yarn install
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000

# Authentication Configuration
REACT_APP_AUTH_BASE_URL=http://localhost:8000

# Optional: Analytics and other services
REACT_APP_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

### 3. Run Development Server

```bash
# Start the Docusaurus development server
npm run start
# or
yarn start
```

The application will be available at `http://localhost:3000`

## Key Features Setup

### Authentication
- The application uses @better-auth/react for authentication
- Protected features are automatically hidden for unauthenticated users
- Session management is handled automatically

### Chatbot Integration
- Floating chat bubble appears in bottom-right corner on all pages
- Click the bubble to open the chat interface
- Select text on any page to automatically include it as context in your next message

### Chapter-Specific Features
- For authenticated users, three buttons appear at the top of each chapter:
  1. "Personalize for Me" - Requests personalized content based on user preferences
  2. "اردو میں ترجمہ" - Translates the chapter to Urdu
  3. Progress circle - Shows completion percentage

## Component Structure

The application follows this component structure:

```
src/
├── components/
│   ├── ChatBubble.tsx          # Floating chat bubble
│   ├── ChatBot.tsx             # Main chat interface
│   ├── ChapterHeader.tsx       # Chapter header with buttons
│   ├── TextSelectionProvider.tsx # Text selection handler
│   └── AuthGuard.tsx          # Authentication wrapper
├── theme/                      # Docusaurus swizzled components
│   ├── Layout.tsx
│   └── DocItem.tsx
├── lib/
│   ├── api.ts                 # API client
│   └── auth.ts                # Authentication utilities
├── hooks/
│   ├── useChat.ts             # Chat functionality
│   └── useTextSelection.ts    # Text selection hook
```

## API Integration

The application communicates with the backend API at these endpoints:

- `POST /api/chat` - Chatbot functionality
- `POST /api/personalize` - Content personalization
- `POST /api/translate` - Content translation
- `GET/PUT /api/progress/:chapterId` - Chapter progress tracking

## Development Workflow

### Adding New Pages
- Add new documentation in the `docs/` directory
- Docusaurus will automatically generate navigation

### Custom Components
- Place reusable components in `src/components/`
- Use TypeScript interfaces for props validation
- Follow accessibility best practices

### Styling
- Use Tailwind CSS utility classes
- Leverage shadcn/ui components for consistency
- Follow the existing design system

## Testing

```bash
# Run unit tests
npm run test
# or
yarn test

# Run end-to-end tests
npm run test:e2e
# or
yarn test:e2e

# Generate coverage report
npm run test:coverage
# or
yarn test:coverage
```

## Building for Production

```bash
# Build the static site
npm run build
# or
yarn build

# Serve the built site locally for testing
npm run serve
# or
yarn serve
```

## Deployment

The built site can be deployed to any static hosting service (Vercel, Netlify, GitHub Pages, etc.) or served by a web server.

For Docusaurus-specific deployment options, refer to the [official deployment guide](https://docusaurus.io/docs/deployment).

## Troubleshooting

### Common Issues

1. **Chatbot not responding**: Check that the backend API is running and accessible
2. **Authentication not working**: Verify environment variables and backend auth service
3. **Text selection not capturing**: Ensure no conflicting event handlers are present
4. **Component not rendering**: Check browser console for JavaScript errors

### Getting Help

- Check the component documentation in `specs/001-docusaurus-chatbot-frontend/`
- Review the API contracts in `specs/001-docusaurus-chatbot-frontend/contracts/`
- Consult the data model in `specs/001-docusaurus-chatbot-frontend/data-model.md`