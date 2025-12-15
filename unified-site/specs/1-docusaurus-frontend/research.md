# Research: Docusaurus 3 Frontend with Embedded Chatbot

## Decision: Floating Chat Bubble Implementation
**Rationale**: Using a React-based floating action button that stays fixed in the bottom-right corner of the screen, implemented as a Docusaurus plugin/component that can be easily integrated into all pages.
**Alternatives considered**:
- Native browser extension (too complex for this use case)
- Iframe-based solution (would create cross-origin issues and limit interaction)
- CSS-only solution (would lack interactivity and dynamic behavior)

## Decision: Text Selection Capture Method
**Rationale**: Using the browser's Selection API to detect and capture text selections, then passing this to the chat component via React state management. This provides reliable cross-browser compatibility and allows for rich text selection handling.
**Alternatives considered**:
- MutationObserver-based detection (overly complex and performance-intensive)
- Custom text selection library (would add unnecessary dependencies)
- Mouse event tracking (less reliable than Selection API)

## Decision: Chapter Action Buttons Implementation
**Rationale**: Creating React components for each button that integrate with the existing authentication and personalization contexts. These will be rendered conditionally based on user authentication status.
**Alternatives considered**:
- Static HTML buttons with JavaScript (less maintainable and harder to integrate with React state)
- Server-side rendering of buttons (would require more complex backend changes)
- Custom Docusaurus theme components (would be harder to maintain)

## Decision: API Integration Pattern
**Rationale**: Using fetch/axios for API calls to the existing /api/chat endpoint, with proper error handling and loading states. This follows standard React patterns and integrates well with the existing backend.
**Alternatives considered**:
- WebSocket connection (unnecessary complexity for this use case)
- GraphQL (would require backend schema changes)
- Custom event system (overly complex for simple API calls)

## Decision: Styling Approach
**Rationale**: Using Tailwind CSS utility classes for styling with shadcn/ui components for complex UI elements. This provides consistent design language and responsive behavior while maintaining performance.
**Alternatives considered**:
- CSS modules (would require more setup and configuration)
- Styled-components (would add bundle size and complexity)
- Vanilla CSS (would not provide the utility-first approach that Tailwind offers)

## Decision: Authentication Integration
**Rationale**: Using @better-auth/react for authentication state management, integrating with the existing authentication system. This provides secure session management and user state.
**Alternatives considered**:
- Custom authentication context (would duplicate existing functionality)
- Third-party auth providers only (would not integrate with existing system)
- JWT tokens only (would lack proper session management)

## Decision: Urdu Translation Implementation
**Rationale**: Using the existing backend translation service with proper RTL (right-to-left) text handling in the UI. This leverages the existing infrastructure while providing proper language support.
**Alternatives considered**:
- Client-side translation libraries (would require additional dependencies and potentially violate API terms)
- Static translation files (would not provide dynamic translation capabilities)
- Third-party translation APIs directly (would duplicate backend functionality)