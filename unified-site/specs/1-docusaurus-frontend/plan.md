# Implementation Plan: Docusaurus 3 Frontend with Embedded Chatbot

**Branch**: `1-docusaurus-frontend` | **Date**: 2025-12-15 | **Spec**: [link to spec](../spec.md)
**Input**: Feature specification from `/specs/1-docusaurus-frontend/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a Docusaurus 3 frontend with React 18, Tailwind CSS, and embedded chatbot functionality. The implementation will include a floating chat bubble, text selection capture for context, personalized chapter actions (personalization, translation to Urdu, progress tracking), and integration with the existing /api/chat endpoint.

## Technical Context

**Language/Version**: TypeScript 5.x, React 18, JavaScript ES2022
**Primary Dependencies**: Docusaurus 3, React 18, Tailwind CSS, shadcn/ui, @better-auth/react, OpenAI SDK
**Storage**: N/A (frontend only - data stored via API calls)
**Testing**: Jest, React Testing Library
**Target Platform**: Web browser (Chrome, Firefox, Safari, Edge)
**Project Type**: Web frontend
**Performance Goals**: <100ms UI response time, <1s chat initialization, 95% text selection accuracy
**Constraints**: <5MB bundle size, WCAG 2.1 AA accessibility compliance, mobile-responsive
**Scale/Scope**: Support 1000+ concurrent users, 100+ book chapters, 10+ interactive components

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ Educational clarity: UI components will be designed for clear learning progression
- ✅ Technical accuracy: Implementation will follow React/Docusaurus best practices
- ✅ Practical outcomes: Interactive components will provide hands-on learning
- ✅ Ethical responsibility: UI will include accessibility features and clear user controls
- ✅ Personalization: Chapter header buttons will enable content adaptation
- ✅ RAG Integration: Chat component will integrate with existing RAG system
- ✅ Standards compliance: Will use Tailwind CSS and shadcn/ui for consistent UI
- ✅ Authentication: Will integrate with Better-Auth for user state
- ✅ Multilingual support: Translation functionality will support Urdu
- ✅ Interactive features: Will include floating chat, selection popup, progress tracking

## Project Structure

### Documentation (this feature)

```text
specs/1-docusaurus-frontend/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
unified-site/
├── src/
│   ├── components/
│   │   ├── Chatbot/
│   │   │   ├── FloatingChat.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   └── TextSelectionHandler.jsx
│   │   ├── Chapter/
│   │   │   ├── ChapterActions.jsx
│   │   │   ├── PersonalizeButton.jsx
│   │   │   ├── TranslateButton.jsx
│   │   │   └── ProgressCircle.jsx
│   │   ├── UI/
│   │   │   ├── Button.jsx
│   │   │   └── Modal.jsx
│   │   └── Auth/
│   │       └── AuthWrapper.jsx
│   ├── contexts/
│   │   ├── AuthContext.jsx
│   │   └── PersonalizationContext.jsx
│   ├── hooks/
│   │   ├── useTextSelection.js
│   │   └── useChat.js
│   ├── pages/
│   │   ├── book.js
│   │   └── chat.js
│   └── utils/
│       ├── api.js
│       └── translation.js
├── static/
└── docusaurus.config.js
```

**Structure Decision**: Web frontend structure chosen with React components organized by feature (Chatbot, Chapter, UI, Auth) and shared utilities in dedicated directories.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [None] | [No violations identified] | [N/A] |