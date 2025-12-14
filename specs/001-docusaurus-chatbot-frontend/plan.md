# Implementation Plan: Docusaurus 3 Frontend with Embedded Chatbot

**Branch**: `001-docusaurus-chatbot-frontend` | **Date**: 2025-12-14 | **Spec**: [../001-docusaurus-chatbot-frontend/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-docusaurus-chatbot-frontend/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a complete, production-ready frontend for an AI-powered interactive book using Docusaurus 3. The site includes an embedded RAG chatbot with text-selection context, per-chapter personalization and Urdu translation buttons (visible only to logged-in users), a progress tracker, and a polished UI. The implementation uses Docusaurus 3, React 18 + TypeScript, Tailwind CSS, shadcn/ui components, and @better-auth/react for authentication.

## Technical Context

**Language/Version**: TypeScript 5.0+, React 18, Node.js 20+
**Primary Dependencies**: Docusaurus 3, @docusaurus/core, @docusaurus/preset-classic, React 18, shadcn/ui, @better-auth/react, Tailwind CSS
**Storage**: Browser localStorage for session state, API backend for user data
**Testing**: Jest, React Testing Library, Cypress for end-to-end tests
**Target Platform**: Web browsers (Chrome, Firefox, Safari, Edge)
**Project Type**: Web application with Docusaurus static site generation
**Performance Goals**: <2s initial page load, <500ms chat response time, <100ms UI interactions
**Constraints**: Responsive design, WCAG 2.1 AA accessibility compliance, <5MB bundle size
**Scale/Scope**: Support 10k+ concurrent users, 100+ book chapters, multiple language support

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

All requirements comply with the project constitution. The implementation follows modern web standards, uses established libraries, and maintains clean architecture patterns.

## Project Structure

### Documentation (this feature)

```text
specs/001-docusaurus-chatbot-frontend/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
frontend/
├── src/
│   ├── components/
│   │   ├── ChatBubble.tsx          # Floating bubble trigger
│   │   ├── ChatBot.tsx             # Main chatbot dialog
│   │   ├── ChapterHeader.tsx       # Injected header with 3 buttons + progress
│   │   ├── TextSelectionProvider.tsx # Global text selection handler
│   │   └── AuthGuard.tsx
│   ├── theme/
│   │   ├── Layout.tsx              # Swizzled to inject global components
│   │   └── DocItem.tsx             # Swizzled to add ChapterHeader
│   ├── lib/
│   │   ├── api.ts                  # Typed API client
│   │   └── auth.ts
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useTextSelection.ts
│   └── plugins/                    # Optional custom plugins
├── static/
│   └── img/                        # Static images
├── docs/                           # Book content
├── docusaurus.config.js            # Docusaurus configuration
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── babel.config.js
```

**Structure Decision**: Web application structure with Docusaurus frontend using the specified component architecture. This follows the requirements from the feature specification with dedicated components for chat, authentication, text selection, and chapter-specific functionality.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
