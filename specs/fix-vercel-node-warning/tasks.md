# Tasks: Handle Vercel Node.js Version Upgrade Warning

## Feature Overview

**Feature**: Handle Vercel warning about automatic Node.js version upgrades
**Context**: Docusaurus project with "engines": { "node": ">=18.0" } in package.json. Vercel shows warning: "Detected "engines": { "node": ">=18.0" } in your `package.json` that will automatically upgrade when a new major Node.js Version is released." Solution: Change to "20.x" to reduce warning noise while still allowing patch updates.

## Dependencies

- User Story 1 must be completed before User Story 2
- No external dependencies required

## Parallel Execution Examples

- [US1] T003 and T004 can be executed in parallel
- [US2] T006 and T007 can be executed in parallel

## Implementation Strategy

MVP scope: Complete User Story 1 (Update Node.js engine version) to eliminate the warning. This will change the package.json engines field to target Node 20.x LTS.

---

## Phase 1: Setup

- [ ] T001 Create feature branch for Node.js version configuration fix
- [ ] T002 Set up local development environment for testing changes
- [ ] T003 Review current package.json engines configuration

## Phase 2: Foundational Tasks

- [ ] T004 Research Node.js 20.x compatibility with Docusaurus 3.x
- [ ] T005 Verify Vercel supports Node.js 20.x in build environment
- [ ] T006 Document current build behavior for regression testing

## Phase 3: [US1] Update Node.js Engine Version

**Story Goal**: Update the engines field in package.json from ">=18.0" to "20.x" to reduce Vercel warning noise.

**Independent Test Criteria**: package.json engines field updated correctly, local build succeeds with new constraint.

- [X] T007 [US1] Locate engines field in unified-site/package.json
- [X] T008 [US1] Update node version from ">=18.0" to "20.x"
- [X] T009 [US1] Verify syntax and formatting of updated engines field
- [X] T010 [US1] Test local build to ensure functionality remains intact
- [X] T011 [US1] Commit updated package.json with descriptive message
- [X] T012 [US1] Push changes to trigger Vercel deployment

## Phase 4: [US2] Verify Configuration and Deployment

**Story Goal**: Verify that the updated configuration eliminates the warning and maintains all functionality.

**Independent Test Criteria**: Vercel build logs show no Node.js version upgrade warning, all site features work as expected.

- [X] T013 [US2] Monitor Vercel deployment logs for warning elimination
- [X] T014 [US2] Verify site accessibility after deployment
- [X] T015 [US2] Test build process with new Node.js constraint
- [X] T016 [US2] Confirm all documentation pages load correctly
- [X] T017 [US2] Validate build time performance
- [X] T018 [US2] Verify Node.js 20.x is being used in deployment

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T019 Update documentation to reflect new Node.js version approach
- [X] T020 Create quickstart guide for Node.js engine configuration
- [X] T021 Review and optimize package.json for best practices
- [X] T022 Document lessons learned for Node.js version management