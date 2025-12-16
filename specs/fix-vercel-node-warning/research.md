# Research: Handle Vercel Node.js Version Upgrade Warning

## Problem Analysis

The Vercel deployment warning "Detected "engines": { "node": ">=18.0" } in your `package.json` that will automatically upgrade when a new major Node.js Version is released" occurs because Vercel detects the loose version constraint in the engines field.

## Node.js Version Landscape

### Current Node.js Versions
- **Node.js 18.x**: Previous LTS, maintenance mode
- **Node.js 20.x**: Current LTS, active long-term support
- **Node.js 22.x**: Current release, not yet LTS

### Docusaurus Compatibility
- Docusaurus 3.x supports Node.js 18, 20, and 22
- Node.js 20 is the recommended version for stability
- All Docusaurus features work properly with Node.js 20

## Vercel Node.js Handling

### Auto-Upgrade Behavior
- Vercel automatically upgrades Node.js minor and patch versions for security
- Major version upgrades are controlled by package.json engines field
- The warning is informational, not an error

### Engine Field Interpretation
- ">=18.0": Triggers warning about potential major version upgrades
- "20.x": Specifies LTS version, reduces warning noise
- Vercel respects package.json engines field for Node.js selection

## Research Findings

### Decision: Change to "20.x" in engines field
**Rationale**: Node.js 20 is the current LTS version providing stability while allowing security patch updates. This reduces warning noise while maintaining Docusaurus compatibility.

### Alternatives Considered

1. **Keep ">=18.0"**
   - Status: Rejected
   - Reason: Warning continues to appear

2. **Pin to "20.x"**
   - Status: Chosen
   - Reason: Reduces warning, targets stable LTS version

3. **Pin to "22.x"**
   - Status: Rejected
   - Reason: Newer version, less tested with Docusaurus ecosystem

4. **Remove engines field**
   - Status: Rejected
   - Reason: Lose version control, potentially cause compatibility issues

## Implementation Considerations

### Compatibility Testing
- Node.js 20.x is fully compatible with Docusaurus 3.1.0
- All Docusaurus plugins and dependencies work with Node.js 20
- No breaking changes expected from version update

### Security Implications
- Node.js 20 LTS receives security patches
- Vercel's auto-upgrade for minors/patches continues
- Maintains security posture while reducing warning noise