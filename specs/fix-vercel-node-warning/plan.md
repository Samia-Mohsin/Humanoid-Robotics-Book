# Implementation Plan: Handle Vercel Node.js Version Upgrade Warning

## Technical Context

**Problem**: Vercel deployment shows warning: "Detected "engines": { "node": ">=18.0" } in your `package.json` that will automatically upgrade when a new major Node.js Version is released."

**Current State**:
- package.json contains: "engines": { "node": ">=18.0" }
- Warning appears during Vercel deployments
- Warning is informational and harmless but creates noise

**Solution**: Update the engines field to specify "node": "20.x" to target the current LTS version while reducing warning noise.

## Constitution Check

- ✅ Minimal change principle: Only update engines field in package.json
- ✅ Backwards compatibility: Node 20.x is compatible with Docusaurus 3.x
- ✅ Security: Allows security patch updates while targeting stable LTS
- ✅ Performance: No performance impact from Node version change

## Gates

- ✅ Scope: Within package.json configuration management
- ✅ Dependencies: No external dependencies affected
- ✅ Architecture: Maintains existing architecture with improved configuration

## Phase 0: Research

### Research Findings

**Decision**: Change from ">=18.0" to "20.x" in package.json engines field
**Rationale**: Node.js 20 is the current LTS version that provides stability while allowing security patch updates. This reduces warning noise while maintaining compatibility with Docusaurus 3.x.
**Alternatives considered**:
1. Keep ">=18.0" (rejected - warning continues)
2. Pin to "20.x" (chosen - reduces warning, maintains LTS stability)
3. Pin to "22.x" (rejected - newer, less tested with Docusaurus)
4. Remove engines field (rejected - lose version control)

### Node.js Version Compatibility

- **Node.js 20.x**: Current LTS, well-tested with Docusaurus 3.x
- **Security**: Automatically receives security patches
- **Compatibility**: Fully compatible with Docusaurus 3.1.0 and React 18
- **Vercel Support**: Well-supported in Vercel build environment

## Phase 1: Implementation

### Files to Modify

1. **unified-site/package.json** - Update the "engines" field

### Implementation Steps

1. **Update engines field**:
   - Locate "engines" key in package.json
   - Change value from { "node": ">=18.0" } to { "node": "20.x" }

2. **Verify remaining configuration**:
   - Ensure other package.json fields remain unchanged
   - Confirm all dependencies are still compatible

3. **Test configuration**:
   - Verify local build works with new Node version
   - Confirm all functionality remains intact

### Expected package.json engines content after change:

```json
{
  "engines": {
    "node": "20.x"
  }
}
```

### Dashboard Project Settings

- No Vercel dashboard adjustments needed
- The engines field in package.json will guide Vercel's Node.js selection
- Vercel will use Node 20.x and continue to apply security patches

## Phase 2: Verification

### Post-Implementation Verification Steps

1. **Check local build**: Confirm site builds successfully with new Node version constraint
2. **Deploy to Vercel**: Push changes to trigger new deployment
3. **Monitor Vercel logs**: Verify absence of Node.js version upgrade warning
4. **Test site functionality**: Verify all pages load correctly after deployment
5. **Validate build time**: Confirm build completes within expected timeframe

### Success Criteria

- [ ] package.json engines field updated to "node": "20.x"
- [ ] Local build completes successfully
- [ ] Vercel deployment succeeds without Node.js version warning
- [ ] Site deploys successfully with all functionality intact
- [ ] All pages accessible and properly formatted
- [ ] Build time performance maintained