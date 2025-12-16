# Implementation Plan: Fix Vercel Builds Warning

## Technical Context

**Problem**: Vercel build shows warning "Due to `builds` existing in your configuration file, the Build and Development Settings defined in your Project Settings will not apply."

**Current State**:
- vercel.json contains a "builds" array that triggers the warning
- Docusaurus site builds properly but with configuration warning
- Vercel natively supports Docusaurus without custom build configuration

**Solution**: Remove the "builds" array from vercel.json to allow Vercel's native Docusaurus detection while preserving other configuration.

## Constitution Check

- ✅ Minimal change principle: Only remove problematic "builds" array
- ✅ Backwards compatibility: Preserve all functionality while removing warning
- ✅ Security: No security implications from configuration change
- ✅ Performance: No performance impact, may improve build times

## Gates

- ✅ Scope: Within configuration management
- ✅ Dependencies: No external dependencies affected
- ✅ Architecture: Maintains existing architecture with improved configuration

## Phase 0: Research

### Research Findings

**Decision**: Remove "builds" array from vercel.json
**Rationale**: Vercel natively supports Docusaurus projects without custom build configuration. The "builds" array is legacy configuration that triggers warnings.
**Alternatives considered**:
1. Keep "builds" array (rejected - causes warning)
2. Replace with newer Vercel build configuration (unnecessary - native detection is sufficient)
3. Remove "builds" array (chosen - eliminates warning, uses native support)

## Phase 1: Implementation

### Files to Modify

1. **vercel.json** - Remove the "builds" array while preserving other configuration

### Implementation Steps

1. **Remove builds array**:
   - Locate "builds" key in vercel.json
   - Remove entire "builds" array while preserving other keys (e.g., "routes", "version")

2. **Verify remaining configuration**:
   - Ensure "routes" configuration is preserved if present
   - Ensure "version" is maintained

3. **Test configuration**:
   - Verify site builds without warning
   - Confirm all functionality remains intact

### Expected vercel.json content after change:

```json
{
  "version": 2,
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/$1"
    }
  ]
}
```

## Phase 2: Verification

### Post-Deploy Verification Steps

1. **Check Vercel logs**: Confirm absence of "builds" configuration warning
2. **Test site functionality**: Verify all pages load correctly
3. **Test routing**: Ensure URL routing works as expected
4. **Test multilingual support**: Verify both English and Urdu locales work
5. **Check build time**: Confirm build completes within expected timeframe

### Success Criteria

- [ ] Vercel build completes without "builds" configuration warning
- [ ] Site deploys successfully with all functionality intact
- [ ] URL routing works correctly
- [ ] All pages accessible and properly formatted
- [ ] Multilingual support preserved