# Quickstart: Node.js Engine Configuration for Vercel Deployment

## Overview
This guide explains how to properly configure Node.js engine version in package.json to eliminate Vercel deployment warnings while maintaining compatibility.

## Prerequisites
- Docusaurus site with package.json configuration
- Vercel account and project set up
- Git repository connected to Vercel

## Configuration Steps

### 1. Update Engine Version
In your package.json, change the engines field:

```json
{
  "engines": {
    "node": "20.x"
  }
}
```

### 2. Rationale for "20.x"
- Node.js 20 is the current LTS version
- Provides stability with active long-term support
- Fully compatible with Docusaurus 3.x
- Allows security patch updates automatically

### 3. Alternative Configurations
- Use ">=18.0" if broader compatibility is needed (warning will appear)
- Use "22.x" for newer features (not LTS yet)
- Omit engines field to use Vercel's default (not recommended)

## Deployment
1. Commit the updated package.json
2. Push changes to trigger Vercel deployment
3. Verify no Node.js version warnings appear in logs

## Verification
- Check Vercel deployment logs for absence of Node.js version warning
- Test site functionality after deployment
- Verify build completes successfully