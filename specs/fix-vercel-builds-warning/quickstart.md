# Quickstart: Vercel Configuration for Docusaurus

## Overview
This guide explains how to properly configure a Docusaurus site for deployment on Vercel without build warnings.

## Prerequisites
- Docusaurus site with standard build configuration
- Vercel account and project set up
- Git repository connected to Vercel

## Configuration Steps

### 1. Create Minimal vercel.json
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

### 2. Remove Legacy Builds Configuration
- Remove any "builds" array from vercel.json
- Allow Vercel to auto-detect Docusaurus configuration
- Keep only non-build configuration (routes, headers, etc.)

### 3. Verify Package.json Scripts
Ensure your package.json includes:
```json
{
  "scripts": {
    "build": "docusaurus build"
  }
}
```

## Deployment
1. Commit the updated vercel.json
2. Push changes to trigger Vercel deployment
3. Verify no build warnings appear in logs

## Verification
- Check Vercel deployment logs for absence of "builds" warning
- Test site functionality after deployment
- Verify all pages load correctly