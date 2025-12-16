# Data Model: Vercel Configuration for Docusaurus

## Configuration Structure

### vercel.json Schema

```json
{
  "version": 2,
  "routes": [
    {
      "src": "string",
      "dest": "string"
    }
  ]
}
```

### Fields Definition

- **version** (number): Vercel platform API version (required: 2)
- **routes** (array): URL routing configuration (optional)
  - **src** (string): Source pattern to match
  - **dest** (string): Destination path to rewrite to

### Valid Values

- **version**: Must be 2 for current Vercel platform
- **routes.src**: Regular expression pattern for URL matching
- **routes.dest**: Valid destination path with optional parameter substitution

## Configuration Validation

### Required Fields
- `version` must be present and equal to 2

### Optional Fields
- `routes` may be omitted if no custom routing needed

### Constraints
- File must be valid JSON
- Version must match supported Vercel API version
- Routes must follow Vercel routing syntax

## State Transitions

### Before Fix
- Contains "builds" array causing Vercel warning
- Legacy configuration format

### After Fix
- No "builds" array
- Uses Vercel native framework detection
- Clean configuration without warnings