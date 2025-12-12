"""
Most basic Vercel entry point possible to avoid any crashes
"""

# Define a basic WSGI application as the entry point
def app_instance(environ, start_response):
    # Determine the requested path
    path = environ.get('PATH_INFO', '/')

    # Define basic responses for different paths
    if path == '/' or path == '/index.html':
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        response_data = {
            "message": "Physical AI & Humanoid Robotics Educational Platform API is running",
            "api_docs": "/docs",
            "status": "healthy",
            "environment": "vercel-basic"
        }
        return [json.dumps(response_data).encode('utf-8')]

    elif path == '/health':
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        response_data = {
            "status": "healthy",
            "service": "Physical AI & Humanoid Robotics API",
            "environment": "vercel-basic"
        }
        return [json.dumps(response_data).encode('utf-8')]

    elif path.startswith('/api/'):
        # Handle various API endpoints
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json

        # Basic response for API endpoints
        response_data = {
            "status": "available",
            "endpoint": path,
            "message": "API endpoint is available in basic mode",
            "environment": "vercel-basic"
        }
        return [json.dumps(response_data).encode('utf-8')]

    else:
        # Catch-all for any other paths
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        import json
        response_data = {
            "message": "Basic API server running",
            "requested_path": path,
            "status": "available",
            "environment": "vercel-basic"
        }
        return [json.dumps(response_data).encode('utf-8')]