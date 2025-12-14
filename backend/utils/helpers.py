"""
Utility functions for the NeuralReader application
"""
import re
from typing import Any, Dict, List
import logging
from datetime import datetime
import hashlib
import uuid

logger = logging.getLogger(__name__)

def generate_slug(text: str) -> str:
    """
    Generate a URL-friendly slug from text
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().replace(" ", "-")
    # Remove special characters
    slug = re.sub(r"[^a-z0-9\-_]", "", slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    return slug

def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique identifier with optional prefix
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id

def hash_content(content: str) -> str:
    """
    Generate a hash for content to check for duplicates
    """
    return hashlib.sha256(content.encode()).hexdigest()

def validate_email(email: str) -> bool:
    """
    Validate email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove potentially dangerous characters
    """
    # Remove path traversal characters
    filename = filename.replace("../", "").replace("..\\", "")
    # Remove other potentially dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename

def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format
    """
    return datetime.utcnow().isoformat()

def safe_dict_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with a default
    """
    try:
        return dictionary.get(key, default)
    except AttributeError:
        return default

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones overriding earlier ones
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result