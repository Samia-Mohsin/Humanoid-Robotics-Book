# Book Embeddings Generator

This project loads a book (PDF or markdown format), splits it into semantic chunks using LangChain's RecursiveCharacterTextSplitter, and generates embeddings using OpenAI's text-embedding-3-large model.

## Features

- Supports both PDF and markdown book formats
- Splits text into chunks of 1000 characters with 200-character overlap
- Generates high-quality embeddings using OpenAI's text-embedding-3-large model
- Saves embeddings with metadata to a JSON file
- Handles large books efficiently with batch processing

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository or create the project directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment:

```bash
cp .env.example .env
# Edit .env to add your OpenAI API key
```

## Usage

1. Place your book file (PDF or markdown) in the project directory or note its path
2. Run the script:

```bash
python embed_book.py
```

3. When prompted, enter the path to your book file, or modify the script to include the path directly

## Configuration

The script uses the following parameters:
- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- `embedding_model`: text-embedding-3-large

These can be modified in the script if needed.

## Output

The script generates a JSON file containing:
- The text chunks
- Embeddings for each chunk
- Metadata including chunk number, size, and source

## Example

```python
from embed_book import BookEmbedder

# Initialize the embedder
embedder = BookEmbedder(openai_api_key="your-api-key-here")

# Process a book
embeddings_data = embedder.process_book("path/to/book.pdf")

# Save the results
embedder.save_embeddings(embeddings_data, "output.json")
```

## Error Handling

The script includes error handling for:
- Missing API keys
- File not found errors
- Unsupported file formats
- API errors during embedding generation

## Notes

- The OpenAI API has rate limits and usage costs. Monitor your usage carefully.
- Large books may take considerable time to process due to API rate limits.
- Make sure you have the proper rights to process the book content.