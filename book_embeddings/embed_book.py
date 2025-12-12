import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import openai
from openai import OpenAI

class BookEmbedder:
    """
    A class to load a book (PDF or markdown), split it into semantic chunks,
    and generate embeddings using OpenAI text-embedding-3-large.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the BookEmbedder with OpenAI API key.

        Args:
            openai_api_key (str): OpenAI API key. If None, will try to get from environment variable.
        """
        if openai_api_key:
            self.api_key = openai_api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Initialize text splitter with specified parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def load_book_content(self, book_path: str) -> str:
        """
        Load the content of a book from PDF or markdown file.

        Args:
            book_path (str): Path to the book file (PDF or markdown)

        Returns:
            str: The full text content of the book
        """
        book_path = Path(book_path)

        if not book_path.exists():
            raise FileNotFoundError(f"Book file not found: {book_path}")

        file_extension = book_path.suffix.lower()

        if file_extension == '.pdf':
            return self._load_pdf(book_path)
        elif file_extension in ['.md', '.markdown']:
            return self._load_markdown(book_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only PDF and markdown files are supported.")

    def _load_pdf(self, pdf_path: Path) -> str:
        """
        Load text content from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF file

        Returns:
            str: The text content of the PDF
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required to read PDF files. Install it with: pip install PyPDF2")

        text_content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num}: {e}")

        return text_content

    def _load_markdown(self, md_path: Path) -> str:
        """
        Load text content from a markdown file.

        Args:
            md_path (Path): Path to the markdown file

        Returns:
            str: The text content of the markdown file
        """
        with open(md_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_text(self, text: str) -> List[Document]:
        """
        Split the text into semantic chunks using RecursiveCharacterTextSplitter.

        Args:
            text (str): The full text to split

        Returns:
            List[Document]: List of Document objects containing the text chunks
        """
        # Create a Document object from the text
        doc = Document(
            page_content=text,
            metadata={"source": "book", "type": "book_content"}
        )

        # Split the document
        chunks = self.text_splitter.split_documents([doc])

        # Add chunk numbers as metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_number"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def generate_embeddings(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for the text chunks using OpenAI text-embedding-3-large.

        Args:
            chunks (List[Document]): List of Document objects containing the text chunks

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the embeddings and metadata
        """
        embeddings_data = []

        # Process chunks in batches to respect API limits
        batch_size = 20  # OpenAI's embedding API can handle up to 2048 texts per request

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Extract the text content from each chunk
            texts = [chunk.page_content for chunk in batch]

            try:
                # Generate embeddings using OpenAI's text-embedding-3-large model
                response = self.client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-large"
                )

                # Process the embeddings response
                for j, embedding_obj in enumerate(response.data):
                    chunk = batch[j]
                    embedding_data = {
                        "chunk_id": i + j,
                        "text": chunk.page_content,
                        "embedding": embedding_obj.embedding,
                        "metadata": chunk.metadata,
                        "embedding_model": "text-embedding-3-large"
                    }
                    embeddings_data.append(embedding_data)

                print(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

            except Exception as e:
                print(f"Error generating embeddings for batch starting at {i}: {e}")
                # If there's an error, add the chunk with a placeholder embedding
                for j, chunk in enumerate(batch):
                    embedding_data = {
                        "chunk_id": i + j,
                        "text": chunk.page_content,
                        "embedding": None,
                        "metadata": chunk.metadata,
                        "embedding_model": "text-embedding-3-large",
                        "error": str(e)
                    }
                    embeddings_data.append(embedding_data)

        return embeddings_data

    def process_book(self, book_path: str) -> List[Dict[str, Any]]:
        """
        Complete process: load book, split into chunks, and generate embeddings.

        Args:
            book_path (str): Path to the book file (PDF or markdown)

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the embeddings and metadata
        """
        print(f"Loading book content from: {book_path}")
        text_content = self.load_book_content(book_path)
        print(f"Book loaded. Total characters: {len(text_content)}")

        print("Splitting text into semantic chunks...")
        chunks = self.split_text(text_content)
        print(f"Text split into {len(chunks)} chunks")

        print("Generating embeddings using OpenAI text-embedding-3-large...")
        embeddings_data = self.generate_embeddings(chunks)
        print(f"Generated embeddings for {len(embeddings_data)} chunks")

        return embeddings_data

    def save_embeddings(self, embeddings_data: List[Dict[str, Any]], output_path: str):
        """
        Save the embeddings data to a JSON file.

        Args:
            embeddings_data (List[Dict[str, Any]]): The embeddings data to save
            output_path (str): Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        print(f"Embeddings saved to: {output_path}")


def main():
    """
    Main function to demonstrate the usage of BookEmbedder.
    """
    # Example usage
    book_path = input("Enter the path to your book file (PDF or markdown): ").strip()

    if not book_path:
        print("No book path provided. Using a sample path for demonstration.")
        book_path = "path/to/your/book.pdf"  # Replace with actual path

    # Initialize the BookEmbedder
    embedder = BookEmbedder()

    # Process the book
    embeddings_data = embedder.process_book(book_path)

    # Save the embeddings to a JSON file
    output_path = "book_embeddings.json"
    embedder.save_embeddings(embeddings_data, output_path)

    # Print summary
    print("\nProcessing complete!")
    print(f"Total chunks processed: {len(embeddings_data)}")
    print(f"Embeddings saved to: {output_path}")

    # Show a sample of the first chunk
    if embeddings_data:
        sample = embeddings_data[0]
        print(f"\nSample chunk (ID: {sample['chunk_id']}):")
        print(f"Text preview: {sample['text'][:100]}...")
        print(f"Embedding length: {len(sample['embedding']) if sample['embedding'] else 'None'}")
        print(f"Chunk size: {sample['metadata']['chunk_size']}")
        print(f"Chunk number: {sample['metadata']['chunk_number']}")


if __name__ == "__main__":
    main()