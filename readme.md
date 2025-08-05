# Server Flask ima TrailSierra

This project is a Retrieval-Augmented Generation (RAG) system that uses a sentence transformer model to find relevant documents and a large language model to generate answers based on the retrieved context.

## Features

- **Semantic Search:** Uses `sentence-transformers` to find the most relevant documents based on the user's query.
- **Response Generation:** Uses the Ollama `gemma3n:e4b` model to generate a response based on the retrieved context.
- **Extensible:** You can easily add more documents to the `datasets/original` directory and generate new embeddings.

## Project Structure

```
/
├───main.py             # Main script to run the RAG system
├───requirements.txt    # Python dependencies
├───config/
│   └───read.py         # Utility functions to read data
├───datasets/
│   ├───original/       # Original text files
│   └───embeddings/     # Pre-computed embeddings
├───embeddings/
│   └───embeddings.py   # Script to generate embeddings
└───prompts/
    └───general.md      # System prompt for the LLM
```

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv imaData
    source imaData/bin/activate
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Ollama and pull the model:**
    Make sure you have Ollama installed and running. You can download it from [https://ollama.ai/](https://ollama.ai/).

    Then, pull the `gemma3n:e4b` model:
    ```bash
    ollama pull gemma3n:e4b
    ```

## Usage

To run the project, you can execute the `init.py` script for start local server:

```bash
python init.py
```

## How it Works

1.  The user's query is converted into an embedding vector using a sentence transformer model.
2.  The system performs a semantic search to find the most similar documents from the pre-computed embeddings.
3.  The content of the most relevant documents is retrieved.
4.  The original query, the retrieved content, and a system prompt are sent to the Ollama model.
5.  The Ollama model generates a final response.

## Dependencies

The main dependencies are listed in the `requirements.txt` file. Some of the key dependencies are:

- `ollama`
- `sentence-transformers`
- `torch`
- `numpy`