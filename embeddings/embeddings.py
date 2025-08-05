from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer, util
import torch

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def find_most_similar(query_embedding: torch.Tensor,
                      embeddings: torch.Tensor,
                      texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)
    top_hits = hits[0]
    results = []
    for hit in top_hits:
        result_data = {
            'corpus_id': hit['corpus_id'],
            'text': texts[hit['corpus_id']],
            'score': f"{hit['score']:.4f}"
        }
        results.append(result_data)

    return results

def load_model():
    """
    Loads the SentenceTransformer model (downloads on first run).
    """
    model = SentenceTransformer(MODEL)
    return model

def vectorize_texts(model, texts):
    """
    Obtains embeddings for a list of texts using the given model.
    """
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return embeddings

def show_embeddings(texts, embeddings, n_values=5):
    """
    Displays each text with the first n_values of its embedding vector.
    """
    for text, emb in zip(texts, embeddings):
        print(f"Text: {text}")
        print(f"Vector (first {n_values} values): {emb[:n_values]}")
        print()

def generate_embeddings(texts):
    """
    Generates embeddings for a list of texts.
    
    Parameters:
    - texts: List of strings to vectorize.

    Returns:
    - List of embeddings corresponding to the input texts.
    """
    model = load_model()
    vectors = vectorize_texts(model, texts)
    return vectors

if __name__ == "__main__":
    # EXAMPLE USAGE
    texts = [
        "The cat is on the roof",
        "The dog barks all night long"
    ]
    vectors = generate_embeddings(texts)
    show_embeddings(texts, vectors)
