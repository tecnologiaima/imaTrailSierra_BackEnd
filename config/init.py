import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "embeddings"))

from embeddings import generate_embeddings

PATH_DATASET_ORIGINAL = '../datasets/original'
EMBEDDINGS_OUTPUT_PATH = '../datasets/embeddings/embeddings.npz'


def save_embeddings_npz(embeddings_dict: dict, output_path: str):
    """
    Saves a dictionary of embeddings to a compressed NumPy .npz file.

    Parameters:
    - embeddings_dict: Dict with sanitized filenames as keys and embeddings (vectors) as values.
    - output_path: Path to save the .npz file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays[key] = np.array(value, dtype=np.float32)

    np.savez_compressed(output_file, **arrays)
    print(f"✅ Embeddings saved to {output_file.resolve()} as .npz")

def read_documents_from_folder(folder_path: str) -> dict:
    """
    Reads all .txt and .md files from a given folder (supports ../ relative paths).
    """
    folder = Path(folder_path)
    documents = {}

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist or is not a directory.")

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in ('.txt', '.md'):
            documents[file.name] = file.read_text(encoding='utf-8').strip()

    return documents

def process_docs(docs: dict):
    embeddings_data = {}

    for filename, content in docs.items():
        print(f"\n📄 {filename}:\n{'-' * 40}\n{content}\n")
        vectors = generate_embeddings([content])
        print(f"Embeddings for {filename} (first 5 values): {vectors[0][:5]}\n")
        safe_key = Path(filename).stem.replace(" ", "_")
        embeddings_data[safe_key] = vectors[0]

    save_embeddings_npz(embeddings_data, EMBEDDINGS_OUTPUT_PATH)

def init(): 
    doc_origin = read_documents_from_folder(PATH_DATASET_ORIGINAL)
    process_docs(doc_origin)

if __name__ == "__main__":
    # EXAMPLE USAGE
    doc_origin = read_documents_from_folder(PATH_DATASET_ORIGINAL)
    process_docs(doc_origin)