import numpy as np
from pathlib import Path

def read_embeddings(PATH: str = '../datasets/embeddings/embeddings.npz') -> dict:
    print(f"Reading embeddings from {PATH}")
    data_export = {}
    data = np.load(PATH)

    for key in data.files:
        data_export[key] = data[key].tolist()
    
    return data_export

def get_text(path):
    """
    Reads the content of a text file and returns it as a string.
    
    Parameters:
    - path: Path to the text file.

    Returns:
    - The content of the text file as a string.
    """
    if Path(path).exists():
        return Path(path).read_text(encoding='utf-8').strip()
    else:
        return ""

if __name__ == "__main__":
    # EXAMPLE USAGE
    read_embeddings()

