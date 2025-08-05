import ollama
from pathlib import Path
import torch
from config.read import read_embeddings, get_text
from embeddings.embeddings import generate_embeddings, find_most_similar

MODEL = 'gemma3n:e4b'
SYSTEM_PROMPT_PATH = Path("prompts/general.md")

def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    """
    Loads the system prompt from a Markdown file.

    Parameters:
    - path: Path to the .md file (default: prompts/system_prompt.md)

    Returns:
    - Contents of the system prompt as a string.
    """
    if path.exists():
        return path.read_text(encoding='utf-8').strip()
    else:
        return ""

def ask_ollama(message: str) -> str:
    """
    Sends a message to the Ollama model and returns the response.
    
    Uses a system prompt loaded from the prompts folder.

    Parameters:
    - message: The user input to send.

    Returns:
    - The model's response as a string.
    """
    try:
        system_prompt = load_system_prompt()
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})

        response = ollama.chat(
            model=MODEL,
            messages=messages
        )
        return response['message']['content']

    except Exception as e:
        return f"[ERROR] Could not connect to Ollama: {e}"

def search_embeddings(msg):
    embeddings = read_embeddings('./datasets/embeddings/embeddings.npz')
    embeddings_msg = generate_embeddings([msg])
    query_tensor = embeddings_msg[0].detach().cpu()
    embeddings_torch = {key: torch.tensor(value, dtype=torch.float32).cpu() for key, value in embeddings.items()}
    keys = list(embeddings_torch.keys())
    embeddings_list = [embeddings_torch[k] for k in keys]
    embeddings_tensor = torch.stack(embeddings_list)
    vectors = find_most_similar(query_tensor, embeddings_tensor, keys)
    context = ""
    for k in vectors:
        doc = k['text']
        path_doc = f'./datasets/original/{doc}.txt'
        content = get_text(path_doc)
        context += f"{content}\n\n"
        
    return context    

def generate_response (msg, data, history):
    context = search_embeddings(msg)
    msg = msg + "\n\nRelevant content:\n" + context
    msg = msg + "\n\nData Extra:\n" + str(data) + "\n\nHistory:\n" + str(history)
    response = ask_ollama(msg)
    return response

if __name__ == "__main__":
    embeddings = read_embeddings('./datasets/embeddings/embeddings.npz')

    msg = "Tengo dificultad para caminar, que debo hacer?"

    embeddings_msg = generate_embeddings([msg])
    query_tensor = embeddings_msg[0].detach().cpu()
    embeddings_torch = {key: torch.tensor(value, dtype=torch.float32).cpu() for key, value in embeddings.items()}
    keys = list(embeddings_torch.keys())
    embeddings_list = [embeddings_torch[k] for k in keys]
    embeddings_tensor = torch.stack(embeddings_list)

    example = find_most_similar(query_tensor, embeddings_tensor, keys)

    for k in example:
        doc = k['text']
        print(doc)
        path_doc = f'./datasets/original/{doc}.txt'
        print(get_text(path_doc))