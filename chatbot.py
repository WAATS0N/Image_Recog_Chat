import requests

OLLAMA_MODEL = "llava"

def chat_with_bot(user_message, image_caption=""):
    """Send user input and optional image caption to LLaVA for a response"""
    url = "http://localhost:11434/api/generate"
    full_prompt = f"Based on this image description: '{image_caption}', answer this question: {user_message}"
    
    data = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json().get("response", "No response generated.") if response.status_code == 200 else "Error in chatbot response."
