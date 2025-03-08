import requests
import base64
import os

OLLAMA_MODEL = "llava"

def preprocess_image(image_path):
    """Convert image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_caption(image_base64):
    """Send image to LLaVA and get a description"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": OLLAMA_MODEL,
        "prompt": "Describe this image in detail.",
        "stream": False,
        "images": [image_base64]
    }
    response = requests.post(url, json=data)
    return response.json().get("response", "No description generated.") if response.status_code == 200 else "Error in LLaVA response."




