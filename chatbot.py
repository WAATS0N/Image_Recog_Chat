import requests

OLLAMA_MODEL = "llava"

def chat_with_bot(user_message, image_caption="", history=[]):
    """Send user input, optional image caption, and conversation history to LLaVA for a response"""
    url = "http://localhost:11434/api/generate"
    history_prompt = " ".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in history])
    if image_caption == "":
        full_prompt=f"Answer general questions" 
    else:
        full_prompt = f"{history_prompt}\nUser: {user_message}\nBased on this image description: '{image_caption}', answer this question:"

    data = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        bot_response = response.json().get("response", "No response generated.")
        history.append({"user": user_message, "bot": bot_response})
        return bot_response
    else:
        return "Error in chatbot response."


