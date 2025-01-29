from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from image_caption import preprocess_image, generate_caption
from chatbot import chat_with_bot

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variable to store image caption
image_caption = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_caption
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process image with LLaVA
            image_base64 = preprocess_image(file_path)
            image_caption = generate_caption(image_base64)

            return render_template('index.html', filename=filename, caption=image_caption)

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot conversation"""
    global image_caption
    user_message = request.json.get("message", "")
    bot_response = chat_with_bot(user_message, image_caption)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
