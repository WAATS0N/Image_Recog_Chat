from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_from_directory
import os
import cv2
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

@app.route('/')
def index():
    return render_template('index.html', filename="")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads"""
    global image_caption
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image with LLaVA
        image_base64 = preprocess_image(file_path)
        image_caption = generate_caption(image_base64)

        return jsonify({"message": "File uploaded successfully", "filename": filename})

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot conversation"""
    global image_caption
    user_message = request.json.get("message", "")
    bot_response = chat_with_bot(user_message, image_caption)
    return jsonify({"response": bot_response})

# Video Streaming Setup
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
