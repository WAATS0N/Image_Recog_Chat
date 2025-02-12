from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import cv2
from werkzeug.utils import secure_filename
from image_caption import preprocess_image, generate_caption
from chatbot import chat_with_bot

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables to store captions
image_caption = ""
video_caption = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_caption, video_caption
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

            if filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}:
                # Process video with LLaVA
                video_caption = process_video(file_path)
            else:
                # Process image with LLaVA
                image_base64 = preprocess_image(file_path)
                image_caption = generate_caption(image_base64)

            return render_template('index.html', filename=filename)

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot conversation"""
    global image_caption, video_caption
    user_message = request.json.get("message", "")
    context_caption = image_caption if image_caption else video_caption
    bot_response = chat_with_bot(user_message, context_caption)
    return jsonify({"response": bot_response})

def process_video(video_path):
    """Extract frames from video and generate captions"""
    cap = cv2.VideoCapture(video_path)
    captions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_frame.jpg')
        cv2.imwrite(frame_path, frame)
        frame_base64 = preprocess_image(frame_path)
        caption = generate_caption(frame_base64)
        captions.append(caption)
    cap.release()
    return " ".join(captions)

# Video Streaming Setup
def generate_frames():
    global video_caption
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_frame.jpg')
            with open(frame_path, 'wb') as f:
                f.write(frame)
            frame_base64 = preprocess_image(frame_path)
            video_caption = generate_caption(frame_base64)
            
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
