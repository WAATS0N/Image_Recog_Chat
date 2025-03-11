#app.py

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from image_caption import preprocess_image, generate_caption
from chatbot import chat_with_bot
from hand_gesture import HandGestureRecognizer

app = Flask(__name__)

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file is of an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables
image_caption = ""
gesture_recognizer = HandGestureRecognizer()
video_capture = None  # Will be initialized when needed

@app.route('/')
def index():
    return render_template('index.html', filename="")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles image uploads, processes the image, and generates a caption."""
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
        try:
            image_base64 = preprocess_image(file_path)
            image_caption = generate_caption(image_base64)
            return jsonify({"message": "File uploaded successfully", "filename": filename, "caption": image_caption})
        except Exception as e:
            return jsonify({"error": f"Could not process image. {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image from the static/uploads directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot conversation."""
    global image_caption
    user_message = request.json.get("message", "")
    bot_response = chat_with_bot(user_message, image_caption)
    return jsonify({"response": bot_response})

# Video Streaming Setup
def generate_frames():
    """Captures live video frames and streams them."""
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Returns the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_gesture', methods=['POST'])
def process_gesture():
    """Process a frame for hand gesture recognition."""
    global video_capture
    
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
    
    # Capture a frame
    ret, frame = video_capture.read()
    if not ret:
        return jsonify({"error": "Could not capture video frame"}), 500
    
    # Process the frame for hand gestures
    try:
        result = gesture_recognizer.process_frame(frame)
        
        # Generate text interpretation of gestures
        gesture_text = ""
        if result["gestures"]:
            for gesture in result["gestures"]:
                if gesture in gesture_recognizer.gestures:
                    gesture_text += f"{gesture_recognizer.gestures[gesture]} "
                else:
                    gesture_text += f"{gesture} "
        
        return jsonify({
            "gestures": result["gestures"],
            "gesture_text": gesture_text.strip(),
            "frame": f"data:image/jpeg;base64,{result['annotated_frame']}"
        })
    except Exception as e:
        return jsonify({"error": f"Error processing gesture: {str(e)}"}), 500

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stops the video capture when not needed."""
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(debug=True)