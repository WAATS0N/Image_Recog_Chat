# Image Recognition Chatbot

## Setup Instructions

1. **Install Ollama**:
   - Follow the instructions on the official Ollama website to install Ollama on your system.
   - And download the llava model with parameters that your system can withold.

2. **Install Required Python Packages**:
   - Ensure you have Python installed on your system.
   - Install the required packages using the following command:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run the Flask Application**:
   - Navigate to the project directory and run the Flask application:
     ```sh
     python app.py
     ```

4. **Access the Application**:
   - Open your web browser and go to `http://localhost:5000` to access the Image Recognition Chatbot.

## Features

- **Upload Image**: Upload an image to get a description and ask questions about it.
- **Upload Video**: Upload a video to get a description of its content and ask questions about it.
- **Live Feed**: Start a live video feed from your camera and ask questions about the live feed.

## Notes

- Ensure that your browser and operating system have granted the necessary permissions for the application to access the camera.
- The application uses the LLaVA model from Ollama for generating image and video descriptions.
