import cv2
import mediapipe as mp
import numpy as np
import base64
import json

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hand solutions
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define some basic gestures and their meanings
        self.gestures = {
            "open_palm": "Hello/Greetings",
            "closed_fist": "Stop",
            "thumbs_up": "Yes/Good",
            "thumbs_down": "No/Bad",
            "victory": "Peace/Victory",
            "pointing_up": "Attention/One moment",
            "pointing_forward": "You/That"
        }
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        detected_gestures = []
        annotated_frame = frame.copy()
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get gesture from hand landmarks
                gesture = self._identify_gesture(hand_landmarks)
                if gesture:
                    detected_gestures.append(gesture)
                    
                    # Add text label for the gesture
                    h, w, _ = annotated_frame.shape
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    cv2.putText(
                        annotated_frame,
                        f"{gesture}", 
                        (wrist_x - 20, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
        
        # Encode the annotated frame to base64 for web display
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "gestures": detected_gestures,
            "annotated_frame": encoded_frame
        }
        
    def _identify_gesture(self, hand_landmarks):
        """
        Simple gesture recognition based on landmark positions
        """
        # Extract landmark coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
        
        # Thumbs up detection
        if self._is_thumbs_up(landmarks):
            return "thumbs_up"
        
        # Thumbs down detection
        elif self._is_thumbs_down(landmarks):
            return "thumbs_down"
        
        # Open palm detection
        elif self._is_open_palm(landmarks):
            return "open_palm"
        
        # Closed fist detection
        elif self._is_closed_fist(landmarks):
            return "closed_fist"
        
        # Victory sign detection
        elif self._is_victory_sign(landmarks):
            return "victory"
        
        # Pointing up detection
        elif self._is_pointing_up(landmarks):
            return "pointing_up"
            
        # Default case - no recognized gesture
        return None
    
    def _is_thumbs_up(self, landmarks):
        # Simplified thumbs up detection
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Thumb is pointing up and other fingers are curled
        return thumb_tip[1] < landmarks[2][1] and index_tip[1] > landmarks[5][1]
    
    def _is_thumbs_down(self, landmarks):
        # Simplified thumbs down detection
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Thumb is pointing down and other fingers are curled
        return thumb_tip[1] > landmarks[2][1] and index_tip[1] > landmarks[5][1]
    
    def _is_open_palm(self, landmarks):
        # Check if all fingertips are extended (simple approach)
        fingertips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb and 4 fingertips
        finger_bases = [landmarks[i] for i in [2, 5, 9, 13, 17]]  # Base of each finger
        
        # Count extended fingers (fingertip y-coordinate should be less than base)
        extended_fingers = sum(1 for tip, base in zip(fingertips[1:], finger_bases[1:]) if tip[1] < base[1])
        thumb_extended = fingertips[0][0] < finger_bases[0][0] if landmarks[0][0] < 0.5 else fingertips[0][0] > finger_bases[0][0]
        
        return extended_fingers >= 3 and thumb_extended
    
    def _is_closed_fist(self, landmarks):
        # Check if all fingers are curled (simple approach)
        fingertips = [landmarks[i] for i in [8, 12, 16, 20]]  # 4 fingertips (excluding thumb)
        finger_bases = [landmarks[i] for i in [5, 9, 13, 17]]  # Base of each finger
        
        # Count curled fingers (fingertip y-coordinate should be greater than base)
        curled_fingers = sum(1 for tip, base in zip(fingertips, finger_bases) if tip[1] > base[1])
        
        return curled_fingers >= 3
    
    def _is_victory_sign(self, landmarks):
        # Check for V sign (index and middle fingers extended, others curled)
        index_extended = landmarks[8][1] < landmarks[5][1]
        middle_extended = landmarks[12][1] < landmarks[9][1]
        ring_curled = landmarks[16][1] > landmarks[13][1]
        pinky_curled = landmarks[20][1] > landmarks[17][1]
        
        return index_extended and middle_extended and ring_curled and pinky_curled
    
    def _is_pointing_up(self, landmarks):
        # Only index finger is extended upward
        index_extended = landmarks[8][1] < landmarks[5][1]
        middle_curled = landmarks[12][1] > landmarks[9][1]
        ring_curled = landmarks[16][1] > landmarks[13][1]
        pinky_curled = landmarks[20][1] > landmarks[17][1]
        
        return index_extended and middle_curled and ring_curled and pinky_curled