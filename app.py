
"""
Real-Time Driver Drowsiness Detection System
Main Flask application with computer vision processing and API endpoints
"""

import cv2
import dlib
import numpy as np
import time
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from scipy.spatial import distance as dist
import threading
import queue
import math
import os

app = Flask(__name__)
CORS(app)

# Global variables for storing detection data
current_metrics = {
    'ear': 0.0,
    'mar': 0.0,
    'head_tilt': 0.0,
    'alert_status': 'Normal',
    'timestamp': datetime.now().isoformat()
}

alert_history = []
metrics_queue = queue.Queue(maxsize=100)

# Detection thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness
MAR_THRESHOLD = 0.6   # Mouth Aspect Ratio threshold for yawning
HEAD_TILT_THRESHOLD = 15.0  # Head tilt threshold in degrees
CONSECUTIVE_FRAMES = 20  # Number of consecutive frames for alert

# Frame counters
ear_counter = 0
mar_counter = 0
tilt_counter = 0

class DrowsinessDetector:
    """Main drowsiness detection class using computer vision"""
    
    def __init__(self):
        """Initialize detector with face cascade and landmark predictor"""
        try:
            # Initialize face detector (Haar Cascade - fast detection)
            self.face_cascade = self.load_cascade('haarcascade_frontalface_default.xml')
            
            # Initialize dlib's face detector and landmark predictor
            self.detector = dlib.get_frontal_face_detector()
            # Download shape predictor if not available
            try:
                self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            except:
                print("Warning: shape_predictor_68_face_landmarks.dat not found. Using Haar cascade only.")
                self.predictor = None
            
            # Define facial landmark indices for eyes and mouth
            self.LEFT_EYE = list(range(36, 42))
            self.RIGHT_EYE = list(range(42, 48))
            self.MOUTH = list(range(48, 68))
            
            print("Drowsiness detector initialized successfully")
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.face_cascade = None
            self.detector = None
            self.predictor = None

    def load_cascade(self, cascade_name):
        """Load Haar cascade from available paths"""
        possible_paths = [
            # Try cv2.data path first (if available)
            None,  # Will be set dynamically
            # Try conda environment path
            f"/opt/conda/envs/drowsiness/share/opencv4/haarcascades/{cascade_name}",
            # Try system paths
            f"/usr/share/opencv4/haarcascades/{cascade_name}",
            f"/opt/conda/share/opencv4/haarcascades/{cascade_name}",
            # Try current directory as fallback
            cascade_name
        ]
        
        # Try cv2.data path first if available
        try:
            cv2_data_path = cv2.data.haarcascades + cascade_name
            possible_paths[0] = cv2_data_path
        except AttributeError:
            print("cv2.data not available, trying alternative paths...")
        
        for path in possible_paths:
            if path is None:
                continue
                
            try:
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        print(f"Successfully loaded cascade from: {path}")
                        return cascade
                    else:
                        print(f"Found file but failed to load cascade: {path}")
                else:
                    print(f"Cascade file not found at: {path}")
            except Exception as e:
                print(f"Error loading cascade from {path}: {e}")
        
        print(f"Failed to load {cascade_name} from all paths")
        return None

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        try:
            # Vertical eye landmarks
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            
            # Horizontal eye landmark
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.0

    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        MAR = (|p14-p18| + |p15-p17|) / (2 * |p12-p16|)
        """
        try:
            # Vertical mouth landmarks
            A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])
            B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])
            
            # Horizontal mouth landmark
            C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])
            
            # Calculate MAR
            mar = (A + B) / (2.0 * C)
            return mar
        except:
            return 0.0

    def calculate_head_tilt(self, landmarks):
        """
        Calculate head tilt angle using nose and chin landmarks
        """
        try:
            # Get nose tip (landmark 30) and chin (landmark 8)
            nose_tip = (landmarks[30][0], landmarks[30][1])
            chin = (landmarks[8][0], landmarks[8][1])
            
            # Calculate angle
            dx = chin[0] - nose_tip[0]
            dy = chin[1] - nose_tip[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Normalize angle to -90 to 90 degrees
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180
                
            return abs(angle - 90)  # Return deviation from vertical
        except:
            return 0.0

    def detect_drowsiness(self, frame):
        """
        Main detection function that processes frame and returns metrics
        """
        global current_metrics, ear_counter, mar_counter, tilt_counter
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check if face cascade is loaded
            if self.face_cascade is None:
                # Draw error message on frame
                cv2.putText(frame, "Face cascade not loaded", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame, False
            
            # Detect faces using Haar cascade (faster)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # Draw "No face detected" message
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return frame, False
            
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Use dlib for precise landmark detection if available
            if self.predictor is not None:
                rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = self.predictor(gray, rect)
                
                # Convert landmarks to numpy array
                coords = np.zeros((68, 2), dtype=int)
                for i in range(68):
                    coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
                
                # Calculate metrics
                left_eye = coords[self.LEFT_EYE]
                right_eye = coords[self.RIGHT_EYE]
                mouth = coords[self.MOUTH]
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                mar = self.calculate_mar(mouth)
                head_tilt = self.calculate_head_tilt(coords)
                
                # Draw landmarks
                for (x_pt, y_pt) in coords:
                    cv2.circle(frame, (x_pt, y_pt), 1, (0, 255, 0), -1)
                
                # Draw eye and mouth contours
                cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
                
            else:
                # Fallback to basic detection without landmarks
                # Estimate EAR based on face dimensions (rough approximation)
                face_height = h
                face_width = w
                ear = 0.25 + (face_width / face_height) * 0.1  # Basic estimation
                mar = 0.3  # Default safe value
                head_tilt = 0.0
                
                # Draw message that landmarks are not available
                cv2.putText(frame, "Using basic detection", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Update counters and detect alerts
            alert_status = "Normal"
            alert_triggered = False
            
            if ear < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= CONSECUTIVE_FRAMES:
                    alert_status = "Drowsy Eyes"
                    alert_triggered = True
            else:
                ear_counter = 0
            
            if mar > MAR_THRESHOLD:
                mar_counter += 1
                if mar_counter >= CONSECUTIVE_FRAMES:
                    alert_status = "Yawning"
                    alert_triggered = True
            else:
                mar_counter = 0
            
            if head_tilt > HEAD_TILT_THRESHOLD:
                tilt_counter += 1
                if tilt_counter >= CONSECUTIVE_FRAMES:
                    alert_status = "Head Tilted"
                    alert_triggered = True
            else:
                tilt_counter = 0
            
            # Update global metrics
            current_metrics.update({
                'ear': round(ear, 3),
                'mar': round(mar, 3),
                'head_tilt': round(head_tilt, 1),
                'alert_status': alert_status,
                'timestamp': datetime.now().isoformat()
            })
            
            # Add to metrics queue for charting
            try:
                metrics_queue.put_nowait(current_metrics.copy())
            except queue.Full:
                # Remove oldest item if queue is full
                try:
                    metrics_queue.get_nowait()
                    metrics_queue.put_nowait(current_metrics.copy())
                except queue.Empty:
                    pass
            
            # Draw metrics on frame
            y_offset = 30
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Head Tilt: {head_tilt:.1f}°", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            # Draw alert status
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.putText(frame, f"Status: {alert_status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, alert_triggered
            
        except Exception as e:
            print(f"Error in detection: {e}")
            # Draw error on frame
            cv2.putText(frame, f"Detection error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame, False

# Initialize detector
detector = DrowsinessDetector()

def generate_frames():
    """Generator function for video streaming"""
    camera = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame for drowsiness detection
            processed_frame, alert_triggered = detector.detect_drowsiness(frame)
            
            # Log alerts
            if alert_triggered and len(alert_history) < 1000:  # Limit history size
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'alert_type': current_metrics['alert_status'],
                    'ear': current_metrics['ear'],
                    'mar': current_metrics['mar'],
                    'head_tilt': current_metrics['head_tilt']
                }
                alert_history.append(alert_data)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()

@app.route('/')
def index():
    """Redirect to dashboard"""
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def get_metrics():
    """Get current detection metrics"""
    return jsonify(current_metrics)

@app.route('/metrics/history')
def get_metrics_history():
    """Get historical metrics data for charting"""
    history = []
    temp_queue = queue.Queue()
    
    # Extract all items from queue
    while not metrics_queue.empty():
        try:
            item = metrics_queue.get_nowait()
            history.append(item)
            temp_queue.put(item)
        except queue.Empty:
            break
    
    # Put items back in queue
    while not temp_queue.empty():
        try:
            metrics_queue.put_nowait(temp_queue.get_nowait())
        except queue.Full:
            break
    
    return jsonify(history[-50:])  # Return last 50 data points

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get alert history"""
    return jsonify(alert_history[-100:])  # Return last 100 alerts

@app.route('/alerts', methods=['POST'])
def log_alert():
    """Log a new alert"""
    try:
        alert_data = request.get_json()
        alert_data['timestamp'] = datetime.now().isoformat()
        alert_history.append(alert_data)
        return jsonify({'status': 'success', 'message': 'Alert logged'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/docs')
def api_docs():
    """API documentation"""
    docs = {
        'endpoints': {
            'GET /': 'Dashboard page',
            'GET /dashboard': 'Main dashboard',
            'GET /video_feed': 'MJPEG video stream',
            'GET /metrics': 'Current detection metrics',
            'GET /metrics/history': 'Historical metrics for charting',
            'GET /alerts': 'Get alert history',
            'POST /alerts': 'Log new alert',
            'GET /api/docs': 'This documentation'
        },
        'metrics_format': {
            'ear': 'Eye Aspect Ratio (float)',
            'mar': 'Mouth Aspect Ratio (float)',
            'head_tilt': 'Head tilt angle in degrees (float)',
            'alert_status': 'Current alert status (string)',
            'timestamp': 'ISO timestamp (string)'
        },
        'thresholds': {
            'ear_threshold': EAR_THRESHOLD,
            'mar_threshold': MAR_THRESHOLD,
            'head_tilt_threshold': HEAD_TILT_THRESHOLD,
            'consecutive_frames': CONSECUTIVE_FRAMES
        }
    }
    return jsonify(docs)

if __name__ == '__main__':
    print("Starting Driver Drowsiness Detection System...")
    print("Dashboard: http://localhost:5000/dashboard")
    print("Video feed: http://localhost:5000/video_feed")
    print("API docs: http://localhost:5000/api/docs")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
