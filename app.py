from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json
import time
import threading
from datetime import datetime
import speech_recognition as sr
import io
#from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for tracking
participants = {}
meeting_stats = {
    'total_participants': 0,
    'active_participants': 0,
    'fake_detections': 0,
    'accuracy_rate': 0
}

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class MeetingDetector:
    def __init__(self):
        self.voice_recognizer = sr.Recognizer()
        self.face_detection_enabled = True
        self.voice_detection_enabled = True
        
    def detect_face(self, image_data):
        """Detect faces in the image"""
        try:
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            return {
                'faces_detected': len(faces),
                'face_present': len(faces) > 0,
                'confidence': min(len(faces) * 0.3 + 0.7, 1.0)  # Simple confidence calculation
            }
        except Exception as e:
            print(f"Face detection error: {e}")
            return {'faces_detected': 0, 'face_present': False, 'confidence': 0.0}
    
    def analyze_voice_activity(self, audio_data):
        """Analyze voice activity (simplified)"""
        try:
            # This is a simplified voice activity detection
            # In a real scenario, you'd use more sophisticated methods
            if len(audio_data) > 1000:  # Basic threshold
                return {
                    'voice_active': True,
                    'confidence': 0.8,
                    'speech_detected': True
                }
            return {
                'voice_active': False,
                'confidence': 0.2,
                'speech_detected': False
            }
        except Exception as e:
            print(f"Voice analysis error: {e}")
            return {'voice_active': False, 'confidence': 0.0, 'speech_detected': False}
    
    def calculate_participation_score(self, face_data, voice_data):
        """Calculate overall participation score"""
        face_score = face_data.get('confidence', 0) * 0.6
        voice_score = voice_data.get('confidence', 0) * 0.4
        
        total_score = face_score + voice_score
        
        # Determine if participation is genuine
        is_genuine = total_score > 0.5
        
        return {
            'score': total_score,
            'is_genuine': is_genuine,
            'face_score': face_score,
            'voice_score': voice_score
        }

detector = MeetingDetector()

@app.route('/')
def index():
    """Main meeting page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get meeting statistics"""
    return jsonify(meeting_stats)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'msg': 'Connected to meeting integrity system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('join_meeting')
def handle_join_meeting(data):
    """Handle participant joining meeting"""
    participant_id = data.get('participant_id', 'anonymous')
    participants[participant_id] = {
        'join_time': datetime.now(),
        'last_activity': datetime.now(),
        'face_detections': 0,
        'voice_activities': 0,
        'participation_score': 0.0,
        'is_genuine': True
    }
    
    meeting_stats['total_participants'] = len(participants)
    meeting_stats['active_participants'] = len([p for p in participants.values() if p['is_genuine']])
    
    emit('participant_joined', {
        'participant_id': participant_id,
        'stats': meeting_stats
    }, broadcast=True)

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame for face detection"""
    participant_id = data.get('participant_id', 'anonymous')
    image_data = data.get('image_data')
    
    if not image_data:
        return
    
    # Detect face
    face_result = detector.detect_face(image_data)
    
    # Update participant data
    if participant_id in participants:
        participants[participant_id]['last_activity'] = datetime.now()
        participants[participant_id]['face_detections'] += 1
        
        # Simulate voice data for demonstration
        voice_result = {'confidence': 0.6, 'voice_active': True}
        
        # Calculate participation score
        participation = detector.calculate_participation_score(face_result, voice_result)
        participants[participant_id]['participation_score'] = participation['score']
        participants[participant_id]['is_genuine'] = participation['is_genuine']
        
        # Update global stats
        meeting_stats['active_participants'] = len([p for p in participants.values() if p['is_genuine']])
        meeting_stats['fake_detections'] = len([p for p in participants.values() if not p['is_genuine']])
        if meeting_stats['total_participants'] > 0:
            meeting_stats['accuracy_rate'] = (meeting_stats['active_participants'] / meeting_stats['total_participants']) * 100
    
    # Send results back to client
    emit('detection_result', {
        'participant_id': participant_id,
        'face_detection': face_result,
        'participation_score': participation['score'] if participant_id in participants else 0,
        'is_genuine': participation['is_genuine'] if participant_id in participants else False,
        'stats': meeting_stats
    })

@socketio.on('audio_data')
def handle_audio_data(data):
    """Process audio data for voice detection"""
    participant_id = data.get('participant_id', 'anonymous')
    audio_data = data.get('audio_data', [])
    
    # Analyze voice activity
    voice_result = detector.analyze_voice_activity(audio_data)
    
    # Update participant data
    if participant_id in participants:
        participants[participant_id]['voice_activities'] += 1
        participants[participant_id]['last_activity'] = datetime.now()
    
    emit('voice_activity', {
        'participant_id': participant_id,
        'voice_result': voice_result
    })

@socketio.on('update_settings')
def handle_settings_update(data):
    """Update detection settings from admin dashboard"""
    if data.get('face_detection') is not None:
        detector.face_detection_enabled = data['face_detection']
    if data.get('voice_detection') is not None:
        detector.voice_detection_enabled = data['voice_detection']
    
    emit('settings_updated', {
        'face_detection': detector.face_detection_enabled,
        'voice_detection': detector.voice_detection_enabled
    }, broadcast=True)

if __name__ == '__main__':
    print("Starting Smart Meeting Integrity System...")
    print("Face detection enabled:", detector.face_detection_enabled)
    print("Voice detection enabled:", detector.voice_detection_enabled)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)