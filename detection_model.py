import cv2
import numpy as np
import speech_recognition as sr
import time
import random
from datetime import datetime
import json

class FaceDetector:
    """
    Face detection and fake participation detection using OpenCV
    """
    def __init__(self):
        # Load OpenCV's pre-trained face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Tracking variables
        self.previous_faces = []
        self.face_history = []
        self.detection_threshold = 0.7
        
    def detect_faces(self, image):
        """
        Detect faces in the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_data = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # Calculate face quality metrics
            face_quality = self._calculate_face_quality(face_roi)
            
            face_data.append({
                'bbox': (x, y, w, h),
                'quality': face_quality,
                'eye_count': len(eyes),
                'timestamp': datetime.now()
            })
            
        return face_data
    
    def _calculate_face_quality(self, face_roi):
        """
        Calculate face quality score (0-100)
        """
        # Check image sharpness using Laplacian variance
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Check brightness
        brightness = np.mean(face_roi)
        
        # Normalize scores
        sharpness_score = min(100, sharpness / 100 * 100)
        brightness_score = 100 - abs(brightness - 127) / 127 * 100
        
        return (sharpness_score + brightness_score) / 2
    
    def detect_fake_participation(self, face_data):
        """
        Detect if the participant is using fake methods
        """
        if not face_data:
            return {'is_fake': True, 'reason': 'No face detected', 'confidence': 0.95}
        
        # Check for multiple faces (screen sharing another person)
        if len(face_data) > 1:
            return {'is_fake': True, 'reason': 'Multiple faces detected', 'confidence': 0.88}
        
        face = face_data[0]
        
        # Check face quality (too perfect might be a photo)
        if face['quality'] > 95:
            return {'is_fake': True, 'reason': 'Suspiciously perfect image quality', 'confidence': 0.82}
        
        # Check if face is too static (might be a photo)
        if len(self.face_history) >= 5:
            position_variance = self._calculate_position_variance()
            if position_variance < 10:  # Very little movement
                return {'is_fake': True, 'reason': 'Static face detected (possible photo)', 'confidence': 0.78}
        
        # Check eye detection (real faces should have eyes)
        if face['eye_count'] < 2:
            return {'is_fake': True, 'reason': 'Eyes not properly detected', 'confidence': 0.72}
        
        # Store face for tracking
        self.face_history.append(face)
        if len(self.face_history) > 10:
            self.face_history.pop(0)
        
        return {'is_fake': False, 'reason': 'Genuine participation detected', 'confidence': 0.92}
    
    def _calculate_position_variance(self):
        """
        Calculate variance in face position over time
        """
        if len(self.face_history) < 2:
            return 100
        
        positions = [(face['bbox'][0], face['bbox'][1]) for face in self.face_history]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        return x_variance + y_variance

class VoiceDetector:
    """
    Voice detection and analysis for fake participation
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.voice_history = []
        self.background_noise_level = 0
        self.speech_patterns = []
        
    def detect_voice_activity(self, audio_data):
        """
        Detect if there's genuine voice activity
        """
        try:
            # Convert audio data to audio source
            audio_source = sr.AudioData(audio_data, 16000, 2)
            
            # Check energy levels
            energy_level = self._calculate_energy_level(audio_data)
            
            # Try to recognize speech
            try:
                text = self.recognizer.recognize_google(audio_source, language='en-US')
                speech_detected = True
                speech_text = text
            except sr.UnknownValueError:
                speech_detected = False
                speech_text = ""
            except sr.RequestError:
                speech_detected = False
                speech_text = ""
            
            voice_data = {
                'timestamp': datetime.now(),
                'energy_level': energy_level,
                'speech_detected': speech_detected,
                'speech_text': speech_text,
                'duration': len(audio_data) / 16000  # Assuming 16kHz sample rate
            }
            
            return voice_data
            
        except Exception as e:
            print(f"Voice detection error: {e}")
            return None
    
    def _calculate_energy_level(self, audio_data):
        """
        Calculate energy level of audio signal
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_array**2))
        
        # Normalize to 0-100 scale
        normalized_energy = min(100, rms_energy / 1000 * 100)
        
        return normalized_energy
    
    def detect_fake_voice_participation(self, voice_data):
        """
        Detect if voice participation is fake
        """
        if not voice_data:
            return {'is_fake': True, 'reason': 'No voice data received', 'confidence': 0.90}
        
        # Check for extremely low energy (microphone might be muted)
        if voice_data['energy_level'] < 5:
            return {'is_fake': True, 'reason': 'Microphone appears to be muted', 'confidence': 0.85}
        
        # Check for unnatural speech patterns
        if voice_data['speech_detected']:
            # Check for repetitive phrases (might be playing recorded audio)
            if self._check_repetitive_speech(voice_data['speech_text']):
                return {'is_fake': True, 'reason': 'Repetitive speech pattern detected', 'confidence': 0.80}
        
        # Check for consistent background noise (might be playing audio file)
        if len(self.voice_history) >= 3:
            if self._check_artificial_background(voice_data):
                return {'is_fake': True, 'reason': 'Artificial background noise detected', 'confidence': 0.75}
        
        # Store voice data for analysis
        self.voice_history.append(voice_data)
        if len(self.voice_history) > 20:
            self.voice_history.pop(0)
        
        return {'is_fake': False, 'reason': 'Genuine voice participation', 'confidence': 0.88}
    
    def _check_repetitive_speech(self, text):
        """
        Check if speech is repetitive (indicating recorded audio)
        """
        if len(self.speech_patterns) >= 3:
            recent_patterns = [pattern for pattern in self.speech_patterns[-3:]]
            if text in recent_patterns:
                return True
        
        self.speech_patterns.append(text)
        if len(self.speech_patterns) > 10:
            self.speech_patterns.pop(0)
        
        return False
    
    def _check_artificial_background(self, current_voice_data):
        """
        Check for artificial background noise patterns
        """
        if len(self.voice_history) < 3:
            return False
        
        # Check if background noise is too consistent
        recent_energies = [voice['energy_level'] for voice in self.voice_history[-3:]]
        energy_variance = np.var(recent_energies)
        
        # Very low variance might indicate artificial audio
        return energy_variance < 2

class IntegrityAnalyzer:
    """
    Combines face and voice detection results to determine overall integrity
    """
    def __init__(self):
        self.participant_profiles = {}
        self.false_positive_threshold = 0.3
        
    def analyze_participant_integrity(self, participant_id, face_result, voice_result):
        """
        Analyze overall participant integrity
        """
        if participant_id not in self.participant_profiles:
            self.participant_profiles[participant_id] = {
                'face_scores': [],
                'voice_scores': [],
                'integrity_history': [],
                'false_positive_count': 0
            }
        
        profile = self.participant_profiles[participant_id]
        
        # Calculate weighted scores
        face_weight = 0.6
        voice_weight = 0.4
        
        face_score = 0 if face_result['is_fake'] else face_result['confidence']
        voice_score = 0 if voice_result['is_fake'] else voice_result['confidence']
        
        # Combined integrity score
        integrity_score = (face_score * face_weight) + (voice_score * voice_weight)
        
        # Apply temporal smoothing
        profile['face_scores'].append(face_score)
        profile['voice_scores'].append(voice_score)
        profile['integrity_history'].append(integrity_score)
        
        # Keep only recent history
        if len(profile['integrity_history']) > 10:
            profile['face_scores'].pop(0)
            profile['voice_scores'].pop(0)
            profile['integrity_history'].pop(0)
        
        # Calculate average integrity over recent history
        avg_integrity = np.mean(profile['integrity_history'])
        
        # Determine if participant is fake
        is_fake = avg_integrity < 50
        
        # Adjust for false positives
        if is_fake and self._check_false_positive(profile):
            is_fake = False
            profile['false_positive_count'] += 1
        
        result = {
            'participant_id': participant_id,
            'is_fake': is_fake,
            'integrity_score': avg_integrity,
            'face_confidence': face_result['confidence'],
            'voice_confidence': voice_result['confidence'],
            'reasons': [],
            'timestamp': datetime.now()
        }
        
        # Add reasons for fake detection
        if face_result['is_fake']:
            result['reasons'].append(f"Face: {face_result['reason']}")
        if voice_result['is_fake']:
            result['reasons'].append(f"Voice: {voice_result['reason']}")
        
        return result
    
    def _check_false_positive(self, profile):
        """
        Check if this might be a false positive
        """
        # If participant has been consistently good, might be temporary issue
        if len(profile['integrity_history']) >= 5:
            recent_good_scores = sum(1 for score in profile['integrity_history'][-5:] if score > 70)
            if recent_good_scores >= 3:
                return True
        
        return False
    
    def get_meeting_statistics(self):
        """
        Get overall meeting statistics
        """
        if not self.participant_profiles:
            return {
                'total_participants': 0,
                'fake_participants': 0,
                'accuracy_rate': 0,
                'false_positive_rate': 0
            }
        
        total_participants = len(self.participant_profiles)
        fake_participants = 0
        total_false_positives = 0
        
        for profile in self.participant_profiles.values():
            if profile['integrity_history']:
                avg_score = np.mean(profile['integrity_history'])
                if avg_score < 50:
                    fake_participants += 1
            
            total_false_positives += profile['false_positive_count']
        
        accuracy_rate = ((total_participants - fake_participants) / total_participants * 100) if total_participants > 0 else 0
        false_positive_rate = (total_false_positives / total_participants * 100) if total_participants > 0 else 0
        
        return {
            'total_participants': total_participants,
            'fake_participants': fake_participants,
            'accuracy_rate': round(accuracy_rate, 2),
            'false_positive_rate': round(false_positive_rate, 2)
        }