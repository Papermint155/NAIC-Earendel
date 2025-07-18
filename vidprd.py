import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from collections import deque
import threading

class RealTimeDessertRecognizer:
    def __init__(self, model_dir="model", use_detection=True, use_classification=True):
        """
        Initialize the real-time dessert recognition system
        
        Args:
            model_dir: Directory containing model files
            use_detection: Whether to use YOLO detection model
            use_classification: Whether to use YOLO classification model
        """
        self.model_dir = model_dir
        self.use_detection = use_detection
        self.use_classification = use_classification
        
        # Model paths
        self.yolo_model_path = os.path.join(model_dir, "best.pt")
        self.yolo_cls_path = os.path.join(model_dir, "best-cls.pt")
        
        # Class names from your original code
        self.yolo_names = ['kek_lapis', 'Kuih_Bahulu', 'kuih_kaswi_pandan', 'Kuih_Ketayap',
                          'Kuih_Ubi_Kayu', 'Kuih_Lapis', 'Kuih_Seri_Muka', 'Kuih_Talam', 'Onde_Onde']
        
        self.cls_names = ['Kuih_Ketayap','Kuih_Lapis','Kuih_Seri_Muka','Kuih_Talam',
                         'Kuih_Ubi_Kayu','Onde_Onde','kek_lapis','kuih_kaswi_pandan']
        
        # Correct name mapping
        self.correct_names = {
            "kek_lapis": "Kek Lapis",
            "kuih_kaswi_pandan": "Kuih Kaswi Pandan",
            "Kuih_Ketayap": "Kuih Ketayap",
            "Kuih_Lapis": "Kuih Lapis",
            "Kuih_Seri_Muka": "Kuih Seri Muka",
            "Kuih_Talam": "Kuih Talam",
            "Kuih_Ubi_Kayu": "Kuih Ubi Kayu",
            "Onde_Onde": "Onde-Onde",
            "Kuih_Bahulu": "Kuih Bahulu"
        }
        
        # Initialize models based on selection
        self.yolo_model = None
        self.yolo_cls_model = None
        self.load_models()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Keep last 5 predictions
        self.confidence_threshold = 0.3
        
        # Threading for async processing
        self.frame_queue = deque(maxlen=2)
        self.prediction_result = None
        self.prediction_lock = threading.Lock()
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
    def load_models(self):
        """Load YOLO detection and classification models based on selection"""
        print("Loading models...")
        
        if self.use_detection:
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"YOLO detection model not found at {self.yolo_model_path}")
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                print("✅ YOLO Detection model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading detection model: {e}")
                raise
        
        if self.use_classification:
            if not os.path.exists(self.yolo_cls_path):
                raise FileNotFoundError(f"YOLO classification model not found at {self.yolo_cls_path}")
            try:
                self.yolo_cls_model = YOLO(self.yolo_cls_path)
                print("✅ YOLO Classification model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading classification model: {e}")
                raise
        
        if not self.use_detection and not self.use_classification:
            raise ValueError("At least one model (detection or classification) must be enabled!")
    
    def find_available_cameras(self):
        """Find all available camera indices"""
        available_cameras = []
        
        print("🔍 Scanning for available cameras...")
        
        # Test camera indices 0 to 5
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"  📹 Camera {i}: Available")
                else:
                    print(f"  ❌ Camera {i}: Can't read frames")
            else:
                print(f"  ❌ Camera {i}: Not available")
            cap.release()
        
        return available_cameras
    
    def initialize_camera(self, camera_index):
        """Initialize camera with robust settings"""
        print(f"🎥 Initializing camera {camera_index}...")
        
        # Try different backends
        backends = [
            cv2.CAP_DSHOW,    # DirectShow (Windows)
            cv2.CAP_V4L2,     # Video4Linux2 (Linux)
            cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
            cv2.CAP_ANY       # Any available backend
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera {camera_index} initialized successfully with backend {backend}")
                        
                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                        
                        # Additional settings for external cameras
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        
                        return cap
                cap.release()
            except Exception as e:
                print(f"❌ Backend {backend} failed: {e}")
                continue
        
        return None
    
    def predict_with_yolo_detection(self, frame):
        """Predict using YOLO detection model"""
        if not self.use_detection or self.yolo_model is None:
            return []
            
        results = self.yolo_model(frame, conf=0.25, verbose=False)
        
        detections = []
        result = results[0]
        
        if len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                bbox = boxes.xyxy[i].cpu().numpy()
                
                if class_id < len(self.yolo_names):
                    class_name = self.yolo_names[class_id]
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'model_type': 'detection'
                    })
        
        return detections
    
    def predict_with_yolo_classification(self, frame):
        """Predict using YOLO classification model"""
        if not self.use_classification or self.yolo_cls_model is None:
            return None
            
        results = self.yolo_cls_model(frame, conf=0.25, verbose=False)
        
        result = results[0]
        
        if hasattr(result, 'probs') and result.probs is not None:
            class_probs = result.probs.data.cpu().numpy()
            max_conf_idx = np.argmax(class_probs)
            confidence = float(class_probs[max_conf_idx])
            
            if max_conf_idx < len(self.cls_names):
                class_name = self.cls_names[max_conf_idx]
                
                return {
                    'class_name': class_name,
                    'confidence': confidence,
                    'model_type': 'classification'
                }
        
        return None
    
    def combine_predictions(self, detection_results, classification_result):
        """Combine detection and classification predictions"""
        combined_predictions = []
        
        # If we have detections, use them with higher priority
        if detection_results:
            # Sort by confidence
            detection_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            for detection in detection_results:
                if detection['confidence'] >= self.confidence_threshold:
                    combined_predictions.append(detection)
        
        # Add classification result if it's confident enough
        if classification_result and classification_result['confidence'] >= self.confidence_threshold:
            combined_predictions.append(classification_result)
        
        return combined_predictions
    
    def smooth_predictions(self, predictions):
        """Smooth predictions over time to reduce flickering"""
        if not predictions:
            return None
        
        # Add current predictions to history
        self.prediction_history.append(predictions)
        
        # Count occurrences of each class
        class_counts = {}
        confidence_sums = {}
        
        for frame_predictions in self.prediction_history:
            for pred in frame_predictions:
                class_name = pred['class_name']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    confidence_sums[class_name] = 0
                
                class_counts[class_name] += 1
                confidence_sums[class_name] += pred['confidence']
        
        # Find most frequent class
        if class_counts:
            best_class = max(class_counts.keys(), key=lambda x: class_counts[x])
            avg_confidence = confidence_sums[best_class] / class_counts[best_class]
            
            return {
                'class_name': best_class,
                'confidence': avg_confidence,
                'count': class_counts[best_class]
            }
        
        return None
    
    def draw_predictions(self, frame, predictions, smoothed_prediction):
        """Draw predictions on frame"""
        height, width = frame.shape[:2]
        
        # Draw detection boxes
        if predictions:
            for pred in predictions:
                if 'bbox' in pred:  # Detection result
                    bbox = pred['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{self.correct_names.get(pred['class_name'], pred['class_name'])}: {pred['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw smoothed prediction (main prediction)
        if smoothed_prediction:
            class_name = smoothed_prediction['class_name']
            confidence = smoothed_prediction['confidence']
            display_name = self.correct_names.get(class_name, class_name)
            
            # Main prediction box
            cv2.rectangle(frame, (10, 10), (width-10, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width-10, 80), (0, 255, 255), 2)
            
            # Main prediction text
            cv2.putText(frame, f"Prediction: {display_name}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show active models
        model_info = []
        if self.use_detection:
            model_info.append("Detection")
        if self.use_classification:
            model_info.append("Classification")
        
        model_text = f"Models: {' + '.join(model_info)}"
        cv2.putText(frame, model_text, (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_counter.append(fps)
        avg_fps = np.mean(self.fps_counter)
        self.last_time = current_time
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width-100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame_async(self):
        """Process frames asynchronously"""
        while True:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                
                # Make predictions
                detection_results = self.predict_with_yolo_detection(frame)
                classification_result = self.predict_with_yolo_classification(frame)
                
                # Combine predictions
                combined_predictions = self.combine_predictions(detection_results, classification_result)
                
                # Smooth predictions
                smoothed_prediction = self.smooth_predictions(combined_predictions)
                
                # Update result
                with self.prediction_lock:
                    self.prediction_result = {
                        'predictions': combined_predictions,
                        'smoothed': smoothed_prediction
                    }
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def select_camera(self):
        """Let user select camera from available options"""
        available_cameras = self.find_available_cameras()
        
        if not available_cameras:
            print("❌ No cameras found!")
            return None
        
        if len(available_cameras) == 1:
            print(f"✅ Using camera {available_cameras[0]}")
            return available_cameras[0]
        
        print(f"\n📹 Available cameras: {available_cameras}")
        print("Select a camera:")
        
        for i, cam_idx in enumerate(available_cameras):
            print(f"  {i+1}. Camera {cam_idx}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(available_cameras)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_cameras):
                    selected_camera = available_cameras[choice_idx]
                    print(f"✅ Selected camera {selected_camera}")
                    return selected_camera
                else:
                    print("❌ Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️ Operation cancelled")
                return None
    
    def run_webcam(self, camera_index=None):
        """Run real-time webcam prediction"""
        print("🎥 Starting webcam...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to restart camera")
        
        # Select camera if not specified
        if camera_index is None:
            camera_index = self.select_camera()
            if camera_index is None:
                return
        
        # Initialize camera with robust settings
        cap = self.initialize_camera(camera_index)
        
        if cap is None:
            print("❌ Error: Could not initialize any camera")
            return
        
        # Start async processing thread
        processing_thread = threading.Thread(target=self.process_frame_async, daemon=True)
        processing_thread.start()
        
        screenshot_counter = 0
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print("❌ Error: Could not read frame")
                    print("🔄 Trying to reinitialize camera...")
                    
                    cap.release()
                    time.sleep(1)
                    cap = self.initialize_camera(camera_index)
                    
                    if cap is None:
                        print("❌ Failed to reinitialize camera")
                        break
                    continue
                
                frame_count += 1
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add frame to queue for processing (skip some frames to reduce load)
                if frame_count % 2 == 0 and len(self.frame_queue) < 2:
                    self.frame_queue.append(frame.copy())
                
                # Get current prediction results
                current_result = None
                with self.prediction_lock:
                    if self.prediction_result:
                        current_result = self.prediction_result.copy()
                
                # Draw predictions
                if current_result:
                    frame = self.draw_predictions(
                        frame, 
                        current_result.get('predictions', []), 
                        current_result.get('smoothed')
                    )
                else:
                    # Show loading message
                    cv2.putText(frame, "Loading...", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Real-time Dessert Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_filename = f"screenshot_{screenshot_counter:03d}.jpg"
                    cv2.imwrite(screenshot_filename, frame)
                    print(f"📸 Screenshot saved as {screenshot_filename}")
                    screenshot_counter += 1
                elif key == ord('r'):
                    # Restart camera
                    print("🔄 Restarting camera...")
                    cap.release()
                    time.sleep(1)
                    cap = self.initialize_camera(camera_index)
                    if cap is None:
                        print("❌ Failed to restart camera")
                        break
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🔚 Webcam stopped")

def get_model_choice():
    """Get user's choice for which models to use"""
    print("\n🤖 Model Selection")
    print("=" * 30)
    print("1. YOLO Detection only (best.pt)")
    print("2. YOLO Classification only (best-cls.pt)")
    print("3. Both Detection + Classification (recommended)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                return True, False, "Detection Only"
            elif choice == '2':
                return False, True, "Classification Only"
            elif choice == '3':
                return True, True, "Both Models"
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user")
            return None, None, None

def main():
    """Main function"""
    print("🍰 Real-time Dessert Recognition System")
    print("=" * 50)
    
    try:
        # Get user's model choice
        use_detection, use_classification, choice_description = get_model_choice()
        
        if use_detection is None:  # User cancelled
            return
        
        print(f"\n✅ Selected: {choice_description}")
        
        # Initialize recognizer with user's choice
        recognizer = RealTimeDessertRecognizer(
            use_detection=use_detection,
            use_classification=use_classification
        )
        
        # Start webcam (will auto-detect cameras)
        recognizer.run_webcam()
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Please make sure the required model files are in the 'model' directory:")
        if use_detection:
            print("  - best.pt (YOLO detection model)")
        if use_classification:
            print("  - best-cls.pt (YOLO classification model)")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
