# navigation/main_navigation.py
import cv2
import time
import argparse
from navigation_controller import NavigationController
from yolo_detector import YOLODetector
from src.Controller import Controller
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

class PupperNavigation:
    def __init__(self, model_path, target_object):
        # Initialize the system
        self.config = Configuration()
        self.controller = Controller(self.config, four_legs_inverse_kinematics)
        self.nav_controller = NavigationController(self.config, self.controller)
        self.detector = YOLODetector(model_path)
        
        # Set target object
        self.target_object = target_object.lower()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        
        # Find target class ID
        self.target_class_id = None
        for class_id, class_name in self.detector.class_names.items():
            if class_name.lower() == self.target_object:
                self.target_class_id = class_id
                break
        
        if self.target_class_id is None:
            raise ValueError(f"Target object '{target_object}' not found in model's classes")
            
    def run(self):
        """Main navigation loop"""
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Find target object
                target_detection = None
                for det in detections:
                    if det[5] == self.target_class_id:
                        target_detection = det[:4]  # Get bbox coordinates
                        break
                
                if target_detection is not None:
                    # Navigate towards the target
                    target_centered = self.nav_controller.navigate_to_target(
                        self.frame_width, self.frame_height, target_detection
                    )
                    
                    if target_centered:
                        print(f"{self.target_object.capitalize()} is centered!")
                        # You might want to stop or perform another action here
                else:
                    print(f"Looking for {self.target_object}...")
                
                # Display the frame with detections
                self._draw_detections(frame, detections)
                cv2.imshow('Navigation', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add a small delay to control the loop rate
                time.sleep(0.1)
                
        finally:
            self.cleanup()
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes on the frame"""
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_name = self.detector.get_class_name(class_id)
            
            # Use different colors for target vs other objects
            color = (0, 255, 0) if class_id == self.target_class_id else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pupper Navigation System')
    parser.add_argument('--target', type=str, required=True,
                      help='Target object to navigate towards (e.g., "door", "chair")')
    parser.add_argument('--model', type=str, default='model/best_int8.tflite',
                      help='Path to the TFLite model')
    args = parser.parse_args()
    
    # Initialize and run navigation
    nav = PupperNavigation(args.model, args.target)
    nav.run()

if __name__ == "__main__":
    main()