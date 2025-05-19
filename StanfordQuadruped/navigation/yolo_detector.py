# navigation/yolo_detector.py
import cv2
import numpy as np
import tensorflow as tf

class YOLODetector:
    def __init__(self, model_path):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model parameters
        self.input_size = (224, 224)  # YOLO input size
        self.confidence_threshold = 0.5
        
        # Class names (update these based on your model's classes)
        self.class_names = {
            0: "door",
            1: "chair",
            2: "table",
            # Add more classes as needed
        }
        
    def preprocess_image(self, frame):
        """Preprocess image for YOLO model"""
        # Resize image
        resized = cv2.resize(frame, self.input_size)
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        return input_data
    
    def detect(self, frame):
        """
        Detect objects in the frame
        Returns list of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # Preprocess image
        input_data = self.preprocess_image(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Process detections
        detections = []
        for detection in output_data[0]:
            confidence = detection[4]
            if confidence > self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                x1 = int(detection[0] * frame.shape[1])
                y1 = int(detection[1] * frame.shape[0])
                x2 = int(detection[2] * frame.shape[1])
                y2 = int(detection[3] * frame.shape[0])
                
                detections.append([x1, y1, x2, y2, confidence, int(detection[5])])
        
        return detections
    
    def get_class_name(self, class_id):
        """Get the name of a class given its ID"""
        return self.class_names.get(class_id, f"unknown_{class_id}")