from time import sleep
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import karelPupper
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import cv2

# Constants
BOX_COORD_NUM = 4
SCORE_THRESHOLD = 0.4
INPUT_MEAN = 0.0
INPUT_STD = 255.0
NUM_THREADS = 2

def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def setup_vision_model():
    # Load the TFLite model
    interpreter = tflite.Interpreter(
        model_path="../vision_object_detect/yolov9c_int8.tflite",
        num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    
    # Load labels
    class_labels = load_labels("../vision_object_detect/labels.txt")
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if model is floating point
    floating_model = input_details[0]["dtype"] == np.float32
    
    # Get input dimensions
    input_height = input_details[0]["shape"][1]
    input_width = input_details[0]["shape"][2]
    
    # Get output details
    max_box_count = output_details[0]["shape"][2]
    class_count = output_details[0]["shape"][1] - BOX_COORD_NUM
    
    # Verify labels match model
    if len(class_labels) != class_count:
        raise ValueError(f"Model has {class_count} classes, but {len(class_labels)} labels")
    
    return interpreter, class_labels, input_width, input_height, floating_model, max_box_count

def detect_person(interpreter, input_width, input_height, floating_model, max_box_count, myPup, class_labels):
    try:
        # Get image from Pupper's camera
        frame = myPup.getImage()
        if frame is None:
            print("Error: Failed to get image from camera")
            return None
            
        # Convert to PIL Image and resize
        img = Image.fromarray(frame).resize(
            size=(input_width, input_height),
            resample=Image.BICUBIC)
        
        # Prepare input data
        input_data = np.expand_dims(np.array(img), axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD
        
        # Run inference
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_data)
        interpreter.invoke()
        
        # Get results
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        results = np.squeeze(output_data).transpose()
        
        # Process detections
        best_detections = {}  # Dictionary to store best detection per class
        best_person_detection = None
        best_person_score = 0
        
        for i in range(max_box_count):
            raw_box = results[i]
            center_x = raw_box[0]
            center_y = raw_box[1]
            w = raw_box[2]
            h = raw_box[3]
            class_scores = raw_box[BOX_COORD_NUM:]
            
            # Find best detection for each class
            for class_id, score in enumerate(class_scores):
                if score > 0.1:  # Only consider scores above 0.1
                    class_name = class_labels[class_id]
                    if class_name not in best_detections or score > best_detections[class_name][4]:
                        best_detections[class_name] = [center_x, center_y, w, h, score, class_id]
            
            # Specifically track person detection
            score = class_scores[0]  # Assuming person is class 0
            if score > SCORE_THRESHOLD and score > best_person_score:
                best_person_score = score
                best_person_detection = [center_x, center_y, w, h, score, 0]
        
        # Print best detection for each class
        if best_detections:
            print("\nBest detections in frame:")
            for class_name, detection in best_detections.items():
                _, _, w, h, score, _ = detection
                print(f"{class_name}: Confidence {score:.3f}, Size {w*h:.3f}")
        
        return best_person_detection
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return None

def main():
    # Initialize Pupper
    myPup = karelPupper.Pupper()
    myPup.wakeup()
    sleep(1)
    print("Pupper is awake")
    
    # Setup vision model
    interpreter, class_labels, input_width, input_height, floating_model, max_box_count = setup_vision_model()
    print("Vision model loaded")
    print(f"Available classes: {class_labels}")
    print(f"Detection threshold: {SCORE_THRESHOLD}")
    
    try:
        # Initial scan by turning left
        print("Starting initial scan")
        myPup.turnI(-np.pi/2, 0.4)  # Turn 90 degrees left
        sleep(1)
        
        while True:
            # Detect person
            detection = detect_person(interpreter, input_width, input_height, floating_model, max_box_count, myPup, class_labels)
            
            if detection:
                center_x, center_y, w, h, score, _ = detection
                print(f"\nPerson detected! Confidence: {score:.3f}")
                
                # If person is detected and close enough (based on bounding box size)
                if w * h > 0.3:  # Adjust this threshold based on testing
                    print("Person is close, stopping")
                    myPup.nap()
                    break
                else:
                    # Move towards the person
                    print("Moving towards person")
                    myPup.forward_for_time(1, 0.2)  # Move forward for 1 second
            else:
                # If no person detected, turn slightly to scan
                print("\nNo person detected, scanning...")
                myPup.turnI(np.pi/6, 0.2)  # Turn 30 degrees right
                sleep(0.5)
            
            sleep(0.5)  # Small delay between detections
            
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    finally:
        myPup.nap()
        print("Pupper is resting")

if __name__ == "__main__":
    main() 