from time import sleep
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import karelPupper
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def main():
    try:
        # Initialize the robot
        print("Initializing robot...")
        myPup = karelPupper.Pupper()
        sleep(2)
        myPup.wakeup()
        sleep(2)

        # Load YOLOv8n TFLite model
        print("Loading TFLite model...")
        model_path = 'yolov8n_int8.tflite'
        if not os.path.exists(model_path):
            print("Error: YOLOv8n TFLite model not found. Please make sure yolov8n_int8.tflite is in the current directory.")
            return

        # Initialize TFLite interpreter with 2 threads
        interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=2)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
        
        # Get model input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Input details:", input_details)
        print("Output details:", output_details)
        
        # Check if model is floating point
        floating_model = input_details[0]["dtype"] == np.float32
        
        # Get input dimensions
        input_height = input_details[0]["shape"][1]
        input_width = input_details[0]["shape"][2]
        print(f"Input shape: {input_height}x{input_width}")

        while True:
            try:
                # Get image from camera
                print("Getting image from camera...")
                frame = myPup.getImage()
                if frame is None:
                    print("Error: Failed to get image from camera")
                    continue
                
                # Convert to PIL Image and resize
                img = Image.fromarray(frame).resize(
                    size=(input_width, input_height),
                    resample=Image.Resampling.LANCZOS)
                
                # Convert to numpy array and add batch dimension
                input_data = np.expand_dims(np.array(img), axis=0)
                
                # Normalize if floating point model
                if floating_model:
                    input_data = (np.float32(input_data) - 0.0) / 255.0
                
                print("Running inference...")
                # Set input tensor
                interpreter.set_tensor(input_details[0]["index"], input_data)
                
                # Run inference
                interpreter.invoke()
                
                # Get output tensor
                output_data = interpreter.get_tensor(output_details[0]["index"])
                results = np.squeeze(output_data).transpose()
                
                # Process detections
                boxes = []
                max_box_count = output_details[0]["shape"][2]
                
                for i in range(max_box_count):
                    raw_box = results[i]
                    center_x = raw_box[0]
                    center_y = raw_box[1]
                    w = raw_box[2]
                    h = raw_box[3]
                    class_scores = raw_box[4:]  # Skip box coordinates
                    
                    # Check for person class (usually index 0) with confidence > 0.5
                    if class_scores[0] > 0.5:  # Assuming person is class 0
                        boxes.append([center_x, center_y, w, h, class_scores[0], 0])
                
                if boxes:
                    # Get the detection with highest confidence
                    best_box = max(boxes, key=lambda x: x[4])
                    center_x, center_y, w, h = best_box[:4]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((center_x - w/2) * frame.shape[1])
                    x2 = int((center_x + w/2) * frame.shape[1])
                    y1 = int((center_y - h/2) * frame.shape[0])
                    y2 = int((center_y + h/2) * frame.shape[0])
                    
                    # Calculate center of detection
                    center_x = (x1 + x2) / 2
                    frame_width = frame.shape[1]
                    
                    # Calculate how far from center (in pixels)
                    offset = center_x - (frame_width / 2)
                    
                    # Convert offset to turn angle (adjust these values based on testing)
                    turn_angle = offset * 0.001  # Scale factor
                    
                    # Move forward and turn to follow
                    if abs(offset) > 50:  # If person is not centered
                        print(f"Turning: {turn_angle}")
                        myPup.turn(turn_angle, 0.5)  # Turn to follow
                    else:
                        print("Moving forward")
                        myPup.forward(0.1, 0.3)  # Move forward slowly
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add confidence score
                    cv2.putText(frame, f'Conf: {best_box[4]:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Person Following', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("Stopping person following...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("Cleaning up...")
        myPup.nap()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 