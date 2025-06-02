mport numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


interpreter = tflite.Interpreter(model_path="~school/ee292d/terabot/yolov11_finetuned/best_float16.tflite")
interpreter.allocate_tensors()



input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


img = Image.open('image.jpg').resize((224, 224))  # match imgsz
img_np = np.array(img).astype(np.float32) / 255.0  # normalize
img_np = np.expand_dims(img_np, axis=0)  # add batch dimension

# Set tensor
interpreter.set_tensor(input_details[0]['index'], img_np)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Raw output:", output_data)

