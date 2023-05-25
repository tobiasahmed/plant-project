import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time
import board

from adafruit_seesaw.seesaw import Seesaw
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.capture('/home/G4/Desktop/test/image.jpg')
print('Picture saved.')
print()

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='/home/G4/Desktop/test/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = '/home/G4/Desktop/test/image.jpg'
image = Image.open(image_path).resize((224, 224))
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)  # Convert to FLOAT32

# Set the image as input tensor
interpreter.set_tensor(input_details[0]['index'], image)

# Run inference
interpreter.invoke()

# Get the predicted output
output = interpreter.get_tensor(output_details[0]['index'])

# Process the output
predicted_class_index = np.argmax(output)
confidence = output[0][predicted_class_index]

class_labels = ['Black rot', 'Blight', 'Mold', 'Powdery Mildew', 'Rust', 'Scab', 'Spot']

if confidence < 0.7:
    print("No disease")

elif predicted_class_index < len(class_labels):
    predicted_class_label = class_labels[predicted_class_index]
    print("Predicted class:", predicted_class_label)
    print("Confidence:", int(confidence * 100), "%")

i2c_bus = board.I2C()
ss = Seesaw(i2c_bus, addr=0x36)

# read moisture level through capacitive touch pad
moist = ss.moisture_read()

# read temperature from the temperature sensor
temp = ss.get_temp()

if moist < 500:
    print()
    print(" Plant needs watering.", "\n", "Moisture level: ", moist, "\n", "Temperature: ", int(temp), "°C")
else:
    print()
    print(" Plant is watered.", "\n", "Moisture level: ", moist, "\n", "Temperature: ", int(temp), "°C")
