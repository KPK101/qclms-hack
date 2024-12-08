import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = 'face_attrib_net.onnx'  # Replace with your model's path
session = ort.InferenceSession(onnx_model_path)

# Get the model's input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model Input Name: {input_name}")
print(f"Model Input Shape: {input_shape}")

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame (resize, normalize, etc. based on your model requirements)
    input_image = cv2.resize(frame, (input_shape[2], input_shape[3]))  # Assuming model expects HxWxC
    input_image = input_image.astype(np.float32)  # Convert to float32
    input_image = np.transpose(input_image, (2, 0, 1))  # Change to CHW format (if required)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    print(f"Input shape = {input_image.shape}")
    # Perform inference
    result = session.run(None, {input_name: input_image})  # Run model and get output
    print('*'*10)
    print(len(result))
    # print(f"Model Output: {result[0]}")
    for i,r in enumerate(result):
        print(i,r.shape)
    print(f"Model out shape: {len(result)}")
    print('*'*10)

    # Process the result (you may need to post-process depending on your model)
    # For instance, displaying output image if model predicts an image
    # or getting classification result if it's a classification model
    # Example: If output is a classification, display the highest confidence label
    # If your model has more complex output, adapt this part accordingly.

    # Show the input frame and model's output (if it's an image or classification result)
    cv2.imshow('Webcam Input', frame)
    if isinstance(result, list) and len(result) > 0:
        # If the result is a list (for example, a classification output)
        # Display the first element (modify as per your model output format)
        output = result[0]
        print(f"Output shape: {output.shape}")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
