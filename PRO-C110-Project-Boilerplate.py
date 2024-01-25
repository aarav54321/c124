# To Capture Frame
import cv2

# To process image array
import numpy as np

# Import the TensorFlow modules and load the model
import tensorflow as tf

# Replace 'your_model_path' with the actual path to your TensorFlow model file
model = tf.keras.models.load_model('your_model_path')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, (224, 224))  # Adjust dimensions as needed

        # Expand the dimensions
        input_frame = np.expand_dims(frame, axis=0)

        # Normalize it before feeding to the model
        input_frame = input_frame / 255.0  # Assuming normalization range is [0, 255]

        # Get predictions from the model
        predictions = model.predict(input_frame)

        # You can use the predictions as needed

        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)

        # If the space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()