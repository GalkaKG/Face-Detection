import cv2
import numpy as np
import streamlit as st
from mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

# Function to detect faces in an image
def detect_faces(image):
    # Detect faces in the image
    results = detector.detect_faces(image)
    
    # Plot and annotate detected faces
    for result in results:
        bounding_box = result['box']
        keypoints = result['keypoints']
        
        # Draw rectangle around the face
        cv2.rectangle(image, 
                      (bounding_box[0], bounding_box[1]), 
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                      (0, 255, 0), 2)
        
        # Annotate key points
        cv2.circle(image, keypoints['left_eye'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['right_eye'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['nose'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['mouth_left'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['mouth_right'], 2, (0, 155, 255), 2)

    return image

# Streamlit GUI
st.title("Face Detection with MTCNN")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))

    if image is not None:
        # Detect faces
        detected_image = detect_faces(image)

        # Convert image to RGB for displaying with Streamlit
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        
        # Show the image with detected faces
        st.image(detected_image, caption='Detected Faces', use_column_width=True)
    else:
        st.error("Error processing the image. Please try a different one.")
