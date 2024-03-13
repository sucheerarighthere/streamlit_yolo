import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect
import PIL
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Set image
image = Image.open('STAT-Header-Logo-V7.png')
st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# Use st.file_uploader for file upload
uploaded_files = st.file_uploader("Choose .jpg pic ...", type= ["jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

# Check if any file is uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            try:
                # Read and decode the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Run YOLOv5 model on the image
                result = model(image, size=300)
                
                # Extract detected objects' information
                detect_class = result.xyxy[0]
                num_objects_detected = len(detect_class)

                # Draw bounding boxes on the image
                for detection in detect_class:
                    x_min, y_min, x_max, y_max, conf, cls = detection
                    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(image, f'Class: {int(cls)}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Convert image to RGB format
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Display the image with bounding boxes and captions in column 3
                st.image(image_rgb, caption=f'Number of objects detected: {num_objects_detected}', use_column_width=True)

            except Exception as e:
                # Display an error message if an exception occurs during processing
                st.error(f"Error processing file: {e}")
