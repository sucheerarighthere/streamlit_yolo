import argparse
import streamlit as st
import io
from PIL import Image
import numpy as np
import torch
import cv2
import detect


# Set image
image = Image.open('STAT-Header-Logo-V7.png')
st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# Use st.file_uploader for file upload
uploaded_files = st.file_uploader("Choose .jpg pic ...", type= ["jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

# Check if any file is uploaded
if uploaded_files:
    # Create columns for layout
    # col1, col2, col3 = st.columns(3)
    col1, col2 = st.columns(2)
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            try:
                # Read and decode the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()))
                image = cv2.imdecode(file_bytes, 1)
                imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Run YOLOv5 model on the image
                result = model(imgRGB, size=300)

                # Extract detected objects' information
                detect_class = result.pandas().xyxy[0]

                # Display the original image in col1
                col1.image(imgRGB, caption='Original Image', use_column_width=True)
                col1.write(f"<h1 style='text-align: center;'>Uploaded File: {uploaded_file.name}</h1>", unsafe_allow_html=True)
                # Display bounding boxes without class names and confidence scores in col
                outputpath = 'output.jpg'
                num_objects_detected = len(detect_class)
                result.render()  # render bbox in image
                for im in result.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)
                    img_ = Image.open(outputpath)
                    col2.image(img_, caption=f'Model Prediction(s)' , use_column_width=True) #Number of objects detected: {num_objects_detected}'
                    num_objects_detected = len(detect_class)
                    col2.write(f"<h1 style='text-align: center;'>Number of objects detected: {num_objects_detected}</h1>", unsafe_allow_html=True)
                    # st.write(f"Number of objects detected: {num_objects_detected}")
                # Display the number of detected objects in col3
                num_objects_detected = len(detect_class)
                # col3.write(f"Number of objects detected: {num_objects_detected}")

                # # Break the loop to process only one uploaded image
                # break
                # Display file name below the original image
  
