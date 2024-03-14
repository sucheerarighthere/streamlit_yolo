import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch
import json
import cv2
import detect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect
# ตั้งค่าเพจให้เป็นแบบที่เราต้องการ พื้นหลัง ตัวหนังสือ ใดๆว่าไป
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🔬",  # Setting page icon
    # layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",  # Expanding sidebar by default
)

# ตั้งค่าภาพ
image = Image.open('STAT-Header-Logo-V7.png')
st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')


# Use st.file_uploader for file upload
uploaded_files = st.file_uploader("Choose .jpg pic ...", type=["jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
model  = torch.hub.load('github.com/ultralytics/yolov5','custom', path='best.pt',force_reload=True,source='local', pretrained =Flase)
# Check if any file is uploaded
if uploaded_files:
    # Create columns for layout
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

                # Display bounding boxes without class names and confidence scores in col2
                num_objects_detected = len(detect_class)
                outputpath = 'output.jpg'
                result.render()  # render bbox in image
                for im in result.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)
                    img_ = Image.open(outputpath)
                    col2.image(img_, caption=f'Model Prediction(s)', use_column_width=True)
                    col2.write(f"<h1 style='text-align: center;'>Number of objects detected: {num_objects_detected}</h1>", unsafe_allow_html=True)

                    fig, ax = plt.subplots()

                    # Display the image
                    ax.imshow(image)

                    # Draw bounding boxes on the image
                    for index, row in detect_class.iterrows():
                        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(xmin, ymin, row['name'], color='r')  # เพิ่มชื่อของวัตถุลงบน bounding box

                    # Show the image with bounding boxes
                    plt.show()
            except Exception as e:
                st.write(f"Error: {e}")
