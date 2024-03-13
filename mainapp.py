# import argparse
# import streamlit as st
# import io
# from PIL import Image
# import numpy as np
# import torch, json, cv2, detect

# # ตั้งค่าเพจให้เป็นแบบที่เราต้องการ พื้นหกลัง ตัวหนังสือ ใดๆว่าไป
# st.set_page_config(page_title="Object Detection",  # Setting page title
#                    page_icon="🔬",  # Setting page icon
#                    layout="wide",  # Setting layout to wide
#                    initial_sidebar_state="expanded",  # Expanding sidebar by default
#                    )

# # ตั้งค่าภาพ
# image = Image.open('STAT-Header-Logo-V7.png')
# st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# # ใช้ st.file_uploader ในการรับไฟล์ทั้งหมด
# uploaded_files = st.file_uploader("Choose .jpg pic ...", type="jpg", accept_multiple_files=True)

# # วนลูปตามไฟล์ที่อัปโหลด
# for uploaded_file in uploaded_files:
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()))
#         image = cv2.imdecode(file_bytes, 1)

#         imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         st.write("")
#         st.write("Detecting...")
#         result = model(imgRGB, size=300)

#         detect_class = result.pandas().xyxy[0]

#         outputpath = 'output.jpg'
#         num_objects_detected = len(detect_class)
#         result.render()  # render bbox in image

#         # แสดงภาพต้นฉบับ
#         col1, col2 = st.beta_columns(2)
#         col1.image(imgRGB, caption='Original Image', use_column_width=True)

#         # แสดง bbox และซ่อน label class
#         for im, pred in zip(result.ims, result.xyxy[0]):
#             im_base64 = Image.fromarray(im)
#             im_base64.save(outputpath)
#             img_ = Image.open(outputpath)

#             col2.image(img_, caption='Model Prediction(s)', use_column_width=True)

#         # แสดงจำนวนโครโมโซม
#         st.write(f"Number of objects detected: {num_objects_detected}")
# =========================================
# import argparse
# import streamlit as st
# import io
# from PIL import Image
# import numpy as np
# import torch
# import cv2
# import detect

# # Set Streamlit page configuration
# st.set_page_config(
#     page_title="Object Detection",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Set image
# image = Image.open('STAT-Header-Logo-V7.png')
# st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# # Use st.file_uploader for file upload
# uploaded_files = st.file_uploader("Choose .jpg pic ...", type="jpg", accept_multiple_files=True)

# # Loop through uploaded files
# for uploaded_file in uploaded_files:
#     if uploaded_file is not None:
#         try:
#             file_bytes = np.asarray(bytearray(uploaded_file.read()))
#             image = cv2.imdecode(file_bytes, 1)

#             imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             st.write("")
#             st.write("Detecting...")
#             result = model(imgRGB, size=300)

#             detect_class = result.pandas().xyxy[0]

#             # Display original image
#             st.image(imgRGB, caption='Original Image', use_column_width=True)

#             # Display bounding boxes and hide label class
#             for im, pred in zip(result.ims, result.xyxy[0]):
#                 im_base64 = Image.fromarray(im)
#                 st.image(im_base64, caption='Model Prediction(s)', use_column_width=True)

#             # Display the number of detected objects
#             num_objects_detected = len(detect_class)
#             st.write(f"Number of objects detected: {num_objects_detected}")

#         except Exception as e:
#             st.error(f"Error processing file: {e}")
# ================================
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
uploaded_files = st.file_uploader("Choose .jpg pic ...", type="jpg", accept_multiple_files=True)

# Loop through uploaded files
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        try:
            # Read and decode the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()))
            image = cv2.imdecode(file_bytes, 1)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display information about the detection process
            st.write("")
            st.write("Detecting...")

            # Run YOLOv5 model on the image
            result = model(imgRGB, size=300)

            # Extract detected objects' information
            detect_class = result.pandas().xyxy[0]

            # Display the original image
            st.image(imgRGB, caption='Original Image', use_column_width=True)

            # # Display bounding boxes without class names and confidence scores
            # for pred in detect_class:
            #     bbox = pred[:4]
            #     st.image(cv2.rectangle(imgRGB.copy(), tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2), use_column_width=True)

            # Display the number of detected objects
            num_objects_detected = len(detect_class)
            st.write(f"Number of objects detected: {num_objects_detected}")

        except Exception as e:
            # Display an error message if an exception occurs during processing
            st.error(f"Error processing file: {e}")

