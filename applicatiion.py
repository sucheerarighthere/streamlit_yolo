# import argparse
# import streamlit as st
# import io
# from PIL import Image
# import numpy as np
# import torch
# import cv2
# import detect
# # Set image
# image = Image.open('STAT-Header-Logo-V7.png')
# st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True)

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# # Use st.file_uploader for file upload
# uploaded_files = st.file_uploader("Choose .jpg pic ...", type= "jpeg", "png", 'bmp', 'webp', accept_multiple_files=True)

# for uploaded_file in uploaded_files:
#     if uploaded_file is not None:
#         try:
#             # Read and decode the uploaded image
#             file_bytes = np.asarray(bytearray(uploaded_file.read()))
#             image = cv2.imdecode(file_bytes, 1)
#             imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Display information about the detection process
#             st.write("")
#             st.write("Detecting...")

#             # Run YOLOv5 model on the image
#             result = model(imgRGB, size=300)

#             # Extract detected objects' information
#             detect_class = result.pandas().xyxy[0]

#             # Display the original image
#             st.image(imgRGB, caption='Original Image', use_column_width=True)

#             # Display bounding boxes without class names and confidence scores
#             for pred in detect_class:
#                 bbox = pred[:4]
#                 st.image(cv2.rectangle(imgRGB.copy(), tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2), use_column_width=True)

#             # Display the number of detected objects
#             num_objects_detected = len(detect_class)
#             st.write(f"Number of objects detected: {num_objects_detected}")

#             # Break the loop to process only one uploaded image
#             break

#         except Exception as e:
#             # Display an error message if an exception occurs during processing
#             st.error(f"Error processing file: {e}")
#=================
# import argparse
# import streamlit as st
# import io
# import os
# from PIL import Image
# import numpy as np
# import torch, json , cv2 , detect


# #ตั้งค่าเพจให้เป็นแบบที่เราต้องการ พื้นหกลัง ตัวหนังสือ ใดๆว่าไป
# st.set_page_config(page_title="Object Detection",  # Setting page title
#     page_icon="🔬",     # Setting page icon
#     layout="wide",      # Setting layout to wide
#     initial_sidebar_state="expanded",# Expanding sidebar by default
    
#         )   

# #ตั้งค่าภาพ
# image = Image.open('STAT-Header-Logo-V7.png')
# st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True )

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

# uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
# if uploaded_file is not None:
  
#   file_bytes = np.asarray(bytearray(uploaded_file.read()))
#   image = cv2.imdecode(file_bytes, 1)

#   imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
#   #st.image(imgRGB)

#   st.write("")
#   st.write("Detecting...")
#   result = model(imgRGB, size=300)
  
#   detect_class = result.pandas().xyxy[0] 


#   # #labels, cord_thres = detect_class[:, :].numpy(), detect_class[:, :].numpy()
  
#   # #     xmin       ymin    xmax        ymax          confidence  class    name
#   # #0  148.605362   0.0    1022.523743  818.618286    0.813045      2      turtle
  
#   #  st.code(detect_class[['name', 'xmin','ymin', 'xmax', 'ymax']])
  
  
  
#   # st.success(detect_class)
  
#   outputpath = 'output.jpg'
#   num_objects_detected = len(detect_class)
#   result.render()  # render bbox in image
#   for im in result.ims:
#       im_base64 = Image.fromarray(im)
#       im_base64.save(outputpath)
#       img_ = Image.open(outputpath)
#       st.image(img_, caption='Model Prediction(s)')
#       st.write(f"Number of objects detected: {num_objects_detected}")
  # ====================================
import argparse
import streamlit as st
import io
from PIL import Image
import numpy as np
import torch
import cv2
import detect
import streamlit as st

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

                # Display information about the detection process
                # st.write("")
                # st.write("Detecting...")

                # Run YOLOv5 model on the image
                result = model(imgRGB, size=300)

                # Extract detected objects' information
                detect_class = result.pandas().xyxy[0]

                # Display the original image in col1
                col1.image(imgRGB, caption='Original Image', use_column_width=True)
                col1.write(f"Uploaded File: {uploaded_file.name}")
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
                
            except Exception as e:
                # Display an error message if an exception occurs during processing
                st.error(f"Error processing file: {e}")

  
