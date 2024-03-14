import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect


#ตั้งค่าเพจให้เป็นแบบที่เราต้องการ พื้นหกลัง ตัวหนังสือ ใดๆว่าไป
st.set_page_config(page_title="Object Detection",  # Setting page title
    page_icon="🔬",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",# Expanding sidebar by default
    
        )   

#ตั้งค่าภาพ
image = Image.open('STAT-Header-Logo-V7.png')
st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True )

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

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
#   #labels, cord_thres = detect_class[:, :].numpy(), detect_class[:, :].numpy()
  
#   #     xmin       ymin    xmax        ymax          confidence  class    name
#   #0  148.605362   0.0    1022.523743  818.618286    0.813045      2      turtle
#  # st.code(detect_class[['name', 'xmin','ymin', 'xmax', 'ymax']])

#  #  st.success(detect_class)
  
#   # outputpath = 'output.jpg'
#   # num_objects_detected = len(detect_class)
#   # result.render()  # render bbox in image
#   # for im in result.ims:
#   #     im_base64 = Image.fromarray(im)
#   #     im_base64.save(outputpath)
#   #     img_ = Image.open(outputpath)
#   #     st.image(img_, caption='Model Prediction(s)')
#   #     st.write(f"Number of objects detected: {num_objects_detected}")

#  for index, row in detect_class.iterrows():
#     x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # วาด bounding box

# # แสดงภาพที่มี bounding box บน Streamlit
# st.image(image, caption='Image with Bounding Boxes', use_column_width=True)



# # for i in range(xysocde):
# #     # ดึงข้อมูล bounding box จาก detect_class
# #     bbox = detect_class[i][:4]  # [x_min, y_min, x_max, y_max]

# #     # กำหนดสีและความหนาของเส้น
# #     color = (255, 0, 0)  # สีแดง (BGR)
# #     thickness = 2

# #     # วาด bounding box บนภาพ
# #     image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

# # # แสดงภาพที่ถูกประมวลผลแล้ว
# # cv2.imshow('Detected Objects', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import streamlit as st
# from PIL import Image
# import matplotlib.pyplot as plt

# # ฟังก์ชันสำหรับวาดกรอบ bounding box บนภาพ
# def draw_bounding_boxes(image, detections):
#     img = Image.open(image)
#     plt.imshow(img)
    
#     # วาด bounding box บนภาพ
#     for detection in detections:
#         x_min, y_min, x_max, y_max = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
#         rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2)
#         plt.gca().add_patch(rect)
    
#     # ปรับขนาดและแสดงภาพ
#     plt.axis('off')  # ปิดเส้นแกน
#     st.pyplot()  # แสดงผลภาพใน Streamlit

#     draw_bounding_boxes(uploaded_file, detect_class)
# Use st.file_uploader for file upload
uploaded_files = st.file_uploader("Choose .jpg pic ...", type= ["jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

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
                    col2.image(img_, caption=f'Model Prediction(s)' , use_column_width=True)
                    col2.write(f"<h1 style='text-align: center;'>Number of objects detected: {num_objects_detected}</h1>", unsafe_allow_html=True)
                
            except Exception as e:
                st.write(f"Error: {e}")
