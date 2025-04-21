import streamlit as st
import cv2
import numpy as np
from PIL import Image
from helper import (
    upload_image_rectangle_detection,
    live_rectangle_detection,
    upload_image_coin_counting,
    count_manyObjects,
    dynamicDetection,
    liveDynamic
)

# Language dictionary
translations = {
    "en": {
        "title": "OpenCV Multiple Objects Counting Using Image Processing",
        "choose_operation": "Choose an Operation:",
        "upload_image_rectangle": "Upload Image - Rectangle Detection",
        "live_rectangle": "Live Rectangle Detection (Webcam)",
        "upload_image_coin": "Upload Image - Coin Counting",
        "upload_image_object": "Upload Image - Object Counting",
        "upload_image_dynamic": "Upload Image - Dynamic Detection",
        "live_dynamic": "Live Dynamic Detection",
        "threshold": "Threshold Value",
        "min_area": "Minimum Area",
        "image_caption": "Detection Objects",
        "select_option": "Select an option"
    },
    "zh": {
        "title": "使用图像处理的OpenCV多对象计数",
        "choose_operation": "选择操作：",
        "upload_image_rectangle": "上传图像 - 矩形检测",
        "live_rectangle": "实时矩形检测（摄像头）",
        "upload_image_coin": "上传图像 - 硬币计数",
        "upload_image_object": "上传图像 - 物体计数",
        "upload_image_dynamic": "上传图像 - 动态检测",
        "live_dynamic": "实时动态检测",
        "threshold": "阈值",
        "min_area": "最小面积",
        "image_caption": "检测对象",
        "select_option": "选择一个选项"
    }
}

# Upload image
def upload_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        return image
    return None

def main():
    # Language selector
    lang = st.sidebar.selectbox("Language / 语言", ("en", "zh"))
    t = translations[lang]

    st.title(t["title"])
    st.image("demo.png", caption=t["image_caption"], use_container_width=True)

    operation_type = st.sidebar.selectbox(
        t["choose_operation"],
        (
            t["select_option"],
            t["upload_image_rectangle"],
            t["live_rectangle"],
            t["upload_image_coin"],
            t["upload_image_object"],
            t["upload_image_dynamic"],
            t["live_dynamic"]
        )
    )

    if operation_type == t["upload_image_rectangle"]:
        image = upload_image()
        if image is not None:
            upload_image_rectangle_detection(image, min_area=500)
    elif operation_type == t["live_rectangle"]:
        live_rectangle_detection()
    elif operation_type == t["upload_image_coin"]:
        image = upload_image()
        if image is not None:
            upload_image_coin_counting(image)
    elif operation_type == t["upload_image_object"]:
        image = upload_image()
        if image is not None:
            threshold_value = st.slider(t["threshold"], min_value=0, max_value=255, value=1)
            min_area_val = st.slider(t["min_area"], min_value=100, max_value=10000, value=4000)
            count_manyObjects(image, threshold_value, min_area_val)
    elif operation_type == t["upload_image_dynamic"]:
        dynamicDetection()
    elif operation_type == t["live_dynamic"]:
        liveDynamic()

if __name__ == "__main__":
    main()
