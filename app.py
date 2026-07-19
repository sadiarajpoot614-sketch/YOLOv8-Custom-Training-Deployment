
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(layout="wide", page_title="YOLOv8 Object Detection App")

st.title("YOLOv8 Vehicle Detection App")
st.write("Upload an image and let the YOLOv8 model detect vehicles!")

# Path to the exported ONNX model
# Make sure this path is correct relative to where you run app.py
model_path = 'yolov8_vehicles_detection_more_epochs/weights/best.pt'

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Please ensure the model file exists at {model_path} and is accessible.")
        return None

model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        if st.button("Detect Objects"):
            st.write("Detecting...")
            # Convert PIL image to numpy array for YOLO
            img_np = np.array(image)

            # Perform inference
            # Confidence threshold and IoU threshold can be adjusted
            results = model.predict(source=img_np, conf=0.1, iou=0.45)

            # Process and display results
            for r in results:
                # Plot results directly on the image
                annotated_frame = r.plot() # Ultralytics function to draw boxes, labels, etc.
                st.image(annotated_frame, caption='Detected Objects.', use_column_width=True)
                
                # Optional: display detected classes and confidence scores
                st.subheader("Detections:")
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        class_name = model.names[cls_id]
                        st.write(f"- {class_name}: {conf:.2f}")
                else:
                    st.write("No objects detected.")
else:
    st.warning("Model could not be loaded. Please check the path and try again.")



