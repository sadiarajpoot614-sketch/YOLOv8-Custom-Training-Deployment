import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(layout="wide", page_title="YOLOv8 Object Detection App")

st.title("YOLOv8 Vehicle Detection App")
st.write("Upload an image and let the YOLOv8 model detect vehicles!")

# Model path (best.pt is in the project root folder)
model_path = "best.pt"

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Please ensure the model file exists at '{model_path}'.")
        return None

model = load_model()

if model:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting..."):
                img_np = np.array(image)

                results = model.predict(
                    source=img_np,
                    conf=0.1,
                    iou=0.45
                )

                for r in results:
                    annotated_frame = r.plot()
                    st.image(
                        annotated_frame,
                        caption="Detected Objects",
                        use_container_width=True
                    )

                    st.subheader("Detections")

                    if len(r.boxes) > 0:
                        for box in r.boxes:
                            cls_id = int(box.cls)
                            conf = float(box.conf)
                            class_name = model.names[cls_id]
                            st.write(f"• {class_name}: {conf:.2f}")
                    else:
                        st.write("No objects detected.")
else:
    st.warning("Model could not be loaded. Please check the model path and try again.")
