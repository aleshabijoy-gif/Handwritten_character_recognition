import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# 1. Set up the web page title
st.title("🔤 Handwritten Capital Letter Recognition")
st.write("Upload an image of a handwritten capital English letter to see the prediction.")

# 2. Load the model and dictionary (Cache it so it only loads once)
@st.cache_resource
def load_my_model():
    # Make sure 'alesha_capital_letter_rcgn.h5' is in the same folder
    model = tf.keras.models.load_model('alesha_capital_letter_rcgn.h5')
    
    # Recreating your word_dict mapping (adjust if your indices differ)
    word_dict = {i: chr(65 + i) for i in range(26)} # A-Z mapping
    return model, word_dict

try:
    model, word_dict = load_my_model()
except Exception as e:
    st.error(f"Could not load model. Ensure 'alesha_capital_letter_rcgn.h5' is in this directory. Error: {e}")
    st.stop()

# 3. File Uploader UI Element
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Handle RGB/RGBA conversions safely
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Display the uploaded image to the user
    st.image(image, caption="Uploaded Image", width=250)
    
    # --- Your Exact Proven Preprocessing Pipeline ---
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Otsu Thresholding
    _, img_thresh = cv2.threshold(
        img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    # Bounding Box Crop
    coords = cv2.findNonZero(img_closed)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 4
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, img_closed.shape[1]), min(y + h + pad, img_closed.shape[0])
        img_cropped = img_closed[y1:y2, x1:x2]
    else:
        img_cropped = img_closed

    # Resize to 28x28 matching training shape
    img_resized = cv2.resize(img_cropped, (28, 28), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype('float32') / 255.0
    img_input = np.reshape(img_norm, (1, 28, 28, 1))

    # --- Run Prediction ---
    with st.spinner("Analyzing handwriting..."):
        pred_probs = model.predict(img_input)[0]
        pred_index = np.argmax(pred_probs)
        prediction = word_dict[pred_index]
        confidence = pred_probs[pred_index] * 100

    # --- Display Results ---
    st.success(f"### Prediction: **{prediction}** (Confidence: {confidence:.1f}%)")
    
    # Show Top 3
    st.write("**Top 3 Predictions:**")
    top3 = np.argsort(pred_probs)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        st.write(f"{rank}. Letter **{word_dict[idx]}** — {pred_probs[idx]*100:.1f}%")
