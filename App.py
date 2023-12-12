import streamlit as st
from fastai.vision.all import *
from PIL import Image
import io

# Load your trained model
model = load_learner('export.pkl')


st.title('Roses Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image_data = uploaded_file.read()
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Convert the file to an image
    pil_image = Image.open(io.BytesIO(image_data))

    # Depending on your model, you may need additional image processing here

    # Make a prediction
    pred,pred_idx,probs = model.predict(pil_image)
    st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

