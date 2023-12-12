import streamlit as st
from fastai.vision.all import *
from PIL import Image
import io
# Homepage content
st.markdown("""
# Welcome to the Roses Image Classification App

This app uses advanced AI to identify various species of roses from your images. Whether you're a gardening enthusiast or just curious about the beautiful world of roses, our app can help you identify the following species:

- Red Roses
- Floribunda Roses
- Miniature Roses
- Hybrid Tea Roses
- Grandiflora Roses
- Shrub Roses
- English Roses

## How to Use
1. Upload an image of a rose (JPG or PNG format).
2. Wait for the AI to analyze the image.
3. View the classification results, including the species name and the confidence level of the prediction.

## About the Classification Model
Our app is powered by a deep learning model trained specifically on these seven types of roses. It combines accuracy with ease of use to bring you a seamless classification experience.

## Privacy and Data Use
Your privacy is important to us. Uploaded images are only used for classification and are not stored or used for any other purposes.

## Feedback
Love our app? Have suggestions? Contact us at [your email/contact form].

## Acknowledgments
Special thanks to those who contributed to the dataset and development of this app.

Discover the variety and beauty of roses with our app!
""")
# Load your trained model
model = load_learner('resnet50_exported.pkl')

st.title('Roses Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        # Read the uploaded image
        image_data = uploaded_file.read()
        st.image(image_data, caption='Uploaded Image', use_column_width=True)

        # Convert the file to an image and resize
        pil_image = Image.open(io.BytesIO(image_data)).resize((224, 224))

        # Make a prediction
        pred,pred_idx,probs = model.predict(pil_image)
        st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
